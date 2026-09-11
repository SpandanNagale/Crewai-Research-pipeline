import logging
import re
import uuid
from typing import List, Optional, Tuple, Type, TypeVar

import openai
from crewai import Crew, Process
from pydantic import BaseModel

from agents.classifier import get_classifier
from agents.critic import get_critic
from agents.evidence_comparator import get_evidence_comparator
from agents.planner import get_planner
from agents.researcher import get_researcher
from agents.writer import get_writer
from db import repository
from db.session import init_db
from guardrails.input_guardrail import GuardrailBlockedError, validate_topic
from llm.router_llm import classify_provider_exception
from memory.reuse import KnowledgeReuseChecker
from observability.context import clear_run, current_run, emit_event, set_stage, start_run
from observability.errors import ErrorCode
from observability.node_state import NodeState
from tasks.classify_task import get_classify_task
from tasks.compare_task import get_compare_task
from tasks.groundedness_task import get_groundedness_task
from tasks.plan_task import get_plan_task
from tasks.research_task import get_research_task
from tasks.write_task import get_write_task
from tools.file_tool import save_report

GROUNDEDNESS_THRESHOLD = 0.7
MAX_REWRITE_ATTEMPTS = 2

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class PipelineStageError(RuntimeError):
    """Raised when a pipeline stage (an LLM call or its parsing) fails outright."""

    def __init__(self, stage: str, detail: str, error_code: ErrorCode = ErrorCode.INTERNAL):
        self.stage = stage
        self.detail = detail
        self.error_code = error_code
        super().__init__(f"Pipeline stage '{stage}' failed: {detail}")


def _run_single_task_crew(agent, task):
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
    return crew.kickoff()


_PROVIDER_UNAVAILABLE_EXCEPTIONS = (
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.NotFoundError,
    openai.InternalServerError,
)

# Kept in sync with llm/router_llm.py's _GENERIC_STATUS_ERROR_ADVANCE_CODES: status
# codes that surface as the generic openai.APIStatusError base class (not one of the
# specific subclasses above) but are still genuine provider-availability trouble, e.g.
# Groq's 413 "tokens per minute" limit. Deliberately excludes deterministic
# client-request errors (400/409/422) that already have their own specific subclass —
# those are a real bug in what this pipeline sent, not a provider being unavailable, so
# they take the "soft retry, then placeholder finding" path instead.
_GENERIC_STATUS_ERROR_UNAVAILABLE_CODES = frozenset({408, 413, 503, 529})


def _is_provider_unavailable_error(e: Exception) -> bool:
    """
    True for genuine provider-availability failures (bad/missing key, rate
    limit, timeout, can't reach the endpoint, unknown model) — these should
    fail the stage immediately, retrying won't help. False for softer,
    often-transient issues (a local model hitting its tool-call iteration cap
    and returning an empty final answer, malformed JSON, etc.) which are
    worth a retry / graceful per-item fallback instead of aborting the whole
    pipeline run.
    """
    if isinstance(e, _PROVIDER_UNAVAILABLE_EXCEPTIONS):
        return True
    if isinstance(e, openai.APIStatusError):
        return e.status_code in _GENERIC_STATUS_ERROR_UNAVAILABLE_CODES
    return False


def _extract_pydantic(crew_output, model: Type[T], stage: str) -> T:
    """
    Best-effort extraction of a structured Pydantic result from a CrewOutput.

    Tasks intentionally do NOT use crewai's output_pydantic here: on Ollama,
    output_pydantic routes through the OpenAI-compatible client's
    .beta.chat.completions.parse() (constrained/json_schema decoding), which
    ignores the num_ctx override and is hard-capped at ~4096 total tokens
    regardless of context window settings — verified directly against the
    Ollama API. Plain-text completions respect num_ctx correctly, so tasks
    ask for raw JSON in the prompt and we parse it here instead, which also
    works uniformly across all three providers.
    """
    if getattr(crew_output, "pydantic", None) is not None:
        return crew_output.pydantic

    raw = str(crew_output).strip()
    # Strip ```json ... ``` / ``` ... ``` fences some models wrap JSON in
    # despite being told not to.
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1).strip()

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return model.model_validate_json(match.group(0))
        except Exception as e:
            raise PipelineStageError(
                stage, f"could not parse structured output: {e}", ErrorCode.SCHEMA_VALIDATION_FAILED
            ) from e

    raise PipelineStageError(
        stage, "LLM did not return the expected structured output", ErrorCode.SCHEMA_VALIDATION_FAILED
    )


def _parse_sub_questions(text: str) -> List[str]:
    questions = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r"^[\-\*\d]+[\.\)]?\s*", "", line).strip()
        if cleaned:
            questions.append(cleaned)
    return questions


def _extract_conclusion_section(report_markdown: str) -> str:
    match = re.search(
        r"#{1,3}\s*\**Conclusion\**.*?\n(.*?)(?=\n#{1,3}\s|\Z)",
        report_markdown,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        text = match.group(1).strip()
        if text:
            return text
    # Fallback: no Conclusion heading found, use the whole report (truncated)
    return report_markdown.strip()[:1500]


class ResearchCrew:
    STAGES = ["planner", "researcher", "evidence_comparator", "classifier", "writer", "critic"]

    def __init__(self):
        init_db()
        self._reuse_checker = KnowledgeReuseChecker()
        self._STAGE_RUNNERS = {
            "planner": self._dispatch_planner,
            "researcher": self._dispatch_researcher,
            "evidence_comparator": self._dispatch_evidence_comparator,
            "classifier": self._dispatch_classifier,
            "writer": self._dispatch_writer,
            "critic": self._dispatch_critic,
        }

    def run(self, topic: str, force_rerun: bool = False) -> str:
        """Backward-compatible entry point: returns just the final report text."""
        return self.run_with_meta(topic, force_rerun=force_rerun)["report"]

    def run_with_meta(
        self,
        topic: str,
        force_rerun: bool = False,
        run_id: Optional[str] = None,
        resume_question_id: Optional[int] = None,
    ) -> dict:
        run_id = run_id or str(uuid.uuid4())
        start_run(run_id)
        try:
            emit_event("run_started", message=f"topic={topic!r}")

            try:
                validate_topic(topic)
            except GuardrailBlockedError as e:
                emit_event("guardrail_fired", message=e.reason, detail=e.reason)
                raise PipelineStageError("input", e.reason, ErrorCode.GUARDRAIL_BLOCKED)

            if resume_question_id is None:
                if not force_rerun:
                    reused = self._try_reuse(topic)
                    if reused is not None:
                        reused["run_id"] = run_id
                        emit_event("run_finished", cache_hit=True, message="reused prior research")
                        return reused

                question_id = repository.create_question(topic)
                logger.info("Created research_question id=%s for topic=%r", question_id, topic)
                repository.create_pipeline_run(run_id, question_id, topic)
                stage_outputs: dict = {}
                start_index = 0
            else:
                question_id = resume_question_id
                repository.reattach_pipeline_run(run_id, question_id)
                stage_outputs = self._load_resumable_state(question_id)
                start_index = self._first_unresolved_stage_index(run_id)

            for i, stage in enumerate(self.STAGES):
                set_stage(stage)
                if i < start_index:
                    repository.set_stage_state(run_id, stage, NodeState.SKIPPED_CACHED.value)
                    continue

                repository.set_stage_state(run_id, stage, NodeState.RUNNING.value)
                emit_event("stage_started")
                try:
                    stage_outputs = self._STAGE_RUNNERS[stage](topic, question_id, stage_outputs)
                except PipelineStageError as e:
                    attempts = self._collect_attempt_history()
                    repository.set_stage_state(
                        run_id, stage, NodeState.FAILED.value, detail=e.detail, attempts=attempts
                    )
                    emit_event("stage_failed", detail=e.detail)
                    for remaining in self.STAGES[i + 1:]:
                        repository.set_stage_state(run_id, remaining, NodeState.NOT_REACHED.value)
                    repository.set_run_status(run_id, "failed")
                    raise

                if stage_outputs.pop("_skipped_cached", False):
                    state = NodeState.SKIPPED_CACHED
                elif stage_outputs.pop("_degraded", False):
                    state = NodeState.DEGRADED
                else:
                    state = NodeState.SUCCEEDED
                attempts = self._collect_attempt_history()
                repository.set_stage_state(run_id, stage, state.value, attempts=attempts)
                emit_event("stage_finished", node_state=state)

            findings = stage_outputs["findings"]
            contradictions = stage_outputs["contradictions"]
            report = stage_outputs["report"]

            conclusion_text = _extract_conclusion_section(report)
            conclusion_id = repository.add_conclusion(
                question_id, conclusion_text, [f["id"] for f in findings]
            )

            filepath = save_report(report, topic)
            logger.info("Report saved to: %s", filepath)

            self._index_for_reuse(question_id, topic, conclusion_text, findings, conclusion_id)
            repository.set_run_status(run_id, "succeeded")
            emit_event("run_finished")

            sub_questions = stage_outputs.get("sub_questions", [])
            unresolved = stage_outputs.get("unresolved_sub_questions", [])
            return {
                "report": report,
                "question_id": question_id,
                "run_id": run_id,
                "conclusion_id": conclusion_id,
                "reused": False,
                "saved_to": filepath,
                "contradictions_count": len(contradictions),
                "findings_count": len(findings),
                "groundedness_score": stage_outputs.get("groundedness_score"),
                "sub_questions_count": len(sub_questions),
                "unresolved_sub_questions_count": len(unresolved),
            }
        finally:
            clear_run()

    def _collect_attempt_history(self) -> List[dict]:
        ctx = current_run()
        if ctx is None:
            return []
        return [e.model_dump(mode="json") for e in ctx.stage_events if e.event_type == "llm_attempt"]

    def _load_resumable_state(self, question_id: int) -> dict:
        # unresolved_sub_questions isn't DB-backed (unlike findings/contradictions),
        # so a resume that skips past a succeeded researcher stage won't recover the
        # original run's unresolved list -- an accepted small gap, not a correctness
        # issue (the report just won't repeat the "Unresolved Questions" note).
        question = repository.get_question(question_id)
        return {
            "sub_questions": question["sub_questions"] if question else [],
            "findings": repository.get_findings_list(question_id),
            "contradictions": repository.get_contradictions(question_id),
            "unresolved_sub_questions": [],
        }

    def _first_unresolved_stage_index(self, run_id: str) -> int:
        states = {s["stage"]: s["state"] for s in repository.get_stage_states(run_id)}
        resolved = {"succeeded", "degraded", "skipped_cached"}
        for i, stage in enumerate(self.STAGES):
            if states.get(stage) not in resolved:
                return i
        return len(self.STAGES)

    # ── Stage dispatch wrappers ─────────────────────────────────────
    # Each wrapper adapts one _run_*_stage method's own (differing) signature to
    # the uniform (topic, question_id, stage_outputs) -> stage_outputs shape the
    # dispatch loop above calls through self._STAGE_RUNNERS.

    def _dispatch_planner(self, topic, question_id, stage_outputs):
        sub_questions = self._run_planning_stage(topic)
        repository.set_sub_questions(question_id, sub_questions)
        stage_outputs["sub_questions"] = sub_questions
        return stage_outputs

    def _dispatch_researcher(self, topic, question_id, stage_outputs):
        findings, degraded, unresolved, all_cached = self._run_research_stage(
            question_id, stage_outputs["sub_questions"]
        )
        stage_outputs["findings"] = findings
        stage_outputs["unresolved_sub_questions"] = unresolved
        stage_outputs["_degraded"] = degraded
        stage_outputs["_skipped_cached"] = all_cached
        return stage_outputs

    def _dispatch_evidence_comparator(self, topic, question_id, stage_outputs):
        contradictions = self._run_comparison_stage(question_id, stage_outputs["findings"])
        stage_outputs["contradictions"] = contradictions
        return stage_outputs

    def _dispatch_classifier(self, topic, question_id, stage_outputs):
        findings = self._run_classification_stage(question_id, stage_outputs["findings"])
        stage_outputs["findings"] = findings
        return stage_outputs

    def _dispatch_writer(self, topic, question_id, stage_outputs):
        report = self._run_writing_stage(
            topic, stage_outputs["findings"], stage_outputs["contradictions"],
            stage_outputs.get("unresolved_sub_questions", []),
        )
        stage_outputs["report"] = report
        return stage_outputs

    def _dispatch_critic(self, topic, question_id, stage_outputs):
        """
        Groundedness gate: score every claim in the report against this
        question's DB findings, and if the score is below threshold, feed the
        specific objections back to the Writer for a bounded rewrite
        (max MAX_REWRITE_ATTEMPTS extra Writer calls) before giving up.
        """
        report = stage_outputs["report"]
        findings = stage_outputs["findings"]
        contradictions = stage_outputs["contradictions"]
        unresolved = stage_outputs.get("unresolved_sub_questions", [])

        parsed = self._run_groundedness_check(report, findings)
        attempts = 0
        while parsed.overall_score < GROUNDEDNESS_THRESHOLD and attempts < MAX_REWRITE_ATTEMPTS:
            objections = [c.objection for c in parsed.claims if not c.grounded and c.objection]
            emit_event(
                "guardrail_fired",
                message=f"groundedness {parsed.overall_score:.2f} below threshold, rewriting",
                detail="; ".join(objections) or "no specific objections returned",
            )
            report = self._run_writing_stage(topic, findings, contradictions, unresolved, feedback=objections)
            parsed = self._run_groundedness_check(report, findings)
            attempts += 1

        repository.replace_claim_groundings(question_id, [c.model_dump() for c in parsed.claims])
        stage_outputs["report"] = report
        stage_outputs["groundedness_score"] = parsed.overall_score
        stage_outputs["_degraded"] = parsed.overall_score < GROUNDEDNESS_THRESHOLD
        return stage_outputs

    # ── Stages ──────────────────────────────────────────────────────

    def _try_reuse(self, topic: str) -> Optional[dict]:
        try:
            match = self._reuse_checker.find_similar(topic)
        except Exception as e:
            logger.warning("Knowledge reuse check failed, proceeding with a fresh run: %s", e)
            return None
        if match is None:
            return None

        logger.info(
            "Knowledge reuse hit for topic=%r (similarity=%s, question_id=%s)",
            topic,
            match.get("similarity"),
            match["question"]["id"],
        )
        report_lines = [
            f"# Research Report: {match['question']['topic']}",
            "",
            f"*Reused from prior research (similarity={match.get('similarity')}, "
            f"originally run {match['question']['created_at']}). "
            "Pass force_rerun=True / --force to run fresh research instead.*",
            "",
            "---",
            "",
        ]
        for entry in match["conclusions"]:
            report_lines.append(f"## Conclusion\n{entry['conclusion']['text']}\n")
            for f in entry["findings"]:
                report_lines.append(
                    f"- [{f['id']}] ({f.get('category') or 'uncategorized'}) {f['text']} "
                    f"— {f.get('source_title') or 'unknown'} ({f.get('source_url') or 'n/a'})"
                )
            report_lines.append("")
        if match["contradictions"]:
            report_lines.append("## Contradictions & Disagreements")
            for c in match["contradictions"]:
                report_lines.append(f"- {c['explanation']}")

        return {
            "report": "\n".join(report_lines),
            "question_id": match["question"]["id"],
            "reused": True,
            "similarity": match.get("similarity"),
            "saved_to": None,
            "contradictions_count": len(match["contradictions"]),
            "findings_count": sum(len(e["findings"]) for e in match["conclusions"]),
            # A reused answer is treated as fully resolved -- the original run's
            # per-sub-question outcome isn't tracked in the trace this is built from.
            "sub_questions_count": len(match["question"].get("sub_questions") or []),
            "unresolved_sub_questions_count": 0,
        }

    def _run_planning_stage(self, topic: str) -> List[str]:
        planner = get_planner()
        plan_task = get_plan_task(topic, planner)
        try:
            result = _run_single_task_crew(planner, plan_task)
        except Exception as e:
            logger.error("Planner stage failed: %s", e)
            raise PipelineStageError("planner", str(e), classify_provider_exception(e)) from e

        sub_questions = _parse_sub_questions(str(result))
        if not sub_questions:
            raise PipelineStageError("planner", "no sub-questions were produced", ErrorCode.SCHEMA_VALIDATION_FAILED)
        return sub_questions

    def _try_subquestion_cache(self, question_id: int, sub_question: str) -> Optional[List[int]]:
        """
        Sub-question-granularity cache hit: copies a prior question's findings
        for a semantically-matching sub-question into this question instead of
        re-running the Researcher. Returns None (never raises) on a miss, a
        lookup failure, or if this crew has no reuse checker (e.g. a unit test
        constructed via ResearchCrew.__new__).
        """
        checker = getattr(self, "_reuse_checker", None)
        if checker is None:
            return None
        try:
            hit = checker.find_similar_subquestion(sub_question)
        except Exception as e:
            logger.warning("Sub-question cache lookup failed, proceeding fresh: %s", e)
            return None
        if hit is None:
            return None

        copied_ids = []
        for finding_id in hit["finding_ids"]:
            source_finding = repository.get_finding_by_id(finding_id)
            if source_finding is None:
                continue
            source_id = repository.get_or_create_source(
                question_id, source_finding["source_url"], source_finding["source_title"] or ""
            )
            copied_ids.append(repository.add_finding(
                question_id, source_id, source_finding["text"], sub_question, source_finding.get("category")
            ))
        if not copied_ids:
            return None

        tokens_saved = None
        if hit.get("prompt_tokens") is not None or hit.get("completion_tokens") is not None:
            tokens_saved = (hit.get("prompt_tokens") or 0) + (hit.get("completion_tokens") or 0)
        emit_event(
            "cache_hit",
            message=f"sub-question cache hit (similarity={hit['similarity']})",
            cache_hit=True,
            tokens_saved=tokens_saved,
            cost_avoided_usd=hit.get("cost_usd"),
            seconds_saved=(hit["latency_ms"] / 1000) if hit.get("latency_ms") is not None else None,
        )
        return copied_ids

    def _index_subquestion_for_reuse(
        self, sub_question: str, finding_ids: List[int],
        latency_ms: Optional[float], prompt_tokens: Optional[int],
        completion_tokens: Optional[int], cost_usd: Optional[float],
    ) -> None:
        checker = getattr(self, "_reuse_checker", None)
        if checker is None:
            return
        try:
            checker.index_sub_question(sub_question, finding_ids, latency_ms, prompt_tokens, completion_tokens, cost_usd)
        except Exception as e:
            logger.warning("Failed to index sub-question for future cache reuse: %s", e)

    @staticmethod
    def _capture_llm_usage_since(start_len: int):
        """
        Real (not estimated) aggregate latency/tokens/cost of every llm_attempt
        event emitted since start_len on the active run's event window -- used
        to record what a sub-question's Researcher call actually cost, so a
        future cache hit can report a genuine "cost avoided" rather than a
        fabricated one.
        """
        ctx = current_run()
        if ctx is None:
            return None, None, None, None
        window = [
            e for e in ctx.stage_events[start_len:]
            if e.event_type == "llm_attempt" and not e.cache_hit
        ]
        if not window:
            return None, None, None, None
        latency_ms = sum(e.latency_ms or 0.0 for e in window)
        prompt_tokens = sum(e.prompt_tokens for e in window if e.prompt_tokens is not None) or None
        completion_tokens = sum(e.completion_tokens for e in window if e.completion_tokens is not None) or None
        costs = [e.cost_usd for e in window]
        cost_usd = sum(costs) if costs and all(c is not None for c in costs) else None
        return latency_ms, prompt_tokens, completion_tokens, cost_usd

    def _run_research_stage(
        self, question_id: int, sub_questions: List[str]
    ) -> Tuple[List[dict], bool, List[str], bool]:
        """
        Runs one small Researcher call per sub-question rather than one call
        covering all of them. A single call asking a local 8B model to search,
        then emit a large JSON blob covering every sub-question, reliably ran
        out of steam mid-generation; per-sub-question calls keep each ask (and
        each JSON response) small enough to complete. A sub-question whose
        response fails to parse (or the agent returns an empty answer, e.g.
        after hitting its tool-call iteration cap) is recorded as unresolved
        after a retry, instead of aborting the whole run -- and instead of
        being inserted as a fake Finding with source "none" (defect #7): an
        unresolved sub-question never reaches the Writer as if it were real
        content, it's reported separately (see get_write_task's
        unresolved_sub_questions param). A genuine provider-availability error
        (bad/missing key, rate limit, timeout, unreachable endpoint) still
        propagates immediately as a PipelineStageError — retrying that would
        just waste time.

        Before running each sub-question fresh, checks the sub-question
        semantic cache (_try_subquestion_cache) for a prior question's
        matching findings -- this is what makes reuse partial rather than
        all-or-nothing at the whole-topic level. Returns a 4th value, True
        only when every sub-question was served from that cache (no Researcher
        call made at all), so the dispatch loop can mark the whole stage
        skipped_cached instead of succeeded.
        """
        from tasks.schemas import ResearchOutput

        researcher = None  # built lazily -- never constructed on a fully-cached run
        any_succeeded = False
        all_from_cache = bool(sub_questions)
        unresolved: List[str] = []

        for sub_question in sub_questions:
            copied_ids = self._try_subquestion_cache(question_id, sub_question)
            if copied_ids:
                any_succeeded = True
                continue
            all_from_cache = False
            if researcher is None:
                researcher = get_researcher()

            parsed = None
            last_error = None
            ctx = current_run()
            usage_start = len(ctx.stage_events) if ctx is not None else 0
            for attempt in range(2):
                research_task = get_research_task(researcher, sub_question)
                try:
                    result = _run_single_task_crew(researcher, research_task)
                except Exception as e:
                    if _is_provider_unavailable_error(e):
                        logger.error("Researcher stage failed (provider unavailable): %s", e)
                        raise PipelineStageError("researcher", str(e), classify_provider_exception(e)) from e
                    last_error = e
                    logger.warning(
                        "Researcher call for sub-question %r raised an error "
                        "(attempt %d/2): %s",
                        sub_question,
                        attempt + 1,
                        e,
                    )
                    continue

                try:
                    parsed = _extract_pydantic(result, ResearchOutput, "researcher")
                    break
                except PipelineStageError as e:
                    last_error = e
                    logger.warning(
                        "Researcher output for sub-question %r failed to parse "
                        "(attempt %d/2): %s",
                        sub_question,
                        attempt + 1,
                        e,
                    )

            if parsed is not None and parsed.findings:
                any_succeeded = True
                new_finding_ids = []
                for item in parsed.findings:
                    source_id = repository.get_or_create_source(
                        question_id, item.source_url, item.source_title
                    )
                    new_finding_ids.append(repository.add_finding(
                        question_id, source_id, item.finding_text, sub_question
                    ))
                latency_ms, prompt_tok, completion_tok, cost_usd = self._capture_llm_usage_since(usage_start)
                self._index_subquestion_for_reuse(
                    sub_question, new_finding_ids, latency_ms, prompt_tok, completion_tok, cost_usd
                )
            else:
                logger.warning(
                    "Giving up on sub-question %r after retries, marking unresolved: %s",
                    sub_question,
                    last_error,
                )
                unresolved.append(sub_question)

        if not any_succeeded:
            raise PipelineStageError(
                "researcher", "no sub-question produced a parseable finding",
                ErrorCode.SCHEMA_VALIDATION_FAILED,
            )

        findings = repository.get_findings_list(question_id)
        degraded = bool(unresolved)
        return findings, degraded, unresolved, all_from_cache

    def _run_comparison_stage(self, question_id: int, findings: List[dict]) -> List[dict]:
        comparator = get_evidence_comparator()
        grouped = repository.get_findings_grouped(question_id)
        compare_task = get_compare_task(comparator, grouped)
        try:
            result = _run_single_task_crew(comparator, compare_task)
        except Exception as e:
            logger.error("EvidenceComparator stage failed: %s", e)
            raise PipelineStageError("evidence_comparator", str(e), classify_provider_exception(e)) from e

        from tasks.schemas import ComparisonOutput

        parsed = _extract_pydantic(result, ComparisonOutput, "evidence_comparator")

        valid_ids = {f["id"] for f in findings}
        for c in parsed.contradictions:
            if c.finding_a_id in valid_ids and c.finding_b_id in valid_ids:
                repository.add_contradiction(c.finding_a_id, c.finding_b_id, c.explanation)
            else:
                logger.warning(
                    "Skipping contradiction referencing unknown finding ids: %s / %s",
                    c.finding_a_id,
                    c.finding_b_id,
                )

        return repository.get_contradictions(question_id)

    def _run_classification_stage(self, question_id: int, findings: List[dict]) -> List[dict]:
        classifier = get_classifier()
        classify_task = get_classify_task(classifier, findings)
        try:
            result = _run_single_task_crew(classifier, classify_task)
        except Exception as e:
            logger.error("Classifier stage failed: %s", e)
            raise PipelineStageError("classifier", str(e), classify_provider_exception(e)) from e

        from tasks.schemas import ClassificationOutput

        parsed = _extract_pydantic(result, ClassificationOutput, "classifier")

        valid_ids = {f["id"] for f in findings}
        for item in parsed.items:
            if item.finding_id in valid_ids:
                repository.set_finding_category(item.finding_id, item.category)
            else:
                logger.warning("Skipping category for unknown finding id: %s", item.finding_id)

        return repository.get_findings_list(question_id)

    def _run_writing_stage(
        self,
        topic: str,
        findings: List[dict],
        contradictions: List[dict],
        unresolved_sub_questions: Optional[List[str]] = None,
        feedback: Optional[List[str]] = None,
    ) -> str:
        writer = get_writer()
        write_task = get_write_task(writer, topic, findings, contradictions, unresolved_sub_questions, feedback)
        try:
            result = _run_single_task_crew(writer, write_task)
        except Exception as e:
            logger.error("Writer stage failed: %s", e)
            raise PipelineStageError("writer", str(e), classify_provider_exception(e)) from e
        return str(result)

    def _run_groundedness_check(self, report: str, findings: List[dict]):
        from tasks.schemas import GroundednessOutput

        critic = get_critic()
        task = get_groundedness_task(critic, report, findings)
        try:
            result = _run_single_task_crew(critic, task)
        except Exception as e:
            logger.error("Critic stage failed: %s", e)
            raise PipelineStageError("critic", str(e), classify_provider_exception(e)) from e
        return _extract_pydantic(result, GroundednessOutput, "critic")

    def _index_for_reuse(self, question_id, topic, conclusion_text, findings, conclusion_id):
        try:
            self._reuse_checker.index_question(question_id, topic, conclusion_text)
            self._reuse_checker.index_conclusion(conclusion_id, question_id, conclusion_text)
            for f in findings:
                self._reuse_checker.index_finding(f["id"], question_id, f["text"])
        except Exception as e:
            logger.warning("Failed to index question for future knowledge reuse: %s", e)
