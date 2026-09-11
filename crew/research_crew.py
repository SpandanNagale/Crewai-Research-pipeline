import logging
import re
import uuid
from typing import List, Optional, Tuple, Type, TypeVar

import openai
from crewai import Crew, Process
from pydantic import BaseModel

from agents.classifier import get_classifier
from agents.evidence_comparator import get_evidence_comparator
from agents.planner import get_planner
from agents.researcher import get_researcher
from agents.writer import get_writer
from db import repository
from db.session import init_db
from llm.router_llm import classify_provider_exception
from memory.reuse import KnowledgeReuseChecker
from observability.context import clear_run, current_run, emit_event, set_stage, start_run
from observability.errors import ErrorCode
from observability.node_state import NodeState
from tasks.classify_task import get_classify_task
from tasks.compare_task import get_compare_task
from tasks.plan_task import get_plan_task
from tasks.research_task import get_research_task
from tasks.write_task import get_write_task
from tools.file_tool import save_report

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
    STAGES = ["planner", "researcher", "evidence_comparator", "classifier", "writer"]

    def __init__(self):
        init_db()
        self._reuse_checker = KnowledgeReuseChecker()
        self._STAGE_RUNNERS = {
            "planner": self._dispatch_planner,
            "researcher": self._dispatch_researcher,
            "evidence_comparator": self._dispatch_evidence_comparator,
            "classifier": self._dispatch_classifier,
            "writer": self._dispatch_writer,
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

                state = NodeState.DEGRADED if stage_outputs.pop("_degraded", False) else NodeState.SUCCEEDED
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

            return {
                "report": report,
                "question_id": question_id,
                "run_id": run_id,
                "conclusion_id": conclusion_id,
                "reused": False,
                "saved_to": filepath,
                "contradictions_count": len(contradictions),
                "findings_count": len(findings),
            }
        finally:
            clear_run()

    def _collect_attempt_history(self) -> List[dict]:
        ctx = current_run()
        if ctx is None:
            return []
        return [e.model_dump(mode="json") for e in ctx.stage_events if e.event_type == "llm_attempt"]

    def _load_resumable_state(self, question_id: int) -> dict:
        question = repository.get_question(question_id)
        return {
            "sub_questions": question["sub_questions"] if question else [],
            "findings": repository.get_findings_list(question_id),
            "contradictions": repository.get_contradictions(question_id),
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
        findings, degraded = self._run_research_stage(question_id, stage_outputs["sub_questions"])
        stage_outputs["findings"] = findings
        stage_outputs["_degraded"] = degraded
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
        report = self._run_writing_stage(topic, stage_outputs["findings"], stage_outputs["contradictions"])
        stage_outputs["report"] = report
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

    def _run_research_stage(self, question_id: int, sub_questions: List[str]) -> Tuple[List[dict], bool]:
        """
        Runs one small Researcher call per sub-question rather than one call
        covering all of them. A single call asking a local 8B model to search,
        then emit a large JSON blob covering every sub-question, reliably ran
        out of steam mid-generation; per-sub-question calls keep each ask (and
        each JSON response) small enough to complete. A sub-question whose
        response fails to parse (or the agent returns an empty answer, e.g.
        after hitting its tool-call iteration cap) gets a placeholder finding
        after a retry, instead of aborting the whole run. A genuine
        provider-availability error (bad/missing key, rate limit, timeout,
        unreachable endpoint) still propagates immediately as a
        PipelineStageError — retrying that would just waste time.
        """
        from tasks.schemas import ResearchOutput

        researcher = get_researcher()
        any_succeeded = False

        for sub_question in sub_questions:
            parsed = None
            last_error = None
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
                for item in parsed.findings:
                    source_id = repository.get_or_create_source(
                        question_id, item.source_url, item.source_title
                    )
                    repository.add_finding(
                        question_id, source_id, item.finding_text, sub_question
                    )
            else:
                logger.warning(
                    "Giving up on sub-question %r after retries, recording a "
                    "placeholder finding: %s",
                    sub_question,
                    last_error,
                )
                source_id = repository.get_or_create_source(
                    question_id, "none", "Extraction failed"
                )
                repository.add_finding(
                    question_id,
                    source_id,
                    "The Researcher agent's response for this sub-question could not "
                    "be parsed after retrying.",
                    sub_question,
                )

        if not any_succeeded:
            raise PipelineStageError(
                "researcher", "no sub-question produced a parseable finding",
                ErrorCode.SCHEMA_VALIDATION_FAILED,
            )

        findings = repository.get_findings_list(question_id)
        degraded = any(f["source_url"] == "none" for f in findings)
        return findings, degraded

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

    def _run_writing_stage(self, topic: str, findings: List[dict], contradictions: List[dict]) -> str:
        writer = get_writer()
        write_task = get_write_task(writer, topic, findings, contradictions)
        try:
            result = _run_single_task_crew(writer, write_task)
        except Exception as e:
            logger.error("Writer stage failed: %s", e)
            raise PipelineStageError("writer", str(e), classify_provider_exception(e)) from e
        return str(result)

    def _index_for_reuse(self, question_id, topic, conclusion_text, findings, conclusion_id):
        try:
            self._reuse_checker.index_question(question_id, topic, conclusion_text)
            self._reuse_checker.index_conclusion(conclusion_id, question_id, conclusion_text)
            for f in findings:
                self._reuse_checker.index_finding(f["id"], question_id, f["text"])
        except Exception as e:
            logger.warning("Failed to index question for future knowledge reuse: %s", e)
