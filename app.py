import asyncio
import json
import logging
import queue
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from crew.research_crew import PipelineStageError, ResearchCrew
from db import repository
from db.session import init_db
from eval.scorer import score_report
from llm.router import check_provider_health
from memory.reuse import KnowledgeReuseChecker
from observability.bus import EventBus
from observability.errors import ErrorCode
from observability import langfuse_sink
from worker import job_manager

_SSE_HEARTBEAT_SECONDS = 15.0


def _sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Enterprise Research Pipeline API",
    description="Multi-agent research pipeline powered by Gemma 4 + CrewAI, with a "
    "structured traceability DB, contradiction detection, and knowledge reuse.",
    version="2.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


class ResearchRequest(BaseModel):
    topic: str
    evaluate: bool = False
    force_rerun: bool = False


class ResearchResponse(BaseModel):
    topic: str
    report: str
    saved_to: str | None = None
    question_id: int | None = None
    run_id: str | None = None
    reused: bool = False
    contradictions_count: int = 0
    findings_count: int = 0
    groundedness_score: float | None = None
    sub_questions_count: int = 0
    unresolved_sub_questions_count: int = 0
    trace_url: str | None = None


def _pipeline_status_code(error_code: ErrorCode) -> int:
    return 400 if error_code == ErrorCode.GUARDRAIL_BLOCKED else 502


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/providers")
def health_providers():
    """Preflight status of every configured LLM deployment, for the UI's status dots."""
    return {"providers": check_provider_health()}


def _run_pipeline(topic: str, force_rerun: bool, run_id: str | None = None) -> dict:
    crew = ResearchCrew()
    return crew.run_with_meta(topic, force_rerun=force_rerun, run_id=run_id)


@app.post("/research", response_model=ResearchResponse)
async def run_research(req: ResearchRequest):
    """
    Run the full multi-agent research pipeline for a given topic, persisting
    every finding/source/contradiction/conclusion to the traceability DB.
    Runs the (blocking, CPU/IO-heavy) crew execution in a worker thread so it
    doesn't block the event loop for other requests.
    """
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")

    run_id = str(uuid.uuid4())
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _run_pipeline, req.topic, req.force_rerun, run_id)
    except PipelineStageError as e:
        raise HTTPException(
            status_code=_pipeline_status_code(e.error_code),
            detail=f"Pipeline stage '{e.stage}' failed: {e.detail}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    return ResearchResponse(
        topic=req.topic,
        report=result["report"],
        saved_to=result.get("saved_to"),
        question_id=result.get("question_id"),
        run_id=result.get("run_id", run_id),
        reused=result.get("reused", False),
        contradictions_count=result.get("contradictions_count", 0),
        findings_count=result.get("findings_count", 0),
        groundedness_score=result.get("groundedness_score"),
        sub_questions_count=result.get("sub_questions_count", 0),
        unresolved_sub_questions_count=result.get("unresolved_sub_questions_count", 0),
        trace_url=langfuse_sink.get_trace_url(result.get("run_id", run_id)),
    )


@app.post("/research/async")
async def run_research_async(req: ResearchRequest):
    """
    Fire-and-forget variant of /research: returns a job_id immediately and
    runs the pipeline in a background thread. Poll GET /jobs/{job_id} for
    status/result. This is the pattern that would map onto a Celery+Redis
    queue at higher volume (see README "Scaling" section) — swap
    worker.job_manager.submit() for a Celery task.delay() call.
    """
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")

    run_id = str(uuid.uuid4())
    job_manager.submit(_run_pipeline, req.topic, req.force_rerun, run_id, job_id=run_id)
    return {"job_id": run_id, "status": "pending"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id.")
    return job


@app.get("/research/{run_id}")
def get_run(run_id: str):
    """Node-state view of one run, for polling without SSE."""
    run = repository.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Unknown run_id.")
    return {**run, "stages": repository.get_stage_states(run_id)}


@app.post("/research/{run_id}/resume", response_model=ResearchResponse)
async def resume_research(run_id: str):
    """Re-enters the pipeline at the first non-succeeded stage of a prior run."""
    run = repository.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Unknown run_id.")
    if run["status"] == "succeeded":
        raise HTTPException(status_code=409, detail="Run already succeeded; nothing to resume.")

    crew = ResearchCrew()
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None, crew.run_with_meta, run["topic"], False, run_id, run["question_id"]
        )
    except PipelineStageError as e:
        raise HTTPException(
            status_code=_pipeline_status_code(e.error_code),
            detail=f"Pipeline stage '{e.stage}' failed: {e.detail}",
        )

    return ResearchResponse(
        topic=run["topic"],
        report=result["report"],
        saved_to=result.get("saved_to"),
        question_id=result.get("question_id"),
        run_id=result.get("run_id", run_id),
        reused=result.get("reused", False),
        contradictions_count=result.get("contradictions_count", 0),
        findings_count=result.get("findings_count", 0),
        groundedness_score=result.get("groundedness_score"),
        sub_questions_count=result.get("sub_questions_count", 0),
        unresolved_sub_questions_count=result.get("unresolved_sub_questions_count", 0),
        trace_url=langfuse_sink.get_trace_url(result.get("run_id", run_id)),
    )


@app.post("/research/stream")
async def stream_research(req: ResearchRequest, request: Request):
    """
    Streams RunEvents live as the pipeline executes, via the same EventBus
    the rest of the pipeline emits to (see observability/bus.py). Named SSE
    event types: stage_update (every RunEvent), error, done. No sentinel
    strings in the data payload -- each is a JSON object.
    """
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")

    run_id = str(uuid.uuid4())

    async def event_generator():
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(None, _run_pipeline, req.topic, req.force_rerun, run_id)
        event_queue = EventBus.instance().queue_sink.get_queue(run_id)

        yield _sse("stage_update", {"event_type": "run_started", "run_id": run_id, "stage": ""})
        last_activity = time.monotonic()

        try:
            while not future.done():
                if await request.is_disconnected():
                    # Best-effort only: a plain ThreadPoolExecutor thread can't be
                    # force-killed once running. This unblocks the client and
                    # records the outcome; the background run finishes on its own.
                    future.cancel()
                    repository.set_run_status(run_id, "aborted")
                    return

                try:
                    event = event_queue.get_nowait()
                except queue.Empty:
                    if time.monotonic() - last_activity > _SSE_HEARTBEAT_SECONDS:
                        yield ": heartbeat\n\n"
                        last_activity = time.monotonic()
                    await asyncio.sleep(0.1)
                    continue

                yield _sse("stage_update", event.model_dump(mode="json", exclude_none=True))
                last_activity = time.monotonic()

            while True:
                try:
                    event = event_queue.get_nowait()
                except queue.Empty:
                    break
                yield _sse("stage_update", event.model_dump(mode="json", exclude_none=True))

            try:
                result = future.result()
            except PipelineStageError as e:
                yield _sse("error", {"stage": e.stage, "detail": e.detail, "error_code": e.error_code.value})
                return
            except Exception as e:
                yield _sse("error", {"detail": str(e)})
                return

            done_payload = {
                "run_id": result.get("run_id", run_id),
                "question_id": result.get("question_id"),
                "report": result["report"],
                "saved_to": result.get("saved_to"),
                "reused": result.get("reused", False),
                "contradictions_count": result.get("contradictions_count", 0),
                "findings_count": result.get("findings_count", 0),
                "groundedness_score": result.get("groundedness_score"),
                "sub_questions_count": result.get("sub_questions_count", 0),
                "unresolved_sub_questions_count": result.get("unresolved_sub_questions_count", 0),
                "trace_url": langfuse_sink.get_trace_url(result.get("run_id", run_id)),
            }
            if req.evaluate:
                eval_result = await loop.run_in_executor(None, score_report, result["report"], req.topic)
                done_payload["eval"] = {
                    "citation_score": eval_result.citation_score,
                    "coverage_score": eval_result.coverage_score,
                    "clarity_score": eval_result.clarity_score,
                    "total": eval_result.total,
                    "feedback": eval_result.feedback,
                    "parse_failed": eval_result.parse_failed,
                }
            yield _sse("done", done_payload)
        finally:
            EventBus.instance().queue_sink.drop(run_id)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


class EvalRequest(BaseModel):
    topic: str
    report: str


@app.post("/eval")
def evaluate_report(req: EvalRequest):
    """Score an existing report without re-running the pipeline."""
    result = score_report(req.report, req.topic)
    return {
        "citation_score": result.citation_score,
        "coverage_score": result.coverage_score,
        "clarity_score": result.clarity_score,
        "total": result.total,
        "feedback": result.feedback,
        "parse_failed": result.parse_failed,
    }


# ── Traceability / knowledge endpoints ──────────────────────────────────


@app.get("/questions/{question_id}/trace")
def get_trace(question_id: int):
    """Conclusion(s) for a question, plus the finding chain and sources behind each."""
    trace = repository.get_trace(question_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Unknown question_id.")
    return trace


@app.get("/questions/{question_id}/contradictions")
def get_contradictions(question_id: int):
    """List of flagged contradictions between findings for a question."""
    question = repository.get_question(question_id)
    if question is None:
        raise HTTPException(status_code=404, detail="Unknown question_id.")
    return {"question_id": question_id, "contradictions": repository.get_contradictions(question_id)}


@app.get("/questions/{question_id}/groundedness")
def get_groundedness(question_id: int):
    """Per-claim groundedness result from the Critic stage's last check for this question."""
    question = repository.get_question(question_id)
    if question is None:
        raise HTTPException(status_code=404, detail="Unknown question_id.")
    return {"question_id": question_id, "claims": repository.get_claim_groundings(question_id)}


@app.get("/knowledge/search")
def search_knowledge(q: str, limit: int = 8):
    """Semantic search over past findings and conclusions across all questions."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query 'q' cannot be empty.")
    checker = KnowledgeReuseChecker()
    return {"query": q, "results": checker.search_knowledge(q, n_results=limit)}
