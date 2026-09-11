import asyncio
import logging
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
from worker import job_manager

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
    reused: bool = False
    contradictions_count: int = 0
    findings_count: int = 0


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/providers")
def health_providers():
    """Preflight status of every configured LLM deployment, for the UI's status dots."""
    return {"providers": check_provider_health()}


def _run_pipeline(topic: str, force_rerun: bool) -> dict:
    crew = ResearchCrew()
    return crew.run_with_meta(topic, force_rerun=force_rerun)


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

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _run_pipeline, req.topic, req.force_rerun)
    except PipelineStageError as e:
        raise HTTPException(status_code=502, detail=f"Pipeline stage '{e.stage}' failed: {e.detail}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    return ResearchResponse(
        topic=req.topic,
        report=result["report"],
        saved_to=result.get("saved_to"),
        question_id=result.get("question_id"),
        reused=result.get("reused", False),
        contradictions_count=result.get("contradictions_count", 0),
        findings_count=result.get("findings_count", 0),
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

    job_id = job_manager.submit(_run_pipeline, req.topic, req.force_rerun)
    return {"job_id": job_id, "status": "pending"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id.")
    return job


@app.post("/research/stream")
async def stream_research(req: ResearchRequest):
    """
    Stream the research report token-by-token as it is generated.
    Uses Server-Sent Events (SSE) format. Writes to the traceability DB
    during execution, same as /research.
    """
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")

    async def event_generator():
        yield f"data: --- STARTING RESEARCH PIPELINE FOR: {req.topic} ---\n\n"
        await asyncio.sleep(0.1)

        loop = asyncio.get_event_loop()

        try:
            result = await loop.run_in_executor(None, _run_pipeline, req.topic, req.force_rerun)
            report = result["report"]
            yield "data: \n\ndata: --- FINAL REPORT ---\n\n"
            for chunk in report.split("\n"):
                yield f"data: {chunk}\n\n"
                await asyncio.sleep(0.01)

            if req.evaluate:
                yield "data: \n\ndata: --- EVALUATING REPORT ---\n\n"
                eval_result = await loop.run_in_executor(None, score_report, report, req.topic)
                yield f"data: {str(eval_result)}\n\n"

            yield "data: [DONE]\n\n"

        except PipelineStageError as e:
            yield f"data: ERROR: pipeline stage '{e.stage}' failed: {e.detail}\n\n"
        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"

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


@app.get("/knowledge/search")
def search_knowledge(q: str, limit: int = 8):
    """Semantic search over past findings and conclusions across all questions."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query 'q' cannot be empty.")
    checker = KnowledgeReuseChecker()
    return {"query": q, "results": checker.search_knowledge(q, n_results=limit)}
