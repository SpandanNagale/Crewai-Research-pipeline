from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from crew.research_crew import ResearchCrew
from eval.scorer import score_report
import asyncio

app = FastAPI(
    title="Research Pipeline API",
    description="Multi-agent research pipeline powered by Gemma 4 + CrewAI",
    version="1.0.0",
)


class ResearchRequest(BaseModel):
    topic: str
    evaluate: bool = False


class ResearchResponse(BaseModel):
    topic: str
    report: str
    saved_to: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
def run_research(req: ResearchRequest):
    """
    Run the full multi-agent research pipeline for a given topic.
    Returns the final report as a string.
    """
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")

    crew = ResearchCrew()
    try:
        report = crew.run(req.topic)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    return ResearchResponse(topic=req.topic, report=report)


@app.post("/research/stream")
async def stream_research(req: ResearchRequest):
    """
    Stream the research report token-by-token as it is generated.
    Uses Server-Sent Events (SSE) format.
    """
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")

    async def event_generator():
        yield f"data: Starting research pipeline for: {req.topic}\n\n"
        await asyncio.sleep(0.1)

        loop = asyncio.get_event_loop()
        crew = ResearchCrew()

        try:
            report = await loop.run_in_executor(None, crew.run, req.topic)
            for chunk in report.split("\n"):
                yield f"data: {chunk}\n\n"
                await asyncio.sleep(0.01)

            if req.evaluate:
                yield "data: \n\ndata: --- EVALUATING REPORT ---\n\n"
                result = await loop.run_in_executor(None, score_report, report, req.topic)
                yield f"data: {str(result)}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/eval")
def evaluate_report(topic: str, report: str):
    """Score an existing report without re-running the pipeline."""
    result = score_report(report, topic)
    return {
        "citation_score": result.citation_score,
        "coverage_score": result.coverage_score,
        "clarity_score": result.clarity_score,
        "total": result.total,
        "feedback": result.feedback,
    }
