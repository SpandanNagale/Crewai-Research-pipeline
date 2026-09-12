# Research Pipeline — frontend

Next.js 16 (App Router, TypeScript, Tailwind v4) dashboard for the research
pipeline's FastAPI backend (`..`). Three panes: run configuration (left),
the live six-stage pipeline driven by `POST /research/stream`'s SSE events
(center), and the finished report with claims/sources/contradictions/
groundedness (right).

## Run it

```bash
npm install
cp .env.local.example .env.local   # point NEXT_PUBLIC_API_BASE_URL at the backend
npm run dev
```

The backend must be running (`uvicorn app:app --reload` from the repo root)
with this app's origin listed in its `CORS_ALLOWED_ORIGINS` (see
`../.env.example`; defaults to `http://localhost:3000`).

## Structure

- `lib/sse.ts` — parses `POST /research/stream`'s raw SSE frames (the browser's
  built-in `EventSource` can't POST a body, so this reads `response.body`
  directly; mirrors `../eval/harness.py`'s `_parse_sse_stream`).
- `lib/pipeline.ts` — a reducer that folds the RunEvent stream into per-stage
  state (node state, duration, attempt history, cache hits, guardrail fires)
  plus running cost/token totals.
- `lib/useResearchRun.ts` — drives a run: starts the stream, dispatches events
  into the reducer, exposes `start`/`cancel`.
- `components/` — `RunConfigPanel` (left), `PipelinePane` + `StageNode`
  (center), `ReportPane` (right).

## Failure demo

The left pane's "Watch the guardrail fire" button submits the same topic as
`../eval/golden/golden_set.json`'s `adversarial-injection` entry, so the input
guardrail visibly blocks it live — no LLM calls, free to click.
