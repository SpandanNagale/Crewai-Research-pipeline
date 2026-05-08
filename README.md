# Multi-Agent Research Pipeline

A fully local multi-agent research pipeline using **Gemma 4** (via Ollama) and **CrewAI**. Four specialized agents — Planner, Researcher, Writer, and Critic — collaborate to produce a cited, structured research report on any topic.

## Architecture

```
User query → Ollama/Gemma 4 → CrewAI Orchestrator
                                    ↓
              Planner → Researcher → Writer → Critic
                                    ↓
                            Final markdown report
```

## Setup

### 1. Start Ollama with Gemma 4
```bash
ollama pull gemma3:12b
ollama serve
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
# Add your SERPER_API_KEY (free at serper.dev)
```

## Usage

### CLI
```bash
# Basic run
python main.py "The impact of AI on drug discovery"

# With evaluation scoring
python main.py "Quantum computing breakthroughs in 2024" --eval
```

### REST API
```bash
uvicorn app:app --reload --port 8000

# POST /research
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "AI in healthcare 2024", "evaluate": true}'

# Streaming
curl -N http://localhost:8000/research/stream \
  -X POST -H "Content-Type: application/json" \
  -d '{"topic": "Fusion energy progress"}'
```

## Project Structure

```
research-pipeline/
├── main.py                  # CLI entry point
├── app.py                   # FastAPI REST + streaming API
├── config/
│   ├── settings.py          # typed config from .env
│   └── agents_config.yaml   # all agent prompts (edit without touching code)
├── agents/                  # one file per agent
├── tasks/                   # one file per task
├── tools/                   # search, wiki, file save
├── crew/
│   └── research_crew.py     # assembles and runs the pipeline
├── memory/
│   └── vector_store.py      # ChromaDB persistent memory
├── eval/
│   └── scorer.py            # LLM-as-judge scoring (citation, coverage, clarity)
└── outputs/                 # generated reports land here
```

## Evaluation

The eval module uses Gemma 4 itself as a judge, scoring reports on:
- **Citation score** (40% weight) — are claims backed by sources?
- **Coverage score** (35% weight) — is the topic fully addressed?
- **Clarity score** (25% weight) — is it readable and well-structured?

## Switching models

Change `OLLAMA_MODEL` in `.env` to any Ollama model (e.g. `llama3.1:8b`, `mistral:7b`).
