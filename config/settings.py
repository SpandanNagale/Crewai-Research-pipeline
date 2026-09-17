from dotenv import load_dotenv
import os

load_dotenv()


class Settings:
    # ── Local Ollama (default, zero-cost, no API key required) ──────────
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma3:12b")
    OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))
    OLLAMA_NUM_CTX: int = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

    # ── Ollama Cloud (opt-in) ─────────────────────────────────────────
    OLLAMA_CLOUD_BASE_URL: str = os.getenv("OLLAMA_CLOUD_BASE_URL", "https://ollama.com")
    OLLAMA_API_KEY: str = os.getenv("OLLAMA_API_KEY", "")

    # ── Groq (opt-in) ────────────────────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

    # ── OpenRouter (opt-in) ──────────────────────────────────────────
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    # ── Gemini (opt-in) ──────────────────────────────────────────────
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # ── Search provider ──────────────────────────────────────────────
    # auto (serper if SERPER_API_KEY is set, else duckduckgo) | serper | duckduckgo
    SEARCH_PROVIDER: str = os.getenv("SEARCH_PROVIDER", "auto")
    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")

    # ── Persistence ──────────────────────────────────────────────────
    _default_db_path = os.path.join(os.path.dirname(__file__), "..", "research.db")
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{_default_db_path}")
    # pgvector requires settings.DATABASE_URL to point at Postgres; memory is
    # an in-process NumPy backend for local dev / tests with no DB dependency.
    VECTOR_BACKEND: str = os.getenv("VECTOR_BACKEND", "memory")
    # Calibrated empirically 2026-09-17 against gemini-embedding-001 @ 768
    # dims via `python -m eval.calibrate_reuse`: true-match (paraphrase)
    # pairs scored 0.8405-0.9416 cosine similarity, near-miss (same-domain,
    # different-question) pairs scored 0.4836-0.6634 -- a clean gap, so
    # 0.75 (the midpoint) is used rather than hugging either edge. This
    # REPLACES the 0.87 value tuned for Chroma's all-MiniLM-L6-v2 default
    # embedding function, which is NOT comparable: different embedding
    # models produce different cosine similarity distributions entirely
    # (notably, Gemini's near-miss scores here sit well below 0.87, not
    # above it -- carrying the old value forward would have made reuse
    # nearly impossible, not falsely trigger it). Re-run the calibration
    # script and update this value if EMBEDDING_MODEL or
    # EMBEDDING_DIMENSIONS ever changes.
    KNOWLEDGE_REUSE_THRESHOLD: float = float(os.getenv("KNOWLEDGE_REUSE_THRESHOLD", "0.75"))

    # ── Embeddings (Gemini, via google-genai) ─────────────────────────
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    # 768, not the model's native 3072: pgvector cannot build an HNSW or
    # IVFFlat index on vectors above 2000 dimensions, which would force a
    # sequential scan on every reuse/cache lookup.
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

    # ── Caching (Phase 5) ──────────────────────────────────────────────
    # Search results go stale; TTL-bound the cache rather than caching forever.
    SEARCH_CACHE_TTL_SECONDS: int = int(os.getenv("SEARCH_CACHE_TTL_SECONDS", str(6 * 3600)))

    OUTPUTS_DIR: str = os.path.join(os.path.dirname(__file__), "..", "outputs")

    # ── Frontend (Phase 8) ───────────────────────────────────────────────
    # FastAPI is API-only now (the Next.js app in frontend/ is the UI) --
    # comma-separated list of origins allowed to call this API cross-origin.
    CORS_ALLOWED_ORIGINS: list = [
        o.strip() for o in os.getenv(
            "CORS_ALLOWED_ORIGINS", "http://localhost:3000"
        ).split(",") if o.strip()
    ]

    # ── Langfuse tracing (Phase 7, opt-in) ──────────────────────────────
    # Unset by default: EventBus falls back to the no-op trace sink, no
    # network calls attempted, nothing in the UI/API breaks.
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    # ── Deployment hardening (Phase 9) ───────────────────────────────────
    # Hard kill-switch on real LLM spend across the whole deployment (not
    # per-user) -- 0 disables it. See observability/budget.py.
    DAILY_BUDGET_USD: float = float(os.getenv("DAILY_BUDGET_USD", "0"))
    # Requests per minute per client IP on the expensive /research* endpoints
    # -- 0 disables it. In-memory, per-process: fine for the single-instance
    # deployment this project targets; a multi-instance deployment needs a
    # shared store (Redis) instead, see app.py::RateLimiter's docstring.
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "0"))


settings = Settings()
