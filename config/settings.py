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
    KNOWLEDGE_REUSE_THRESHOLD: float = float(os.getenv("KNOWLEDGE_REUSE_THRESHOLD", "0.87"))

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


settings = Settings()
