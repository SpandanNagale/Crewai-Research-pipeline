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

    # ── LLM provider selection ──────────────────────────────────────
    # ollama_local (default) | ollama_cloud | groq. Per-agent overrides live
    # in config/agents_config.yaml (llm_provider / llm_model / llm_temperature).
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama_local")
    LLM_MODEL: str = os.getenv("LLM_MODEL", os.getenv("OLLAMA_MODEL", "gemma3:12b"))

    # ── Search provider ──────────────────────────────────────────────
    # auto (serper if SERPER_API_KEY is set, else duckduckgo) | serper | duckduckgo
    SEARCH_PROVIDER: str = os.getenv("SEARCH_PROVIDER", "auto")
    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")

    # ── Persistence ──────────────────────────────────────────────────
    _default_db_path = os.path.join(os.path.dirname(__file__), "..", "research.db")
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{_default_db_path}")
    KNOWLEDGE_REUSE_THRESHOLD: float = float(os.getenv("KNOWLEDGE_REUSE_THRESHOLD", "0.87"))

    OUTPUTS_DIR: str = os.path.join(os.path.dirname(__file__), "..", "outputs")


settings = Settings()
