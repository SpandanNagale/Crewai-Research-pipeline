from dotenv import load_dotenv
import os

load_dotenv()


class Settings:
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma3:12b")
    OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))
    OLLAMA_NUM_CTX: int = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")
    OUTPUTS_DIR: str = os.path.join(os.path.dirname(__file__), "..", "outputs")


settings = Settings()
