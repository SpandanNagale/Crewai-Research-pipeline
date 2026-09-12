# FastAPI backend only -- the frontend/ Next.js app has its own Dockerfile
# and is normally deployed separately (Vercel), see README's Deployment
# section. This image is for the API + CLI.
FROM python:3.12-slim

WORKDIR /app

# System deps for building any C-extension wheels chromadb/tokenizers pull in.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# research.db (path from config/settings.py::DATABASE_URL, overridable) and
# chroma_db/ (hardcoded in memory/vector_store.py -- not currently
# configurable) are created here, at the repo root, on first use -- both
# ephemeral unless DATABASE_URL is redirected into a mounted volume and
# chroma_db/ is mounted directly, as docker-compose.yml does. See README's
# Deployment section for why that's the explicit choice here rather than
# managed Postgres + a hosted vector store.
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
