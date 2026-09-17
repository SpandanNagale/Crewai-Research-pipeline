# FastAPI backend only -- the frontend/ Next.js app has its own Dockerfile
# and is deployed separately (Vercel), see README's Deployment section.
# This image is for the API + CLI, targeting Railway.
#
# No build-essential: verified with a real `docker build` (no compiler
# stage) that every dependency installs from prebuilt wheels. Note this
# does NOT mean chromadb/onnxruntime/tokenizers are gone from the image --
# crewai itself unconditionally depends on all three (see its own
# dependency metadata, unrelated to whether this project's code uses
# Chroma). Dropping our own chromadb pin only removes a redundant version
# constraint, not the package.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Storage is managed Postgres + pgvector (Neon), not local disk -- this
# image assumes DATABASE_URL points at Postgres and VECTOR_BACKEND=pgvector
# (see docker-compose.yml for the local equivalent). Schema is owned by
# Alembic, run here as a start step rather than lazily on first request so
# a migration never races two concurrent cold requests.
#
# Railway injects $PORT at runtime; never hardcode 8000 in CMD.
EXPOSE 8000

CMD alembic upgrade head && uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
