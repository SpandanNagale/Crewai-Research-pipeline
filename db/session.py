from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config.settings import settings
from db.models import Base

_connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(settings.DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    """
    Local-dev-only convenience: create all tables if they don't exist yet.

    A no-op when settings.VECTOR_BACKEND == "pgvector" (the deployed
    configuration) -- that path's schema is owned by Alembic
    (`alembic upgrade head`, run as a start step; see Dockerfile), because
    create_all() has no concept of migrations and would corrupt a real
    database the moment its schema needs to change. Only safe to leave live
    for the in-memory-vector-backend / SQLite local-dev path, which has no
    persistent data to protect.
    """
    if settings.VECTOR_BACKEND == "pgvector":
        return
    Base.metadata.create_all(engine)


@contextmanager
def get_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
