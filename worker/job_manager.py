"""
Minimal in-process async job runner.

This is intentionally NOT Celery: it's a ThreadPoolExecutor, just enough so
that POST /research/async can return immediately with a job_id instead of
blocking a request thread for the duration of a full research pipeline run.

Status is persisted to the `jobs` table (db/repository.py) rather than an
in-memory dict (fixed defect #14: a process restart used to silently lose
every in-flight/completed job's status -- GET /jobs/{job_id} would 404 for a
job that had, in reality, completed just fine). The execution itself is
still in-process and does not survive a restart -- scaling that out to a
real queue (see README "Scaling" section) means swapping `submit()`'s body
for `task.delay(*args, **kwargs)` on a Celery task backed by Redis/RabbitMQ;
the `jobs` table already gives that a durable status store to write to.
"""
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from db import repository

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="research-job")


def submit(fn: Callable[..., Any], *args, job_id: Optional[str] = None, **kwargs) -> str:
    """
    Submit fn(*args, **kwargs) to run in the background. Returns a job_id --
    the caller's own job_id when given (e.g. the pipeline's run_id, so
    GET /jobs/{job_id} and GET /research/{run_id} address the same run), or a
    freshly generated one otherwise.
    """
    job_id = job_id or str(uuid.uuid4())
    repository.create_job(job_id)

    def _run():
        repository.set_job_status(job_id, "running")
        try:
            result = fn(*args, **kwargs)
            repository.set_job_status(job_id, "done", result=result)
        except Exception as e:
            logger.error("Background job %s failed: %s", job_id, e)
            repository.set_job_status(job_id, "failed", error=str(e))

    _executor.submit(_run)
    return job_id


def get_job(job_id: str) -> Optional[dict]:
    return repository.get_job(job_id)
