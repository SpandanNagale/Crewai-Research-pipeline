"""
Minimal in-process async job runner.

This is intentionally NOT Celery: it's a ThreadPoolExecutor plus an
in-memory status dict, just enough so that POST /research/async can return
immediately with a job_id instead of blocking a request thread for the
duration of a full research pipeline run.

Scaling this to a real queue (see README "Scaling" section) means swapping
`submit()`'s body for `task.delay(*args, **kwargs)` on a Celery task backed
by Redis/RabbitMQ, and swapping the in-memory `_jobs` dict for a `jobs` table
(or Redis hash) so status survives a process restart and is visible across
multiple worker processes.
"""
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="research-job")
_jobs: dict = {}
_lock = threading.Lock()


def submit(fn: Callable[..., Any], *args, **kwargs) -> str:
    """Submit fn(*args, **kwargs) to run in the background. Returns a job_id."""
    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {"status": "pending", "result": None, "error": None}

    def _run():
        with _lock:
            _jobs[job_id]["status"] = "running"
        try:
            result = fn(*args, **kwargs)
            with _lock:
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["result"] = result
        except Exception as e:
            logger.error("Background job %s failed: %s", job_id, e)
            with _lock:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["error"] = str(e)

    _executor.submit(_run)
    return job_id


def get_job(job_id: str) -> Optional[dict]:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job is not None else None
