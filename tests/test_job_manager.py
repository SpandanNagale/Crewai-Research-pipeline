import time

from worker import job_manager


def test_submit_uses_given_job_id_when_provided():
    job_id = job_manager.submit(lambda: "result", job_id="custom-id")
    assert job_id == "custom-id"
    for _ in range(50):
        job = job_manager.get_job(job_id)
        if job["status"] == "done":
            break
        time.sleep(0.02)
    assert job["result"] == "result"


def test_submit_generates_a_job_id_when_not_provided():
    job_id = job_manager.submit(lambda: "result")
    assert job_id
    assert job_manager.get_job(job_id) is not None
