import threading
import time

from observability.bus import EventBus
from observability.context import clear_run, current_run, emit_event, set_stage, start_run


def test_emit_event_is_a_noop_with_no_active_run():
    clear_run()
    emit_event("stage_started")  # must not raise
    assert current_run() is None


def test_start_run_sets_context_and_emit_event_stamps_it():
    EventBus.reset_for_tests()
    start_run("run-1")
    set_stage("planner")
    emit_event("stage_started", message="go")

    ctx = current_run()
    assert ctx.run_id == "run-1"
    assert len(ctx.stage_events) == 1
    assert ctx.stage_events[0].event_type == "stage_started"
    assert ctx.stage_events[0].seq == 1
    clear_run()


def test_set_stage_resets_stage_events_for_the_new_stage():
    start_run("run-2")
    set_stage("planner")
    emit_event("stage_started")
    set_stage("researcher")
    assert current_run().stage_events == []
    clear_run()


def test_seq_increments_across_multiple_emits_within_a_run():
    start_run("run-3")
    set_stage("planner")
    emit_event("a")
    emit_event("b")
    assert [e.seq for e in current_run().stage_events] == [1, 2]
    clear_run()


def test_run_context_is_isolated_per_thread():
    results = {}

    def worker(run_id):
        start_run(run_id)
        set_stage("planner")
        emit_event("x")
        time.sleep(0.05)
        results[run_id] = current_run().run_id
        clear_run()

    t1 = threading.Thread(target=worker, args=("thread-a",))
    t2 = threading.Thread(target=worker, args=("thread-b",))
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert results == {"thread-a": "thread-a", "thread-b": "thread-b"}
