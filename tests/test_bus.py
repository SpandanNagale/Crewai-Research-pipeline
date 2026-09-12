from datetime import datetime

from observability.budget import BudgetSink
from observability.bus import EventBus, NoOpTraceSink, QueueSink, StdoutSink
from observability.events import RunEvent


def _event(run_id="r1", seq=1, event_type="stage_started") -> RunEvent:
    return RunEvent(run_id=run_id, seq=seq, timestamp=datetime.utcnow(),
                     stage="planner", event_type=event_type)


def test_instance_returns_singleton_with_default_sinks():
    EventBus.reset_for_tests()
    bus = EventBus.instance()
    assert bus is EventBus.instance()
    assert any(isinstance(s, StdoutSink) for s in bus.sinks)
    assert any(isinstance(s, QueueSink) for s in bus.sinks)
    assert any(isinstance(s, NoOpTraceSink) for s in bus.sinks)
    assert any(isinstance(s, BudgetSink) for s in bus.sinks)


def test_emit_delivers_to_queue_sink_isolated_by_run_id():
    EventBus.reset_for_tests()
    bus = EventBus.instance()
    bus.emit(_event(run_id="run-a", seq=1))
    bus.emit(_event(run_id="run-b", seq=1))

    got_a = bus.queue_sink.get_queue("run-a").get_nowait()
    assert got_a.run_id == "run-a"
    assert bus.queue_sink.get_queue("run-a").empty()
    assert bus.queue_sink.get_queue("run-b").qsize() == 1


def test_emit_survives_a_raising_sink_and_still_reaches_the_others():
    EventBus.reset_for_tests()
    bus = EventBus.instance()

    class BrokenSink:
        def emit(self, event):
            raise RuntimeError("boom")

    bus.sinks.insert(0, BrokenSink())
    bus.emit(_event(run_id="run-c"))

    assert bus.queue_sink.get_queue("run-c").get_nowait().run_id == "run-c"
