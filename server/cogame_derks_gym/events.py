"""The replay's event list: a closed seven-value vocabulary.

Extracted server-side while the episode runs (the engine reads
``agent_stat`` for all ten heroes each tick) and stored in the replay
header, so the viewer's feed and scrubber beats need no re-derivation and
no second source of truth.

| kind         | when                                | payload            |
|--------------|-------------------------------------|--------------------|
| draft        | always, at tick 0                   | pids               |
| first_blood  | the first hero death of the episode | pid, victim_pid    |
| kill         | any later hero kill                 | pid, victim_pid    |
| tower        | towers_killed[pid] increased        | pid, team          |
| level_spike  | level[pid] increased                | pid, level         |
| ancient      | an Ancient fell                     | team               |
| end          | always, last                        | reason             |

``victim_pid`` is the pid whose ``deaths`` counter increased on the same
tick; if several did, the lowest pid (deterministic).

Cap: 400 events. On overflow the oldest ``level_spike`` is dropped first,
then the oldest ``kill``; ``draft``, ``first_blood``, ``tower``,
``ancient`` and ``end`` are never dropped.
"""

from __future__ import annotations

from . import defaults

KINDS = ("draft", "first_blood", "kill", "tower", "level_spike",
         "ancient", "end")
MAX_EVENTS = 400
# Drop order on overflow (never draft/first_blood/tower/ancient/end).
_DROPPABLE = ("level_spike", "kill")


class EventLog:
    """Append-only, capped event list in tick order."""

    def __init__(self, max_events: int = MAX_EVENTS):
        self._events: list[dict] = []
        self._max = max_events
        self._had_death = False

    def __len__(self) -> int:
        return len(self._events)

    def events(self) -> list[dict]:
        return list(self._events)

    def add(self, tick: int, kind: str, **payload) -> None:
        assert kind in KINDS, kind
        self._events.append({"tick": int(tick), "kind": kind, **payload})
        self._trim()

    def _trim(self) -> None:
        while len(self._events) > self._max:
            for kind in _DROPPABLE:
                index = next((i for i, e in enumerate(self._events)
                              if e["kind"] == kind), None)
                if index is not None:
                    del self._events[index]
                    break
            else:
                # Nothing droppable left. draft/first_blood/tower/ancient/
                # end are NEVER dropped, so the cap yields instead: the
                # old `del self._events[self._max:]` here deleted the tail,
                # which on an add_end() past the cap would have deleted the
                # `end` record that had just been appended. Unreachable in
                # practice (at most ~29 undroppable events exist: 1 draft,
                # 1 first_blood, <=24 tower, <=2 ancient, 1 end), but the
                # invariant is the one the replay format promises.
                return

    def add_draft(self) -> None:
        self.add(0, "draft", pids=list(defaults.SEAT_HERO_PIDS))

    def add_end(self, tick: int, reason: str) -> None:
        self.add(tick, "end", reason=reason)

    def observe_tick(self, tick: int, stats: list[dict],
                     previous: list[dict]) -> None:
        """Emit the events implied by one tick's stat increases.

        ``stats`` / ``previous`` are ten dicts with the level / kills /
        deaths / towers_killed values read this tick and last tick.
        """
        victims = [pid for pid in range(defaults.NUM_HEROES)
                   if stats[pid]["deaths"] > previous[pid]["deaths"]]
        killers = [pid for pid in range(defaults.NUM_HEROES)
                   if stats[pid]["kills"] > previous[pid]["kills"]]
        victim_pid = victims[0] if victims else None

        if victims and not self._had_death:
            # The episode's first hero death, whoever (or whatever — a
            # tower, a creep) landed it.
            self._had_death = True
            self.add(tick, "first_blood",
                     pid=killers[0] if killers else victim_pid,
                     victim_pid=victim_pid)
            killers = killers[1:] if killers else []
        for pid in killers:
            payload = {"pid": pid}
            if victim_pid is not None:
                payload["victim_pid"] = victim_pid
            self.add(tick, "kill", **payload)

        for pid in range(defaults.NUM_HEROES):
            if stats[pid]["towers_killed"] > previous[pid]["towers_killed"]:
                self.add(tick, "tower", pid=pid,
                         team=defaults.team_for_pid(pid))
            if stats[pid]["level"] > previous[pid]["level"]:
                self.add(tick, "level_spike", pid=pid,
                         level=int(stats[pid]["level"]))
