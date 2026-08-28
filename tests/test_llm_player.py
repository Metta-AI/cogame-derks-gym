"""The prompt policies: parsing, the one retry, and the fallbacks.

The Anthropic transport is stubbed everywhere here — a test suite must
never make a network call — but the code under test is the real
`PromptDraftPolicy.on_draft`, including its timeout, its single retry at
temperature 0 and its fall-back to the scripted draft rule.
"""

import asyncio
import json
import time
from pathlib import Path

import pytest

from cogame_derks_gym import catalog, defaults, draft
from cogame_derks_gym.config import GameConfig

from players.client import PlayerError
from players.derk_player import (CALL_TIMEOUT_SECONDS,
                                 DEADLINE_SAFETY_SECONDS, DEFAULT_SCRIPTED,
                                 FALLBACK_REASONS, MAX_NOTE_RUNES, PROMPTS,
                                 SCRIPTED_NAMES,
                                 PromptDraftPolicy, ScriptedDraftPolicy,
                                 brawler_picks, call_timeout,
                                 first_json_object, forge_picks, legal_picks,
                                 resolve_mode, strip_one_fence)

REAL_NAMES = [f"champion-{i}" for i in range(defaults.NUM_SEATS)]
REPO_ROOT = Path(__file__).resolve().parents[1]


def observation(seat=2):
    cfg = GameConfig.from_dict({
        "players": [{"name": n} for n in REAL_NAMES],
        "tokens": [f"t{i}" for i in range(defaults.NUM_SEATS)],
        "seed": 1,
    })
    return draft.draft_observation(seat, cfg)


class Transport:
    """Records request bodies and replays a scripted list of outcomes."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.bodies = []

    async def __call__(self, body):
        self.bodies.append(body)
        outcome = self.outcomes.pop(0) if self.outcomes else ""
        if isinstance(outcome, Exception):
            raise outcome
        if outcome == "__hang__":
            await asyncio.sleep(60)
        return outcome


def policy(*outcomes, prompt="derk-drafter-v1"):
    return PromptDraftPolicy(prompt, micro=lambda t, rows: [],
                             api_key="test-key", transport=Transport(*outcomes))


# -- parsing -----------------------------------------------------------------

def test_strip_one_fence():
    assert strip_one_fence('{"a":1}') == '{"a":1}'
    assert strip_one_fence('```json\n{"a":1}\n```') == '{"a":1}'
    assert strip_one_fence('```\n{"a":1}\n```') == '{"a":1}'
    assert strip_one_fence('  {"a":1}  ') == '{"a":1}'


def test_legal_picks_accepts_plain_and_fenced_json():
    obs = observation()
    body = '{"arm":"arm_cleaver","tail":"tail_plate","misc":"misc_regen",' \
           '"note":"tanky"}'
    assert legal_picks(body, obs) == {
        "arm": "arm_cleaver", "tail": "tail_plate", "misc": "misc_regen",
        "note": "tanky"}
    assert legal_picks(f"```json\n{body}\n```", obs)["arm"] == "arm_cleaver"


@pytest.mark.parametrize("reply", [
    "sure! here you go",                       # prose
    "{broken",                                 # not JSON
    '["arm_cleaver"]',                         # not an object
    '{"arm":"arm_nope","tail":"tail_plate","misc":"misc_regen"}',
    '{"arm":"tail_plate","tail":"tail_plate","misc":"misc_regen"}',
    '{"arm":"arm_cleaver","tail":"tail_plate"}',
    '{"arm":1,"tail":"tail_plate","misc":"misc_regen"}',
])
def test_legal_picks_rejects(reply):
    assert legal_picks(reply, observation()) is None


LEGAL_BODY = ('{"arm":"arm_cleaver","tail":"tail_plate","misc":"misc_regen",'
              '"note":"tanky"}')


@pytest.mark.parametrize("reply", [
    f"Here is my draft: {LEGAL_BODY}",                  # prose before
    f"{LEGAL_BODY}\nHope that helps!",                  # prose after
    f"Sure!\n{LEGAL_BODY}\nDone.",                      # prose both sides
    f"Here you go:\n```json\n{LEGAL_BODY}\n```",        # prose + fence
    f"```json\n{LEGAL_BODY}\n```\nlet me know!",        # fence + trailing prose
    f"[{LEGAL_BODY}]",                                  # object inside an array
])
def test_legal_picks_extracts_the_object_from_surrounding_prose(reply):
    """Checklist item 8: parsing accepts surrounding prose and extracts
    the JSON object."""
    assert legal_picks(reply, observation()) == {
        "arm": "arm_cleaver", "tail": "tail_plate", "misc": "misc_regen",
        "note": "tanky"}


def test_two_objects_in_one_reply_the_first_balanced_one_wins():
    """Documented behaviour: a model that offers alternatives gets its
    first answer counted, never a merge of the two."""
    second = '{"arm":"arm_needler","tail":"tail_rotor","misc":"misc_focus"}'
    picks = legal_picks(f"maybe {LEGAL_BODY} or {second}?", observation())
    assert picks["arm"] == "arm_cleaver"
    assert first_json_object(f"maybe {LEGAL_BODY} or {second}?")["arm"] == \
        "arm_cleaver"


def test_extraction_is_not_confused_by_braces_inside_a_string():
    """The JSON parser, not a brace counter, decides where the object
    ends — so braces inside the note cannot truncate it."""
    body = json.dumps({"arm": "arm_cleaver", "tail": "tail_plate",
                       "misc": "misc_regen", "note": "} not the end {"})
    picks = legal_picks(f"draft: {body} — good luck!", observation())
    assert picks == {"arm": "arm_cleaver", "tail": "tail_plate",
                     "misc": "misc_regen", "note": "} not the end {"}


def test_first_json_object_returns_none_when_there_is_no_object():
    assert first_json_object("sure! here you go") is None
    assert first_json_object("{broken") is None
    assert first_json_object('["arm_cleaver"]') is None


def test_a_long_note_is_trimmed_before_the_frame_is_sent():
    """A model that writes an essay must lose its note, never its picks.

    The server drops a >4096-byte frame BEFORE the JSON parse, so an
    unbounded note would cost the seat its whole draft. The player trims
    to the server's own rune cap on the way out; the server's truncation
    on receipt stays authoritative.
    """
    assert MAX_NOTE_RUNES == catalog.MAX_NOTE_RUNES
    rocket = "\U0001F680"
    reply = json.dumps({"arm": "arm_cleaver", "tail": "tail_plate",
                        "misc": "misc_regen", "note": rocket * 3000})
    picks = legal_picks(reply, observation())
    assert picks["arm"] == "arm_cleaver"          # the picks survive
    assert picks["note"] == rocket * MAX_NOTE_RUNES
    picks["note"].encode("utf-8", "strict")       # never a split codepoint
    frame = json.dumps({"phase": "draft", "picks": [picks]})
    assert len(frame.encode("utf-8")) <= catalog.MAX_DRAFT_FRAME_BYTES
    # and the server keeps it whole: nothing left to truncate
    assert draft.truncate_note(picks["note"]) == picks["note"]


# -- the call, the retry, the fallback ---------------------------------------

async def test_valid_reply_is_used_with_one_call():
    p = policy('{"arm":"arm_blaster","tail":"tail_rotor","misc":"misc_focus"}')
    picks = await p.on_draft(observation())
    assert picks == {"arm": "arm_blaster", "tail": "tail_rotor",
                     "misc": "misc_focus"}
    assert len(p._transport.bodies) == 1
    body = p._transport.bodies[0]
    assert body["model"] == "claude-sonnet-4-5"
    assert body["max_tokens"] == 400
    assert "temperature" not in body


async def test_fenced_reply_is_accepted():
    p = policy('```json\n{"arm":"arm_blaster","tail":"tail_rotor",'
               '"misc":"misc_focus"}\n```')
    assert (await p.on_draft(observation()))["arm"] == "arm_blaster"


async def test_malformed_reply_retries_once_at_temperature_zero():
    p = policy("no json here",
               '{"arm":"arm_needler","tail":"tail_plate","misc":"misc_regen"}')
    picks = await p.on_draft(observation())
    assert picks["arm"] == "arm_needler"
    assert len(p._transport.bodies) == 2
    second = p._transport.bodies[1]
    assert second["temperature"] == 0
    assert "Reply with the JSON object only." in second["system"]


async def test_two_failures_fall_back_to_the_scripted_rule(capsys):
    obs = observation(seat=2)  # burst
    p = policy("nope", "still nope")
    picks = await p.on_draft(obs)
    assert picks == forge_picks(obs)
    assert len(p._transport.bodies) == 2
    assert "draft_fallback=scripted" in capsys.readouterr().err


async def test_timeout_falls_back_without_hanging(monkeypatch):
    import players.derk_player as derk_player

    monkeypatch.setattr(derk_player, "CALL_TIMEOUT_SECONDS", 0.05)
    obs = observation(seat=0)
    p = policy("__hang__", "__hang__")
    picks = await asyncio.wait_for(p.on_draft(obs), 5)
    assert picks == forge_picks(obs)


async def test_transport_error_falls_back():
    obs = observation(seat=1)
    p = policy(IOError("anthropic HTTP 500"), IOError("again"))
    assert await p.on_draft(obs) == forge_picks(obs)


# -- the server's draft deadline bounds our own budget -----------------------

def test_call_timeout_is_capped_by_the_servers_draft_deadline():
    """The server hands the seat the neutral loadout the moment its own
    deadline passes, so our call budget must never exceed it."""
    assert call_timeout({"deadline_ms": 45000}) == CALL_TIMEOUT_SECONDS
    assert call_timeout({}) == CALL_TIMEOUT_SECONDS          # none offered
    assert call_timeout({"deadline_ms": "soon"}) == CALL_TIMEOUT_SECONDS
    # the certification fixture's deadline: one short call, then no retry
    assert call_timeout({"deadline_ms": 5000}) == pytest.approx(3.5)
    assert call_timeout({"deadline_ms": 5000}, elapsed=3.4) is None
    # the schema minimum leaves no room for a call at all
    assert call_timeout({"deadline_ms": defaults.MIN_DRAFT_DEADLINE_MS}) \
        is None
    # and the two-attempt worst case still fits the default 45 s deadline
    assert 2 * call_timeout({"deadline_ms": 45000}) + \
        DEADLINE_SAFETY_SECONDS <= 45.0


@pytest.mark.slow
async def test_under_the_certification_deadline_one_short_call_then_scripted(
        capsys):
    """The cert fixture seats the keyed drafter under
    ``draft_deadline_ms: 5000``. A model that does not answer must cost
    that seat less than the server's deadline, not 2 x 20 s."""
    obs = dict(observation(seat=2), deadline_ms=5000)
    p = policy("__hang__", "__hang__")
    started = time.monotonic()
    picks = await p.on_draft(obs)
    elapsed = time.monotonic() - started
    assert picks == forge_picks(obs)
    assert len(p._transport.bodies) == 1     # one call, no retry: no room
    assert elapsed < 5.0, elapsed
    assert "reason=timeout" in capsys.readouterr().err


async def test_a_deadline_too_short_for_any_call_skips_the_llm(capsys):
    obs = dict(observation(seat=0), deadline_ms=1000)
    p = policy('{"arm":"arm_blaster","tail":"tail_rotor","misc":"misc_focus"}')
    picks = await p.on_draft(obs)
    assert picks == forge_picks(obs)
    assert p._transport.bodies == []         # no call was made at all
    assert "draft_fallback=scripted reason=no_time" in capsys.readouterr().err


async def test_the_default_deadline_leaves_both_attempts_intact(monkeypatch):
    import players.derk_player as derk_player

    monkeypatch.setattr(derk_player, "CALL_TIMEOUT_SECONDS", 0.05)
    obs = observation(seat=1)
    assert obs["deadline_ms"] == 45000
    p = policy("__hang__", "__hang__")
    assert await p.on_draft(obs) == forge_picks(obs)
    assert len(p._transport.bodies) == 2


async def test_missing_api_key_makes_no_call_at_all(capsys):
    obs = observation(seat=1)
    transport = Transport('{"arm":"arm_cleaver","tail":"tail_plate",'
                          '"misc":"misc_regen"}')
    p = PromptDraftPolicy("derk-drafter-v1", micro=lambda t, rows: [],
                          api_key=None, transport=None)
    picks = await p.on_draft(obs)
    assert picks == forge_picks(obs)
    assert transport.bodies == []
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err


# -- the fallback log line: one shape, one closed vocabulary -----------------

async def _fallback_reason(capsys, monkeypatch, kind):
    """Drive on_draft into each fallback and return the logged reason."""
    obs = observation(seat=2)
    if kind == "no_key":
        p = PromptDraftPolicy("derk-drafter-v1", micro=lambda t, rows: [],
                              api_key=None, transport=None)
    elif kind == "no_time":
        obs = dict(obs, deadline_ms=1000)
        p = policy("unused")
    elif kind == "timeout":
        import players.derk_player as derk_player
        monkeypatch.setattr(derk_player, "CALL_TIMEOUT_SECONDS", 0.05)
        p = policy("__hang__", "__hang__")
    elif kind == "parse":
        p = policy("sure! here you go", "{broken")
    elif kind == "illegal":
        body = ('{"arm":"arm_nope","tail":"tail_plate","misc":"misc_regen"}')
        p = policy(f"here you go: {body}", body)
    elif kind == "transport":
        p = policy(IOError("anthropic HTTP 500"), OSError("connection reset"))
    assert await p.on_draft(obs) == forge_picks(obs)
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines()
             if ln.startswith("draft_fallback=scripted ")]
    assert len(lines) == 1, err
    return lines[0]


@pytest.mark.parametrize("kind", FALLBACK_REASONS)
async def test_every_fallback_logs_its_own_reason(capsys, monkeypatch, kind):
    """docs/DRAFT.md pins this line as the ONLY record of a player-side
    LLM -> scripted fallback (the reply the player sends is legal, so the
    server records fallback: false). One shape, one closed vocabulary,
    one reason per cause."""
    line = await _fallback_reason(capsys, monkeypatch, kind)
    assert line.startswith(f"draft_fallback=scripted reason={kind} picks=")


def test_the_fallback_vocabulary_is_closed_and_documented():
    assert FALLBACK_REASONS == ("no_key", "no_time", "timeout", "parse",
                                "illegal", "transport")
    drafted = (REPO_ROOT / "docs" / "DRAFT.md").read_text()
    for reason in FALLBACK_REASONS:
        assert f"`{reason}`" in drafted, reason
    # and the doc says which record phase 60 must count
    assert "draft_fallback=scripted" in drafted
    assert "results.draft_fallbacks" in drafted


async def test_a_broken_object_is_parse_not_illegal(capsys, monkeypatch):
    """The reviewer's mislabel: the old rule was "contains a '{' ->
    illegal", so a reply whose object never parsed was reported as an
    illegal pick. It is a parse failure."""
    p = policy("{broken", '{"arm":')
    obs = observation(seat=2)
    assert await p.on_draft(obs) == forge_picks(obs)
    assert "draft_fallback=scripted reason=parse" in capsys.readouterr().err


async def test_a_prose_wrapped_legal_reply_never_reaches_the_fallback(capsys):
    """The other half: prose around a legal object is now accepted, so it
    is neither `parse` nor `illegal` — it is a draft."""
    body = ('{"arm":"arm_cleaver","tail":"tail_plate","misc":"misc_regen"}')
    p = policy(f"Here is my draft: {body}")
    assert (await p.on_draft(observation(seat=2)))["arm"] == "arm_cleaver"
    err = capsys.readouterr().err
    assert "draft_fallback=scripted" not in err
    assert "attempt=1" in err


async def test_request_body_never_contains_a_real_player_name():
    """Two name spaces: what goes to the model is the alias-only draft
    observation, minus the deadline."""
    obs = observation(seat=3)
    p = policy("nope", "nope")
    await p.on_draft(obs)
    for body in p._transport.bodies:
        blob = json.dumps(body)
        for name in REAL_NAMES:
            assert name not in blob
        user = json.loads(body["messages"][0]["content"])
        assert "deadline_ms" not in user
        assert "Cog-Delta" in blob  # the alias IS there
        assert "arm_cleaver" in blob  # and the catalog


def test_the_two_prompts_are_distinct_and_metagamer_extends_drafter():
    assert set(PROMPTS) == {"derk-drafter-v1", "derk-metagamer-v1"}
    drafter, metagamer = PROMPTS["derk-drafter-v1"], PROMPTS["derk-metagamer-v1"]
    assert drafter != metagamer
    assert "Think about the metagame" in metagamer
    assert "Think about the metagame" not in drafter
    for prompt in PROMPTS.values():
        assert "No prose, no code fences, no markdown." in prompt
        assert '{"arm":"<id>","tail":"<id>","misc":"<id>"' in prompt


# -- the scripted draft rules ------------------------------------------------

def test_forge_table_covers_the_three_seated_roles():
    for seat, expected in enumerate([
            {"arm": "arm_blaster", "tail": "tail_plate",
             "misc": "misc_battery"},
            {"arm": "arm_needler", "tail": "tail_rotor",
             "misc": "misc_focus"},
            {"arm": "arm_blaster", "tail": "tail_stinger",
             "misc": "misc_battery"}]):
        assert forge_picks(observation(seat)) == expected
    # an unknown role falls back to the neutral loadout
    assert forge_picks({"hero": {"role": "tank"}}) == dict(
        catalog.NEUTRAL_PICKS)


def test_brawler_rule_is_derived_from_the_observed_base_stats():
    support = observation(0)   # 500 hp, +100 hp/level
    assassin = observation(1)  # 400 hp, +100 hp/level
    burst = observation(2)     # 400 hp, +75 hp/level
    assert brawler_picks(support)["arm"] == "arm_cleaver"
    assert brawler_picks(assassin)["arm"] == "arm_needler"
    assert brawler_picks(support)["tail"] == "tail_plate"
    assert brawler_picks(burst)["tail"] == "tail_rotor"
    assert brawler_picks(support)["misc"] == "misc_focus"
    assert brawler_picks(burst)["misc"] == "misc_regen"
    assert brawler_picks(support)["note"] == "brawl build"


@pytest.mark.parametrize("seat", range(defaults.NUM_SEATS))
@pytest.mark.parametrize("rule", [forge_picks, brawler_picks])
def test_scripted_draft_rules_are_always_legal(rule, seat):
    obs = observation(seat)
    picks = rule(obs)
    assert catalog.normalized_picks(picks) is not None, picks
    assert len(draft.truncate_note(picks.get("note"))) <= \
        catalog.MAX_NOTE_RUNES


def test_scripted_rules_are_deterministic():
    obs = observation(1)
    assert [forge_picks(obs) for _ in range(5)] == [forge_picks(obs)] * 5
    assert [brawler_picks(obs) for _ in range(5)] == [brawler_picks(obs)] * 5


# -- env switching -----------------------------------------------------------

def test_resolve_mode_defaults_to_puffer_forge():
    assert resolve_mode({}) == ("scripted", DEFAULT_SCRIPTED)
    assert DEFAULT_SCRIPTED == "puffer-forge"


def test_resolve_mode_prompt_wins_over_scripted(capsys):
    assert resolve_mode({"PLAYER_PROMPT": "derk-metagamer-v1",
                         "PLAYER_SCRIPTED": "lane-brawler"}) == \
        ("prompt", "derk-metagamer-v1")
    assert "PLAYER_PROMPT wins" in capsys.readouterr().err


@pytest.mark.parametrize("name", SCRIPTED_NAMES)
def test_resolve_mode_accepts_every_scripted_name(name):
    assert resolve_mode({"PLAYER_SCRIPTED": name}) == ("scripted", name)


def test_unknown_names_raise_with_the_legal_list():
    with pytest.raises(PlayerError, match="lane-brawler"):
        resolve_mode({"PLAYER_SCRIPTED": "typo"})
    with pytest.raises(PlayerError, match="derk-drafter-v1"):
        resolve_mode({"PLAYER_PROMPT": "typo"})


def test_main_exits_2_on_an_unknown_baseline(monkeypatch, capsys):
    from players import derk_player

    monkeypatch.setenv("PLAYER_SCRIPTED", "not-a-baseline")
    assert derk_player.main() == 2
    err = capsys.readouterr().err
    assert "unknown PLAYER_SCRIPTED" in err
    assert "puffer-forge, lane-brawler" in err


def test_scripted_policy_rejects_an_unknown_name():
    with pytest.raises(PlayerError):
        ScriptedDraftPolicy("nope", micro=lambda t, rows: [])
