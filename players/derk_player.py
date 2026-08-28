"""The cogame-derks-gym policy: ``python -m players.derk_player``.

ONE entrypoint, one image, env-switched — the draft is the decision an
LLM makes, the per-tick micro is always local:

    PLAYER_PROMPT=<prompt name>      LLM draft + puffernet micro  (champion)
    PLAYER_SCRIPTED=<baseline name>  scripted draft + its micro    (filler)

Both unset defaults to ``PLAYER_SCRIPTED=puffer-forge``, so a bare
``docker run`` plays. Both set: ``PLAYER_PROMPT`` wins and the choice is
logged. An unknown ``PLAYER_SCRIPTED`` exits 2 with the legal names — a
typo must fail loudly, not silently ship a different policy.

Why the split: a MOBA tick is 100 ms and an episode is 6000 ticks. No LLM
plays that, and pretending otherwise would produce a coworld whose
champions are 6000 NOOPs. The metagame is where a prompt has real
leverage — 4^3 = 64 loadouts per hero, counter-drafting against an unseen
opponent, one decision that shapes the whole match.

Degrade, never hang: the LLM path has a 20 s per-call timeout, ONE retry
at temperature 0, and then falls back to ``puffer-forge``'s draft rule.
20 + 20 s fits inside the server's 45 s draft deadline, so a
doubly-failing champion still submits a legal loadout on time. No API key
at all means no call is made.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from .baseline_player import BaselinePolicy
from .client import PlayerError, run_policy_main, seed_from_env
from .scripted_player import ScriptedPolicy

# -- the LLM contract --------------------------------------------------------

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 400
CALL_TIMEOUT_SECONDS = 20.0
RETRY_REMINDER = "Reply with the JSON object only."

_SYSTEM_BASE = """\
You are drafting the loadout for one cog in Derk's Gym, a 3v3 MOBA skirmish on the
Puffer MOBA map. You pick exactly one ARM, one TAIL and one MISC item from the catalog
you are given. The items change your cog's physical stats for the whole match; you do
not control the cog after the draft, a trained network does.

Radiant and Dire each field three drafted cogs (support, assassin, burst) plus two
house-controlled cogs (tank, carry) that always run the neutral all-zero loadout.
The match is won by destroying the enemy Ancient (4500 HP, deals no damage); if
neither Ancient falls by the tick cap the team with more Ancient health wins.
Lane towers hit for 110-175 per shot with the same scan radius as a cog, so diving
one without a creep wave is death. Cogs regenerate 2 health and 2 mana per tick and
gain stats per level from their per-level gain values.

All six drafts are simultaneous and hidden: you cannot see what anyone else picked.
"""

_ANSWER_FORMAT = """
Answer with a single JSON object and nothing else:
{"arm":"<id>","tail":"<id>","misc":"<id>","note":"<=120 chars"}
Use only ids from the catalog you were given. No prose, no code fences, no markdown.
"""

_METAGAME = """
Think about the metagame before you answer. Assume opponents over-value raw damage
and under-value the per-level gain values, which compound: a cog that survives to
level 8 with a high hp_gain_per_level out-trades a cog that bought flat damage.
Your note field must name, in a few words, the enemy build you are countering.
"""

PROMPTS: dict[str, str] = {
    "derk-drafter-v1": _SYSTEM_BASE + _ANSWER_FORMAT,
    "derk-metagamer-v1": _SYSTEM_BASE + _METAGAME + _ANSWER_FORMAT,
}

# -- the scripted draft rules -----------------------------------------------

# puffer-forge: deterministic, straight off hero.role. No RNG.
FORGE_BY_ROLE: dict[str, dict[str, str]] = {
    "support": {"arm": "arm_blaster", "tail": "tail_plate",
                "misc": "misc_battery"},
    "assassin": {"arm": "arm_needler", "tail": "tail_rotor",
                 "misc": "misc_focus"},
    "burst": {"arm": "arm_blaster", "tail": "tail_stinger",
              "misc": "misc_battery"},
}
NEUTRAL = {"arm": "arm_none", "tail": "tail_none", "misc": "misc_none"}

SCRIPTED_NAMES = ("puffer-forge", "lane-brawler")
DEFAULT_SCRIPTED = "puffer-forge"


def forge_picks(observation: dict) -> dict:
    """puffer-forge's draft: a fixed table keyed by the hero's role."""
    role = (observation.get("hero") or {}).get("role")
    return dict(FORGE_BY_ROLE.get(role, NEUTRAL))


def brawler_picks(observation: dict) -> dict:
    """lane-brawler's draft: derived from the OBSERVED base stats, so it
    adapts if upstream's role table ever changes."""
    hero = observation.get("hero") or {}
    health = float(hero.get("base_health", 0) or 0)
    hp_gain = float(hero.get("hp_gain_per_level", 0) or 0)
    return {
        "arm": "arm_cleaver" if health >= 500 else "arm_needler",
        "tail": "tail_plate" if hp_gain >= 100 else "tail_rotor",
        "misc": "misc_regen" if health < 500 else "misc_focus",
        "note": "brawl build",
    }


SCRIPTED_DRAFTS = {
    "puffer-forge": forge_picks,
    "lane-brawler": brawler_picks,
}


# -- reply parsing ----------------------------------------------------------

def strip_one_fence(text: str) -> str:
    """Strip ONE leading/trailing code fence — the single documented
    tolerance, so it is testable."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped[3:]
    newline = body.find("\n")
    if newline != -1 and "`" not in body[:newline]:
        body = body[newline + 1:]  # drop a ```json language tag
    if body.rstrip().endswith("```"):
        body = body.rstrip()[:-3]
    return body.strip()


def legal_picks(text: str, observation: dict) -> dict | None:
    """Parse one reply into legal picks, or None.

    Exactly the legality check the server applies, against the catalog
    the seat was actually given: one JSON object, one id per slot, each
    an id of the MATCHING slot.
    """
    try:
        payload = json.loads(strip_one_fence(text))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    cat = observation.get("catalog") or {}
    picks: dict[str, str] = {}
    for slot in ("arm", "tail", "misc"):
        value = payload.get(slot)
        if not isinstance(value, str):
            return None
        ids = {entry.get("id") for entry in (cat.get(slot) or [])
               if isinstance(entry, dict)}
        if value.strip(" ") not in ids:
            return None
        picks[slot] = value.strip(" ")
    note = payload.get("note")
    if isinstance(note, str) and note:
        picks["note"] = note
    return picks


def _prompt_payload(observation: dict) -> str:
    """The user message: the draft observation as compact JSON, minus the
    deadline (a model does not need to reason about our wall clock)."""
    return json.dumps(
        {k: v for k, v in observation.items() if k != "deadline_ms"},
        separators=(",", ":"))


# -- the policies -----------------------------------------------------------

class ScriptedDraftPolicy:
    """A filler: a deterministic draft rule plus a local micro policy."""

    def __init__(self, name: str, micro):
        if name not in SCRIPTED_DRAFTS:
            raise PlayerError(
                f"unknown PLAYER_SCRIPTED {name!r}; legal names: "
                f"{', '.join(SCRIPTED_NAMES)}")
        self.name = name
        self._draft = SCRIPTED_DRAFTS[name]
        self._micro = micro

    def on_draft(self, observation: dict) -> dict:
        picks = self._draft(observation)
        print(f"draft={self.name} picks={picks}", file=sys.stderr)
        return picks

    def __call__(self, tick: int, obs_rows: list) -> list:
        return self._micro(tick, obs_rows)


class PromptDraftPolicy:
    """A champion: one Anthropic call for the draft, local micro after.

    The micro layer is byte-identical to the scripted baseline's
    (``MobaBrain`` on the vendored pretrained weights) — the prompt is
    the whole strategy.
    """

    def __init__(self, prompt_name: str, micro, api_key: str | None,
                 fallback=forge_picks, transport=None):
        if prompt_name not in PROMPTS:
            raise PlayerError(
                f"unknown PLAYER_PROMPT {prompt_name!r}; legal names: "
                f"{', '.join(sorted(PROMPTS))}")
        self.prompt_name = prompt_name
        self.system = PROMPTS[prompt_name]
        self._micro = micro
        self._api_key = api_key
        self._fallback = fallback
        # Injectable for tests; None means "real HTTP".
        self._transport = transport
        self.last_request: dict | None = None

    async def on_draft(self, observation: dict) -> dict:
        if not self._api_key and self._transport is None:
            print("ANTHROPIC_API_KEY is not set: no LLM call, using the "
                  "scripted draft rule (draft_fallback=scripted "
                  "reason=no_key)", file=sys.stderr)
            return self._fallback(observation)
        user = _prompt_payload(observation)
        for attempt in (1, 2):
            reminder = attempt == 2
            try:
                text = await asyncio.wait_for(
                    self._call(user, reminder=reminder),
                    CALL_TIMEOUT_SECONDS)
            except (asyncio.TimeoutError, TimeoutError):
                reason = "timeout"
                text = None
            except Exception as exc:
                reason = f"transport:{type(exc).__name__}"
                text = None
            else:
                picks = legal_picks(text, observation)
                if picks is not None:
                    print(f"draft={self.prompt_name} attempt={attempt} "
                          f"picks={picks}", file=sys.stderr)
                    return picks
                reason = "parse" if "{" not in (text or "") else "illegal"
            if attempt == 1:
                print(f"draft={self.prompt_name} attempt 1 failed "
                      f"({reason}); will retry once at temperature 0",
                      file=sys.stderr)
        picks = self._fallback(observation)
        print(f"draft_fallback=scripted reason={reason} picks={picks}",
              file=sys.stderr)
        return picks

    async def _call(self, user: str, *, reminder: bool) -> str:
        body = {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "system": self.system + (f"\n{RETRY_REMINDER}\n" if reminder
                                     else ""),
            "messages": [{"role": "user", "content": user}],
        }
        if reminder:
            body["temperature"] = 0
        self.last_request = body
        if self._transport is not None:
            return await self._transport(body)
        return await _anthropic_call(body, self._api_key)

    def __call__(self, tick: int, obs_rows: list) -> list:
        return self._micro(tick, obs_rows)


async def _anthropic_call(body: dict, api_key: str) -> str:
    """One Anthropic Messages call; returns the concatenated text."""
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=CALL_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
                ANTHROPIC_URL,
                headers={"x-api-key": api_key,
                         "anthropic-version": ANTHROPIC_VERSION,
                         "content-type": "application/json"},
                json=body) as resp:
            if resp.status != 200:
                detail = (await resp.text())[:200]
                raise IOError(f"anthropic HTTP {resp.status}: {detail}")
            payload = await resp.json()
    return "".join(
        block.get("text", "") for block in (payload.get("content") or [])
        if isinstance(block, dict))


# -- entry point ------------------------------------------------------------

def _micro_for(name: str, seed: int | None):
    """lane-brawler plays the hand-coded lane-push FSM; everything else
    plays the vendored pretrained network."""
    if name == "lane-brawler":
        return ScriptedPolicy(seed)
    return BaselinePolicy(seed if seed is not None else 1)


def resolve_mode(env: dict | None = None) -> tuple[str, str]:
    """``("prompt"|"scripted", name)`` from the environment.

    Raises PlayerError on an unknown name — checked BEFORE anything
    expensive (a brain wasm instance) is built.
    """
    env = os.environ if env is None else env
    prompt = (env.get("PLAYER_PROMPT") or "").strip()
    scripted = (env.get("PLAYER_SCRIPTED") or "").strip()
    if prompt:
        if scripted:
            print(f"both PLAYER_PROMPT={prompt} and "
                  f"PLAYER_SCRIPTED={scripted} are set; PLAYER_PROMPT wins",
                  file=sys.stderr)
        if prompt not in PROMPTS:
            raise PlayerError(
                f"unknown PLAYER_PROMPT {prompt!r}; legal names: "
                f"{', '.join(sorted(PROMPTS))}")
        return "prompt", prompt
    name = scripted or DEFAULT_SCRIPTED
    if name not in SCRIPTED_DRAFTS:
        raise PlayerError(
            f"unknown PLAYER_SCRIPTED {name!r}; legal names: "
            f"{', '.join(SCRIPTED_NAMES)}")
    return "scripted", name


def policy_from_env():
    kind, name = resolve_mode()
    seed = seed_from_env(default=1)
    if kind == "prompt":
        print(f"policy: prompt {name} (LLM draft + puffernet micro)",
              file=sys.stderr)
        return PromptDraftPolicy(
            name, _micro_for(name, seed),
            os.environ.get("ANTHROPIC_API_KEY", "").strip() or None)
    print(f"policy: scripted {name}", file=sys.stderr)
    return ScriptedDraftPolicy(name, _micro_for(name, seed))


def main() -> int:
    try:
        resolve_mode()
    except PlayerError as exc:
        # An unknown baseline/prompt name must fail loudly (exit 2), not
        # quietly ship a different policy.
        print(f"player failed: {exc}", file=sys.stderr)
        return 2
    return run_policy_main(policy_from_env)


if __name__ == "__main__":
    sys.exit(main())
