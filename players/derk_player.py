"""The cogame-derks-gym policy: ``python -m players.derk_player``.

ONE entrypoint, one image, env-switched — the draft is the decision an
LLM makes, the per-tick micro is always local:

    PLAYER_PROMPT=<prompt name>      LLM draft + puffernet micro  (champion)
    PLAYER_JEV=true                  Jev draft + puffernet micro
    PLAYER_SCRIPTED=<baseline name>  scripted draft + its micro    (filler)

All unset defaults to ``PLAYER_SCRIPTED=puffer-forge``, so a bare
``docker run`` plays. Jev takes priority, then ``PLAYER_PROMPT`` wins over
``PLAYER_SCRIPTED``; that choice is
logged. An unknown ``PLAYER_SCRIPTED`` exits 2 with the legal names — a
typo must fail loudly, not silently ship a different policy.

Why the split: a MOBA tick is 100 ms and an episode is 6000 ticks. No LLM
plays that, and pretending otherwise would produce a coworld whose
champions are 6000 NOOPs. The metagame is where a prompt has real
leverage — 4^3 = 64 loadouts per hero, counter-drafting against an unseen
opponent, one decision that shapes the whole match.

Hosted prompt and Jev policies use the sidecar endpoint injected as
``AWS_ENDPOINT_URL_BEDROCK_RUNTIME``. Prompt calls use ``/v1/messages``;
Jev uses ``/v1/systemone``. Without a runtime endpoint or a direct local
key, a policy drafts with its scripted rule — invisible to
``results.draft_fallbacks``, which counts server-side substitutions only
(cogolf, 2026-08-24). ``provider_from_env`` picks the prompt transport; both
share the same prompt, the same tolerant parse, the same single retry at
temperature 0 and the same deadline-derived timeout.

Degrade, never hang: the LLM path has a 20 s per-call timeout, ONE retry
at temperature 0, and then falls back to ``puffer-forge``'s draft rule.
Each call's timeout is additionally capped by what is left of the
SERVER's own draft deadline (the observation's ``deadline_ms``, see
``call_timeout``), so a short deadline — the certification fixture runs
5 s — buys one short call instead of two the server stopped waiting for.
No provider at all means no call is made. The decision runs off the
websocket read loop (``players/client.py``), so a slow call cannot cost
the seat its socket to the server's ping/pong heartbeat.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time

from .baseline_player import BaselinePolicy
from .client import PlayerError, run_policy_main, seed_from_env
from .scripted_player import ScriptedPolicy

# -- the LLM contract --------------------------------------------------------

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
MODEL = "claude-sonnet-4-5"

# Optional direct AWS Bedrock mode. Hosted players use /v1/messages on the
# sidecar instead; these ids and the InvokeModel wire format are for an
# explicitly selected direct Bedrock client.
BEDROCK_ANTHROPIC_VERSION = "bedrock-2023-05-31"
BEDROCK_DEFAULT_REGION = "us-west-2"
# Inference profiles, tried in order within one attempt: model access is a
# per-account subscription and hosted capacity is shared, so a 403/429 on
# the first id must not idle the champion for the whole episode. Sonnet 4.5
# is the design note's model; the Haiku fallback is the id a shipped
# coworld (cogame-factorio) uses, and falling back to it can only LOWER
# the per-call cost. One call per episode either way.
BEDROCK_MODEL_CANDIDATES = (
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
)
PROVIDERS = ("anthropic", "bedrock", "sidecar", "none")

MAX_TOKENS = 400
CALL_TIMEOUT_SECONDS = 20.0
# The draft observation carries the SERVER's deadline (`deadline_ms`); our
# own call budget is capped by what is left of it. These two carve out the
# rest: the margin kept for encoding and sending the reply frame (plus the
# scripted fallback), and the floor below which starting a call is not
# worth it at all.
DEADLINE_SAFETY_SECONDS = 1.5
MIN_CALL_SECONDS = 1.0
RETRY_REMINDER = "Reply with the JSON object only."
# The CLOSED vocabulary of the player-side fallback log line
#     draft_fallback=scripted reason=<one of these> picks={...}
# on stderr. This line is the ONLY record of an LLM->scripted fallback:
# the reply the player then sends is a legal scripted pick, so the
# server records fallback: false and results.draft_fallbacks counts
# server-side neutral substitutions only (docs/DRAFT.md, "Two kinds of
# fallback"). Phase 60 counts LLM usage from player logs, not results.
FALLBACK_REASONS = ("no_key", "no_time", "timeout", "parse", "illegal",
                    "transport")
# The server's cap on the one free-text field (docs/DRAFT.md); mirrored
# here so an over-long note is trimmed before it is sent rather than
# pushing the frame past the 4096-byte drop, which would cost the seat
# its picks too.
MAX_NOTE_RUNES = 120

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

def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def provider_from_env(env: dict | None = None) -> str:
    """Which LLM transport this process can use: one of PROVIDERS.

    1. ``COGAME_LLM_PROVIDER`` is an explicit override (an unknown value is
       logged and ignored rather than silently disabling the champion);
    2. A runtime sidecar endpoint -> ``sidecar`` (the hosted league path);
    3. ``ANTHROPIC_API_KEY`` present -> ``anthropic`` (local/dev);
    4. otherwise ``none`` -> no call is made at all.
    """
    env = os.environ if env is None else env
    if (env.get("AWS_ENDPOINT_URL_BEDROCK_RUNTIME") or "").strip():
        return "sidecar"
    explicit = (env.get("COGAME_LLM_PROVIDER") or "").strip().lower()
    if explicit in PROVIDERS:
        return explicit
    if explicit:
        print(f"COGAME_LLM_PROVIDER={explicit!r} is not one of "
              f"{PROVIDERS}; detecting the provider from the environment",
              file=sys.stderr)
    if (env.get("ANTHROPIC_API_KEY") or "").strip():
        return "anthropic"
    return "none"


def bedrock_endpoint(env: dict | None = None) -> str:
    """The sidecar endpoint, else the regional Bedrock runtime default."""
    env = os.environ if env is None else env
    region = ((env.get("AWS_REGION") or "").strip()
              or (env.get("AWS_DEFAULT_REGION") or "").strip()
              or BEDROCK_DEFAULT_REGION)
    endpoint = ((env.get("AWS_ENDPOINT_URL_BEDROCK_RUNTIME") or "").strip()
                or f"https://bedrock-runtime.{region}.amazonaws.com")
    return endpoint.rstrip("/")


def bedrock_models(env: dict | None = None) -> list[str]:
    """The Bedrock model ids to try, in order: a pinned id first (the
    platform sets ``BEDROCK_MODEL`` when the sidecar pins one), then the
    shared candidates."""
    env = os.environ if env is None else env
    pinned = [(env.get("BEDROCK_MODEL") or "").strip(),
              (env.get("COGAME_LLM_MODEL") or "").strip()]
    return list(dict.fromkeys(
        [model for model in pinned if model] + list(BEDROCK_MODEL_CANDIDATES)))


def model_for_provider(provider: str, env: dict | None = None) -> str:
    if provider == "sidecar":
        env = os.environ if env is None else env
        return (env.get("BEDROCK_MODEL") or "").strip() or "anthropic/claude-haiku-4.5"
    return bedrock_models(env)[0] if provider == "bedrock" else MODEL


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
    """Strip ONE leading/trailing code fence (kept as the cheap,
    exactly-specified normalisation before the object scan)."""
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


def first_json_object(text: str) -> dict | None:
    """The FIRST balanced JSON object anywhere in ``text``, or None.

    Tolerant parsing: a model that wraps its answer in prose ("Here is my
    draft: {...}"), signs off after a code fence, or answers with a
    one-element list still gets its draft counted.
    ``json.JSONDecoder.raw_decode`` parses one value starting at a
    candidate ``{`` and stops at its end, so nesting, strings and escapes
    are handled by the JSON parser itself instead of a brace counter, and
    trailing prose is simply ignored.

    Two documented choices, so the behaviour is testable:
    * two objects in one reply -> the FIRST one that parses wins (a model
      that offers alternatives gets its first answer, never a merge);
    * an object inside an array (``[{...}]``) is accepted — the object is
      found at its own ``{``.
    """
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text, start)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def legal_picks(text: str, observation: dict) -> dict | None:
    """Parse one reply into legal picks, or None.

    Exactly the legality check the server applies, against the catalog
    the seat was actually given: one JSON object (extracted from whatever
    prose or fence surrounds it), one id per slot, each an id of the
    MATCHING slot.
    """
    payload = first_json_object(strip_one_fence(text))
    if payload is None:
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
        # The server truncates to 120 runes on receipt and that stays
        # authoritative, but a frame larger than 4096 bytes is dropped
        # BEFORE the JSON parse and costs the seat its picks, not just
        # its note. Slicing a str slices Unicode scalars, so this can
        # never split a codepoint.
        picks["note"] = note[:MAX_NOTE_RUNES]
    return picks


def _prompt_payload(observation: dict) -> str:
    """The user message: the draft observation as compact JSON, minus the
    deadline (a model does not need to reason about our wall clock)."""
    return json.dumps(
        {k: v for k, v in observation.items() if k != "deadline_ms"},
        separators=(",", ":"))


def call_timeout(observation: dict, elapsed: float = 0.0) -> float | None:
    """Timeout for the next LLM call, or None for "no time left".

    Our own 20 s per-call budget, capped by what is left of the SERVER's
    draft deadline (the observation's ``deadline_ms``) minus the margin
    that sending the reply needs. A seat that overruns the server's
    deadline gets the neutral loadout, so overrunning it is strictly worse
    than a shorter call: under the certification fixture's
    ``draft_deadline_ms: 5000`` this yields ONE ~3.5 s call and then the
    scripted rule, instead of two 20 s calls the server stopped waiting
    for. An observation with no usable ``deadline_ms`` keeps the full
    20 s budget.
    """
    deadline_ms = observation.get("deadline_ms")
    if isinstance(deadline_ms, bool) or not isinstance(
            deadline_ms, (int, float)):
        return CALL_TIMEOUT_SECONDS
    remaining = deadline_ms / 1000.0 - DEADLINE_SAFETY_SECONDS - elapsed
    if remaining < MIN_CALL_SECONDS:
        return None
    return min(CALL_TIMEOUT_SECONDS, remaining)


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
    """A champion: one model call for the draft, local micro after.

    The micro layer is byte-identical to the scripted baseline's
    (``MobaBrain`` on the vendored pretrained weights) — the prompt is
    the whole strategy.

    ``provider`` selects the transport (see ``provider_from_env``):
    ``sidecar`` is the hosted league path, ``anthropic`` the local one,
    and ``bedrock`` is an explicitly selected direct AWS route;
    ``none`` means no call is made. Defaults to ``anthropic`` when an API
    key was passed, so a direct construction keeps its old meaning.
    """

    def __init__(self, prompt_name: str, micro, api_key: str | None = None,
                 fallback=forge_picks, transport=None,
                 provider: str | None = None, env: dict | None = None):
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
        self._env = env
        self.provider = provider or ("anthropic" if api_key else "none")
        self.model = model_for_provider(self.provider, env)
        self.last_request: dict | None = None

    def _scripted_fallback(self, observation: dict, reason: str) -> dict:
        """The one place a player-side fallback is recorded, so the log
        line phase 60 counts has exactly one shape and one vocabulary."""
        assert reason in FALLBACK_REASONS, reason
        picks = self._fallback(observation)
        print(f"draft_fallback=scripted reason={reason} picks={picks}",
              file=sys.stderr)
        return picks

    async def on_draft(self, observation: dict) -> dict:
        if self.provider == "none" and self._transport is None:
            # A hosted pod gets a Bedrock sidecar, never the key: this line
            # is the symptom to grep for when champions play scripted.
            print("no LLM provider: ANTHROPIC_API_KEY is not set and no "
                  "Bedrock sidecar was granted (USE_BEDROCK / "
                  "AWS_ENDPOINT_URL_BEDROCK_RUNTIME / "
                  "AWS_BEARER_TOKEN_BEDROCK): no LLM call at all",
                  file=sys.stderr)
            return self._scripted_fallback(observation, "no_key")
        user = _prompt_payload(observation)
        started = time.monotonic()
        reason = "no_time"
        for attempt in (1, 2):
            timeout = call_timeout(observation, time.monotonic() - started)
            if timeout is None:
                print(f"draft={self.prompt_name} attempt {attempt} skipped: "
                      f"no time left inside the server's "
                      f"{observation.get('deadline_ms')} ms draft deadline",
                      file=sys.stderr)
                break
            reminder = attempt == 2
            try:
                text = await asyncio.wait_for(
                    self._call(user, reminder=reminder), timeout)
            except (asyncio.TimeoutError, TimeoutError):
                reason = "timeout"
                text = None
            except Exception as exc:
                reason = "transport"
                print(f"draft={self.prompt_name} attempt={attempt} "
                      f"transport error: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                text = None
            else:
                picks = legal_picks(text, observation)
                if picks is not None:
                    print(f"draft={self.prompt_name} attempt={attempt} "
                          f"picks={picks}", file=sys.stderr)
                    return picks
                # "parse": no JSON object could be extracted at all;
                # "illegal": an object was extracted, its ids were not
                # this seat's catalog. A prose-wrapped object is now
                # parsed (F1), so it can only be illegal on its ids.
                reason = ("illegal"
                          if first_json_object(text or "") is not None
                          else "parse")
            if attempt == 1:
                print(f"draft={self.prompt_name} attempt 1 failed "
                      f"({reason}); will retry once at temperature 0",
                      file=sys.stderr)
        return self._scripted_fallback(observation, reason)

    async def _call(self, user: str, *, reminder: bool) -> str:
        """One request, built once, sent through whichever transport this
        process has. The body is provider-neutral apart from the model id
        and the version key direct Bedrock's InvokeModel wants."""
        body = {
            "model": self.model,
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
        if self.provider == "sidecar":
            return await _sidecar_call(body, env=self._env)
        if self.provider == "bedrock":
            return await _bedrock_call(body, env=self._env)
        return await _anthropic_call(body, self._api_key)

    def __call__(self, tick: int, obs_rows: list) -> list:
        return self._micro(tick, obs_rows)


class JevDraftPolicy:
    """Rank the 64 legal loadouts from the ordinary private draft view."""

    def __init__(self, micro, transport=None, env: dict | None = None):
        self._micro = micro
        self._transport = transport
        self._env = os.environ if env is None else env

    async def on_draft(self, observation: dict) -> dict:
        endpoint = self._env.get("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", "").strip()
        key = self._env.get("TYPESAFE_API_KEY", "").strip()
        if not endpoint and not key and self._transport is None:
            print("draft_fallback=scripted reason=no_key", file=sys.stderr)
            return forge_picks(observation)
        timeout = call_timeout(observation)
        if timeout is None:
            print("draft_fallback=scripted reason=no_time", file=sys.stderr)
            return forge_picks(observation)

        catalog = observation["catalog"]
        candidates = {
            str(arm + 4 * tail + 16 * misc): {
                "arm": catalog["arm"][arm]["id"],
                "tail": catalog["tail"][tail]["id"],
                "misc": catalog["misc"][misc]["id"],
            }
            for misc in range(4) for tail in range(4) for arm in range(4)
        }
        body = {
            "model": (self._env.get("BEDROCK_MODEL", "").strip()
                      if endpoint else self._env.get("TYPESAFE_DEFAULT_MODEL", "").strip())
                     or ("typesafe/jev-1.13" if endpoint or self._transport
                         else "jev-latest"),
            "state": json.dumps({
                "observation": {k: v for k, v in observation.items()
                                if k != "deadline_ms"},
                "candidates": candidates,
            }, separators=(",", ":")),
            "questions": {"loadout": {
                "type": "choice",
                "instructions": "Choose the loadout most likely to win this hidden "
                                "draft and the ensuing MOBA match for this hero.",
                "criteria": {label: ", ".join(picks.values())
                             for label, picks in candidates.items()},
            }},
        }
        if self._transport is not None:
            response = await asyncio.wait_for(self._transport(body), timeout)
        else:
            import aiohttp

            headers = {"content-type": "application/json"}
            if not endpoint:
                endpoint = self._env.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai")
                headers["authorization"] = f"Bearer {key}"
            async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.post(endpoint.rstrip("/") + "/v1/systemone",
                                        headers=headers, json=body) as reply:
                    reply.raise_for_status()
                    response = await reply.json(content_type=None)
        answer = response["answers"]["loadout"]
        probabilities = answer["probabilities"]
        confidence = answer["confidence"]
        if (answer["type"] != "choice" or set(probabilities) != set(candidates)
                or type(confidence) not in (int, float)
                or not math.isfinite(confidence) or not 0 <= confidence <= 1
                or any(type(p) not in (int, float) or not math.isfinite(p)
                       or not 0 <= p <= 1 for p in probabilities.values())
                or abs(sum(probabilities.values()) - 1) > 0.02):
            raise ValueError("Jev returned an invalid loadout choice")
        chosen = max(candidates, key=probabilities.__getitem__)
        print(f"draft=jev loadout={chosen}", file=sys.stderr)
        return {**candidates[chosen], "note": f"Jev loadout {chosen}"}

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


async def _sidecar_call(body: dict, env: dict | None = None) -> str:
    """Send the ordinary Anthropic Messages body to the hosted player sidecar."""
    import aiohttp

    env = os.environ if env is None else env
    endpoint = env["AWS_ENDPOINT_URL_BEDROCK_RUNTIME"].rstrip("/")
    async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=CALL_TIMEOUT_SECONDS)) as session:
        async with session.post(
                f"{endpoint}/v1/messages",
                headers={"content-type": "application/json",
                         "anthropic-version": ANTHROPIC_VERSION},
                json=body) as reply:
            reply.raise_for_status()
            payload = await reply.json(content_type=None)
    return "".join(
        block["text"] for block in payload["content"]
        if block["type"] == "text")


async def _bedrock_call(body: dict, env: dict | None = None) -> str:
    """One Bedrock InvokeModel call; returns the concatenated text.

    The request is the Anthropic Messages body with the model id moved
    into the URL and ``anthropic_version`` added, which is what
    InvokeModel expects. Model ids are tried in order (see
    BEDROCK_MODEL_CANDIDATES): a 403/404/429 on one profile falls through
    to the next instead of idling the champion, and only if every
    candidate fails does this raise — which the caller logs as
    ``reason=transport``.
    """
    import aiohttp

    env = os.environ if env is None else env
    endpoint = bedrock_endpoint(env)
    token = (env.get("AWS_BEARER_TOKEN_BEDROCK") or "").strip()
    headers = {"content-type": "application/json",
               "accept": "application/json"}
    if token:
        headers["authorization"] = f"Bearer {token}"
    payload = {key: value for key, value in body.items() if key != "model"}
    payload["anthropic_version"] = BEDROCK_ANTHROPIC_VERSION

    candidates = list(dict.fromkeys(
        [body.get("model")] + bedrock_models(env)))
    failures: list[str] = []
    timeout = aiohttp.ClientTimeout(total=CALL_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for model in [c for c in candidates if c]:
            async with session.post(
                    f"{endpoint}/model/{model}/invoke",
                    headers=headers, json=payload) as resp:
                if resp.status != 200:
                    detail = (await resp.text())[:200]
                    failures.append(f"{model}: HTTP {resp.status}: {detail}")
                    continue
                data = await resp.json(content_type=None)
            return "".join(
                block.get("text", "") for block in (data.get("content") or [])
                if isinstance(block, dict))
    raise IOError("bedrock invoke failed: " + "; ".join(failures))


# -- entry point ------------------------------------------------------------

def _micro_for(name: str, seed: int | None):
    """lane-brawler plays the hand-coded lane-push FSM; everything else
    plays the vendored pretrained network."""
    if name == "lane-brawler":
        return ScriptedPolicy(seed)
    return BaselinePolicy(seed if seed is not None else 1)


def resolve_mode(env: dict | None = None) -> tuple[str, str]:
    """``("jev"|"prompt"|"scripted", name)`` from the environment.

    Raises PlayerError on an unknown name — checked BEFORE anything
    expensive (a brain wasm instance) is built.
    """
    env = os.environ if env is None else env
    if _truthy(env.get("PLAYER_JEV")):
        return "jev", "jev"
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
    if kind == "jev":
        print("policy: Jev draft + puffernet micro", file=sys.stderr)
        return JevDraftPolicy(_micro_for(name, seed))
    if kind == "prompt":
        provider = provider_from_env()
        model = model_for_provider(provider)
        print(f"policy: prompt {name} (LLM draft + puffernet micro); "
              f"provider={provider} model={model}"
              f"{' endpoint=' + bedrock_endpoint() if provider in ('bedrock', 'sidecar') else ''}",
              file=sys.stderr)
        return PromptDraftPolicy(
            name, _micro_for(name, seed),
            os.environ.get("ANTHROPIC_API_KEY", "").strip() or None,
            provider=provider)
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
