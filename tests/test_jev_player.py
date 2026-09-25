"""Jev drafts through the same private observation and reply as other players."""

import json

import pytest

from players.derk_player import JevDraftPolicy, resolve_mode
from tests.test_llm_player import FakeBedrock, observation


def test_jev_mode_is_a_normal_player_mode():
    assert resolve_mode({"PLAYER_JEV": "true"}) == ("jev", "jev")


async def test_jev_ranks_all_catalog_loadouts_without_private_names():
    requests = []

    async def respond(body):
        requests.append(body)
        return {"answers": {"loadout": {
            "type": "choice", "confidence": 1,
            "probabilities": {str(i): int(i == 57) for i in range(64)},
        }}}

    policy = JevDraftPolicy(micro=lambda tick, rows: [], transport=respond)
    picks = await policy.on_draft(observation(seat=2))
    assert picks == {"arm": "arm_blaster", "tail": "tail_stinger",
                     "misc": "misc_focus", "note": "Jev loadout 57"}
    body = requests[0]
    assert body["model"] == "typesafe/jev-1.13"
    assert len(body["questions"]["loadout"]["criteria"]) == 64
    state = json.loads(body["state"])
    assert state["observation"]["hero"]["role"] == "burst"
    assert "deadline_ms" not in state["observation"]
    for name in [f"champion-{i}" for i in range(6)]:
        assert name not in body["state"]
    assert policy(0, []) == []


async def test_jev_without_endpoint_plays_a_legal_baseline(monkeypatch):
    monkeypatch.delenv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    policy = JevDraftPolicy(micro=lambda tick, rows: [])
    picks = await policy.on_draft(observation())
    assert picks["arm"] == "arm_blaster"


async def test_jev_sidecar_uses_systemone_without_bearer(monkeypatch):
    import aiohttp

    fake = FakeBedrock((200, {"answers": {"loadout": {
        "type": "choice", "confidence": 1,
        "probabilities": {str(i): int(i == 0) for i in range(64)},
    }}}))
    monkeypatch.setattr(aiohttp, "ClientSession", fake)
    policy = JevDraftPolicy(micro=lambda tick, rows: [], env={
        "AWS_ENDPOINT_URL_BEDROCK_RUNTIME": "http://sidecar",
        "BEDROCK_MODEL": "typesafe/jev-1.13",
        "AWS_BEARER_TOKEN_BEDROCK": "placeholder",
    })
    picks = await policy.on_draft(observation())
    assert picks["arm"] == "arm_none"
    request = fake.requests[0]
    assert request["url"] == "http://sidecar/v1/systemone"
    assert request["body"]["model"] == "typesafe/jev-1.13"
    assert "authorization" not in request["headers"]


async def test_jev_local_key_uses_direct_systemone(monkeypatch):
    import aiohttp

    fake = FakeBedrock((200, {"answers": {"loadout": {
        "type": "choice", "confidence": 1,
        "probabilities": {str(i): int(i == 1) for i in range(64)},
    }}}))
    monkeypatch.setattr(aiohttp, "ClientSession", fake)
    policy = JevDraftPolicy(micro=lambda tick, rows: [], env={
        "TYPESAFE_BASE_URL": "http://typesafe",
        "TYPESAFE_API_KEY": "test-key",
        "TYPESAFE_DEFAULT_MODEL": "jev-latest",
    })
    assert (await policy.on_draft(observation()))["arm"] == "arm_blaster"
    request = fake.requests[0]
    assert request["url"] == "http://typesafe/v1/systemone"
    assert request["headers"]["authorization"] == "Bearer test-key"
    assert request["body"]["model"] == "jev-latest"


@pytest.mark.parametrize("answer", [
    {"type": "choice", "confidence": 0.8, "probabilities": {"0": 1}},
    {"type": "choice", "confidence": 1.5,
     "probabilities": {str(i): int(i == 0) for i in range(64)}},
])
async def test_jev_rejects_invalid_choice_payload(answer):
    async def respond(body):
        return {"answers": {"loadout": answer}}

    policy = JevDraftPolicy(micro=lambda tick, rows: [], transport=respond)
    with pytest.raises(ValueError):
        await policy.on_draft(observation())
