"""Tripwires: the manifest template stays in sync with the code.

The results schema is CLOSED (AGENTS.md): ``_results_doc``, the manifest
``results_schema`` and ``docker_smoke.sh``'s expected key set must all
list exactly the same keys, and the ``end_reason`` enum must match the
engine's. Every variant/certification ``game_config`` in the template must
parse through ``GameConfig.from_dict`` (with runner-injected tokens), and
``num_agents`` must live INSIDE every ``game_config`` — never at a
variant's top level, which ``CoworldVariant`` rejects
(cogame-goofspiel-oshi-zumo 0.1.0, 2026-08-26).
"""

import json
import re
from pathlib import Path
from typing import get_args

from cogame_derks_gym import defaults, draft
from cogame_derks_gym.config import GameConfig
from cogame_derks_gym.engine import (NOOP_CAUSES, STAT_NAMES, EndReason,
                                     EpisodeResult)
from cogame_derks_gym.server import GameServer

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(
    (REPO_ROOT / "coworld_manifest_template.json").read_text())
DOCKER_SMOKE = (REPO_ROOT / "tools" / "ci" / "docker_smoke.sh").read_text()
POLICIES = json.loads(
    (REPO_ROOT / "tools" / "ci" / "policies.json").read_text())


def _game_server():
    cfg = GameConfig.from_dict({
        "players": [{"name": f"p{i}"} for i in range(defaults.NUM_SEATS)],
        "tokens": [f"t{i}" for i in range(defaults.NUM_SEATS)],
        "seed": 1,
    })
    server = GameServer(cfg)
    server.draft_records = draft.neutral_records(cfg)
    return server


def _dummy_result():
    seats = defaults.NUM_SEATS
    return EpisodeResult(
        winner=0,
        end_reason="ancient",
        seat_scores=(1.0,) + (0.0,) * (seats - 1),
        seat_reward_sums=(0.0,) * seats,
        agent_stats=tuple({n: 0 for n in STAT_NAMES}
                          for _ in range(defaults.NUM_HEROES)),
        final_tick=1,
        ancient_healths=(1.0, 1.0),
        seat_noop_ticks=(0,) * seats,
        seat_dead=(False,) * seats,
        seat_noop_causes=tuple(dict.fromkeys(NOOP_CAUSES, 0)
                               for _ in range(seats)),
    )


def _schema_keys():
    schema = MANIFEST["game"]["results_schema"]
    assert schema["additionalProperties"] is False, \
        "results_schema must stay closed"
    return set(schema["required"]), set(schema["properties"])


# -- the closed results schema ----------------------------------------------

def test_results_doc_matches_manifest_results_schema():
    doc_keys = set(_game_server()._results_doc(_dummy_result()))
    required, properties = _schema_keys()
    assert doc_keys == required, sorted(doc_keys ^ required)
    assert doc_keys == properties, sorted(doc_keys ^ properties)
    assert {"draft", "draft_fallbacks"} <= doc_keys


def test_fault_results_doc_has_same_closed_key_set():
    server = _game_server()
    assert set(server._fault_results_doc(0)) == \
        set(server._results_doc(_dummy_result()))


def test_docker_smoke_expected_keys_match_results_doc():
    # third leg of the triple-sync rule: docker_smoke.sh's expected set
    match = re.search(r"expected = \{(.*?)\}", DOCKER_SMOKE, re.DOTALL)
    assert match, "docker_smoke.sh expected-keys block not found"
    smoke_keys = set(re.findall(r'"(\w+)"', match.group(1)))
    doc_keys = set(_game_server()._results_doc(_dummy_result()))
    assert smoke_keys == doc_keys, sorted(smoke_keys ^ doc_keys)


def test_end_reason_enum_matches_engine():
    schema_enum = set(
        MANIFEST["game"]["results_schema"]["properties"]["end_reason"]["enum"])
    assert schema_enum == set(get_args(EndReason))
    assert len(schema_enum) == 4, "end_reason is a CLOSED four-value enum"


def test_fallback_cause_enum_matches_the_draft_module():
    record_schema = (MANIFEST["game"]["results_schema"]["properties"]
                     ["draft"]["items"])
    enum = set(record_schema["properties"]["fallback_cause"]["enum"])
    assert enum == set(draft.FALLBACK_CAUSES)
    assert set(record_schema["properties"]["source"]["enum"]) == \
        set(draft.SOURCES)


def test_draft_record_schema_matches_a_real_record():
    record_schema = (MANIFEST["game"]["results_schema"]["properties"]
                     ["draft"]["items"])
    assert record_schema["additionalProperties"] is False
    cfg = GameConfig.from_dict({
        "players": [{"name": f"p{i}"} for i in range(defaults.NUM_SEATS)],
        "tokens": [f"t{i}" for i in range(defaults.NUM_SEATS)],
        "seed": 1,
    })
    record = draft.neutral_records(cfg)[0]
    assert set(record) == set(record_schema["required"])
    assert set(record) == set(record_schema["properties"])
    applied_schema = record_schema["properties"]["applied"]
    assert set(record["applied"]) == set(applied_schema["required"])


# -- seat count and the platform budget --------------------------------------

def test_num_agents_is_inside_every_game_config_and_absent_above_it():
    for variant in MANIFEST["variants"]:
        assert "num_agents" not in variant, \
            f"variant {variant['id']}: num_agents must live inside game_config"
        assert variant["game_config"]["num_agents"] == defaults.NUM_SEATS
        assert len(variant["game_config"]["players"]) == defaults.NUM_SEATS
    cert = MANIFEST["certification"]
    assert cert["game_config"]["num_agents"] == defaults.NUM_SEATS
    assert len(cert["players"]) == defaults.NUM_SEATS
    assert len(cert["game_config"]["players"]) == defaults.NUM_SEATS


def test_docker_smoke_seat_count_agrees():
    assert re.search(r'seats_expected="\$\{SMOKE_SEATS:-6\}"', DOCKER_SMOKE), \
        "docker_smoke.sh's SMOKE_SEATS cross-check disagrees with num_agents"


def test_episode_timeout_and_wall_clock_budgets():
    assert MANIFEST["episode_timeout_minutes"] == 20
    assert defaults.PLATFORM_EPISODE_TIMEOUT_MINUTES == \
        MANIFEST["episode_timeout_minutes"]
    for variant in MANIFEST["variants"]:
        budget = variant["game_config"]["wall_clock_budget_seconds"]
        assert budget <= 645, variant["id"]
        # connect + draft + play must stay inside 60% of the platform budget
        config = variant["game_config"]
        worst = (config["player_connect_timeout_seconds"]
                 + config.get("draft_deadline_ms", 0) / 1000
                 + config["max_ticks"] * config["tick_deadline_ms"] / 1000)
        assert worst <= 0.6 * MANIFEST["episode_timeout_minutes"] * 60, \
            variant["id"]
    cert = MANIFEST["certification"]["game_config"]
    assert (cert["player_connect_timeout_seconds"]
            + cert["draft_deadline_ms"] / 1000
            + cert["max_ticks"] * cert["tick_deadline_ms"] / 1000) <= \
        0.6 * MANIFEST["episode_timeout_minutes"] * 60


def test_both_variants_are_declared():
    ids = [v["id"] for v in MANIFEST["variants"]]
    assert ids == ["draft", "nodraft"]
    by_id = {v["id"]: v for v in MANIFEST["variants"]}
    assert by_id["draft"]["game_config"]["draft_enabled"] is True
    assert by_id["nodraft"]["game_config"]["draft_enabled"] is False
    for variant in MANIFEST["variants"]:
        assert variant["description"], variant["id"]
        assert "tokens" not in variant["game_config"], \
            "the runner injects tokens; a literal list is rejected"
    assert "tokens" not in MANIFEST["certification"]["game_config"]


def test_variant_and_certification_configs_parse():
    """Every game_config the manifest ships must be accepted by the
    server's parser once the runner injects tokens."""
    configs = [(v["id"], v["game_config"]) for v in MANIFEST["variants"]]
    configs.append(("certification", MANIFEST["certification"]["game_config"]))
    for label, game_config in configs:
        data = dict(game_config)
        data["tokens"] = [f"tok-{i}" for i in range(len(data["players"]))]
        cfg = GameConfig.from_dict(data)  # raises ConfigError on drift
        assert cfg.num_seats == len(game_config["players"]), label


# -- the platform upload contract -------------------------------------------

def test_manifest_upload_contract():
    assert MANIFEST["game"]["replay_viewer"] == {
        "bundle": "static-replay-viewer"}
    assert MANIFEST["game"]["runnable"]["type"] == "game"
    assert MANIFEST["game"]["description"]
    assert MANIFEST["game"]["owner"]
    assert "tags" not in MANIFEST["game"], \
        "tags live at the manifest top level only"
    assert len(MANIFEST["tags"]) >= 3
    assert "version" not in MANIFEST
    assert "display_name" not in MANIFEST["game"]
    assert MANIFEST["game"]["name"] == "derks-gym"


def test_protocols_and_docs():
    protocols = MANIFEST["game"]["protocols"]
    assert set(protocols) == {"player", "global"}
    for entry in protocols.values():
        assert set(entry) == {"type", "value"}
        assert entry["value"].startswith("https://")
    docs = MANIFEST["game"]["docs"]
    assert set(docs["readme"]) == {"type", "value"}
    assert [page["id"] for page in docs["pages"]] == \
        ["draft.md", "porting.md"]
    for page in docs["pages"]:
        assert page["title"]
        assert page["content"]["value"].startswith("https://")


def test_config_schema_arrays_are_bounded_and_closed():
    schema = MANIFEST["game"]["config_schema"]
    assert schema["additionalProperties"] is False
    assert "heroes_per_seat" not in schema["properties"]
    for name, prop in schema["properties"].items():
        if prop.get("type") == "array":
            assert prop["minItems"] == defaults.NUM_SEATS, name
            assert prop["maxItems"] == defaults.NUM_SEATS, name
    num_agents = schema["properties"]["num_agents"]
    assert num_agents["minimum"] == num_agents["maximum"] == \
        defaults.NUM_SEATS
    for name in ("draft_enabled", "draft_deadline_ms", "catalog_version"):
        assert name in schema["properties"], name


def test_players_are_one_image_env_switched():
    entries = {p["id"]: p for p in MANIFEST["player"]}
    assert set(entries) == {"baseline", "lane-brawler", "drafter"}
    for entry in entries.values():
        assert entry["run"] == ["python", "-m", "players.derk_player"]
        assert entry["image"] == "{{PLAYER_IMAGE}}"
        assert entry["type"] == "player"
        assert entry["description"]
        assert entry["resources"]["limits"]["cpu"] in ("1", "2")
    assert entries["baseline"]["env"] == {"PLAYER_SCRIPTED": "puffer-forge"}
    assert entries["lane-brawler"]["env"] == {"PLAYER_SCRIPTED": "lane-brawler"}
    assert entries["drafter"]["env"] == {
        "PLAYER_PROMPT": "derk-drafter-v1",
        "USE_BEDROCK": "true"}


def test_every_declared_player_has_a_certification_slot():
    """Hosted certification fails `players_missing` the moment the
    manifest declares a runnable the fixture never seats (raid 0.1.2 ->
    0.1.3, 2026-08-23). The strong baseline keeps the seats that decide
    the fixture's outcome; the other two are seated once each."""
    declared = {entry["id"] for entry in MANIFEST["player"]}
    seated = [entry["player_id"]
              for entry in MANIFEST["certification"]["players"]]
    assert set(seated) == declared, sorted(declared ^ set(seated))
    assert seated.count("baseline") >= len(seated) - 2


def test_llm_runnables_do_not_reference_a_game_secret():
    prompt_entries = [entry for entry in MANIFEST["player"]
                      if "PLAYER_PROMPT" in (entry.get("env") or {})]
    assert prompt_entries
    for entry in prompt_entries:
        assert "ANTHROPIC_API_KEY_URI" not in entry["env"], entry
    for row in POLICIES:
        assert "ANTHROPIC_API_KEY_URI" not in row["env"], row


def test_every_llm_policy_gates_the_bedrock_sidecar():
    """A hosted player pod never receives ANTHROPIC_API_KEY: the platform
    grants it a Bedrock sidecar, and it gates that on USE_BEDROCK in the
    policy env (`resolve_player_bedrock`). Without this the champions
    silently draft with their scripted rule — invisible to
    results.draft_fallbacks, which counts server-side substitutions only
    (cogolf, 2026-08-24; observed on derks-gym 0.1.0's league rounds).

    Every PLAYER_PROMPT entry, in the manifest AND in the release's policy
    set, must carry it; a PLAYER_SCRIPTED entry must not (it makes no
    calls, and a needless sidecar is a needless cost)."""
    from players.derk_player import provider_from_env

    for entry in MANIFEST["player"]:
        env = entry.get("env") or {}
        if "PLAYER_PROMPT" in env:
            assert env.get("USE_BEDROCK") == "true", entry["id"]
            assert provider_from_env(env) == "none", entry["id"]
        else:
            assert "USE_BEDROCK" not in env, entry["id"]
    for row in POLICIES:
        if "PLAYER_PROMPT" in row["env"] or "PLAYER_JEV" in row["env"]:
            assert row["env"].get("USE_BEDROCK") == "true", row["name"]
            assert provider_from_env(row["env"]) == "none", row["name"]
        else:
            assert "USE_BEDROCK" not in row["env"], row["name"]


def test_policies_json_has_prompt_scripted_and_jev_players():
    from players.derk_player import PROMPTS, SCRIPTED_NAMES

    names = [row["name"] for row in POLICIES]
    assert len(names) == len(set(names)) == 5
    prompts = [row for row in POLICIES if "PLAYER_PROMPT" in row["env"]]
    scripted = [row for row in POLICIES if "PLAYER_SCRIPTED" in row["env"]]
    assert len(prompts) == 2, "both champions must be PLAYER_PROMPT policies"
    assert len(scripted) == 2
    for row in prompts:
        assert row["env"]["PLAYER_PROMPT"] in PROMPTS
    for row in scripted:
        assert row["env"]["PLAYER_SCRIPTED"] in SCRIPTED_NAMES
    # champion #2 is owned by daveey-1 (a version uploaded as daveey
    # cannot later be submitted as daveey-1)
    assert prompts[1]["player"] == "ply_bac48eb1-662e-44f8-973d-f3e016dccf5d"
    assert "player" not in prompts[0]
    for row in POLICIES:
        assert row["run"] == "/bin/derks-gym-player"


def test_versions_look_like_semver_where_declared():
    version_re = re.compile(r"^\d+\.\d+\.\d+$")
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    match = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)
    assert match and version_re.match(match.group(1))
