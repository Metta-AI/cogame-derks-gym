"""async_main startup behavior: config errors are clean exit-2 paths
(no tracebacks), replay mode serves.

Covers review item C16: the ConfigError path used to be dead code
because async_main parsed the config bytes directly.
"""

import asyncio
import json
import socket

import aiohttp
import numpy as np
import pytest

from cogame_derks_gym import defaults
from cogame_derks_gym.replay import ReplayWriter
from cogame_derks_gym.server import async_main

from tests.test_server import make_config


def _clear_cogame_env(monkeypatch):
    for name in ("COGAME_CONFIG_URI", "COGAME_LOAD_REPLAY_URI",
                 "COGAME_RESULTS_URI", "COGAME_SAVE_REPLAY_URI",
                 "COGAME_PLAYER_FAILURE_URI", "COGAME_HOST", "COGAME_PORT"):
        monkeypatch.delenv(name, raising=False)


async def test_missing_config_uri_exits_2(monkeypatch, capsys):
    _clear_cogame_env(monkeypatch)
    assert await async_main() == 2
    assert "COGAME_CONFIG_URI is required" in capsys.readouterr().err


async def test_unreadable_config_uri_exits_2(monkeypatch, capsys):
    _clear_cogame_env(monkeypatch)
    monkeypatch.setenv("COGAME_CONFIG_URI", "file:///no/such/config.json")
    assert await async_main() == 2
    err = capsys.readouterr().err
    assert "invalid config" in err
    assert "Traceback" not in err


async def test_malformed_config_json_exits_2(monkeypatch, capsys, tmp_path):
    _clear_cogame_env(monkeypatch)
    bad = tmp_path / "config.json"
    bad.write_text("{not json")
    monkeypatch.setenv("COGAME_CONFIG_URI", f"file://{bad}")
    assert await async_main() == 2
    err = capsys.readouterr().err
    assert "not valid JSON" in err
    assert "Traceback" not in err


async def test_invalid_config_shape_exits_2(monkeypatch, capsys, tmp_path):
    _clear_cogame_env(monkeypatch)
    bad = tmp_path / "config.json"
    bad.write_text(json.dumps({"players": []}))
    monkeypatch.setenv("COGAME_CONFIG_URI", f"file://{bad}")
    assert await async_main() == 2
    assert "players" in capsys.readouterr().err


async def test_malformed_http_config_json_exits_2(monkeypatch, capsys):
    """The non-local path also surfaces bad JSON as ConfigError/exit 2."""
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    async def handle(request):
        return web.Response(text="{not json", content_type="application/json")

    app = web.Application()
    app.router.add_get("/config", handle)
    server = TestServer(app)
    await server.start_server()
    try:
        _clear_cogame_env(monkeypatch)
        monkeypatch.setenv("COGAME_CONFIG_URI",
                           str(server.make_url("/config")))
        assert await async_main() == 2
        assert "not valid JSON" in capsys.readouterr().err
    finally:
        await server.close()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_replay_mode_entry_serves(monkeypatch, tmp_path):
    """COGAME_LOAD_REPLAY_URI enters replay mode: /healthz and
    /replay-data serve until the process is stopped."""
    cfg = make_config()
    writer = ReplayWriter(cfg, "aa" * 32)
    rng = np.random.default_rng(1)
    for t in range(4):
        writer.append_tick(t, rng.integers(
            0, defaults.ACT_HIGH, size=(10, 6)).astype(np.uint8))
    data = writer.finalize({"winner": None, "end_reason": "tick_cap",
                            "final_tick": 4})
    replay_path = tmp_path / "replay.bin"
    replay_path.write_bytes(data)

    port = _free_port()
    _clear_cogame_env(monkeypatch)
    monkeypatch.setenv("COGAME_LOAD_REPLAY_URI", f"file://{replay_path}")
    monkeypatch.setenv("COGAME_HOST", "127.0.0.1")
    monkeypatch.setenv("COGAME_PORT", str(port))

    task = asyncio.create_task(async_main())
    try:
        async with aiohttp.ClientSession() as session:
            for _ in range(100):
                if task.done():  # startup failure: surface it
                    raise AssertionError(f"async_main exited: {task.result()}")
                try:
                    async with session.get(
                            f"http://127.0.0.1:{port}/healthz") as resp:
                        assert resp.status == 200
                        break
                except aiohttp.ClientConnectorError:
                    await asyncio.sleep(0.05)
            else:
                pytest.fail("replay-mode server never came up")
            async with session.get(
                    f"http://127.0.0.1:{port}/replay-data") as resp:
                assert resp.status == 200
                assert await resp.read() == data
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
