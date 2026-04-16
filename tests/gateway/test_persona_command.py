"""Tests for the /persona [reload] gateway command.

Verifies that /persona invalidates all cached agent system prompts,
notifies other active sessions, and returns a correct summary string.
"""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source(platform=Platform.WHATSAPP, chat_id="chat1", user_id="u1") -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str = "/persona", source: SessionSource = None) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=source or _make_source(),
        message_id="m1",
    )


def _make_runner(session_entries=None):
    """Create a minimal GatewayRunner for testing /persona."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.WHATSAPP: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.WHATSAPP: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._background_tasks = set()
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None

    # Agent cache + lock
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()

    # Session store mock
    store = MagicMock()
    store._entries = session_entries if session_entries is not None else {}
    runner.session_store = store

    return runner


def _make_agent(has_invalidate=True):
    """Make a mock agent, optionally with _invalidate_system_prompt."""
    agent = MagicMock()
    if has_invalidate:
        agent._invalidate_system_prompt = MagicMock()
    else:
        # Remove the attribute entirely
        if hasattr(agent, "_invalidate_system_prompt"):
            del agent._invalidate_system_prompt
        agent._spec_class = None
    return agent


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPersonaCommand:
    """Tests for _handle_persona_command."""

    @pytest.mark.asyncio
    async def test_persona_invalidates_cached_agents(self, tmp_path):
        """'/persona' calls _invalidate_system_prompt on every cached agent."""
        runner = _make_runner()

        agent1 = _make_agent()
        agent2 = _make_agent()
        # Cache as tuples (agent, ...) mimicking real cache structure
        runner._agent_cache = {
            "whatsapp:chat1": (agent1,),
            "whatsapp:chat2": (agent2,),
        }

        event = _make_event("/persona")

        with patch("gateway.run._hermes_home", tmp_path):
            result = await runner._handle_persona_command(event)

        agent1._invalidate_system_prompt.assert_called_once()
        agent2._invalidate_system_prompt.assert_called_once()
        assert "Invalidated 2 cached session(s)" in result

    @pytest.mark.asyncio
    async def test_persona_reload_subcommand_same_as_bare(self, tmp_path):
        """/persona reload behaves exactly the same as /persona."""
        runner = _make_runner()
        agent = _make_agent()
        runner._agent_cache = {"whatsapp:chat1": (agent,)}

        event = _make_event("/persona reload")

        with patch("gateway.run._hermes_home", tmp_path):
            result = await runner._handle_persona_command(event)

        agent._invalidate_system_prompt.assert_called_once()
        assert "Invalidated 1 cached session(s)" in result

    @pytest.mark.asyncio
    async def test_persona_notifies_other_sessions_only(self, tmp_path):
        """Only sessions OTHER than the current one receive a notification."""
        current_source = _make_source(chat_id="chat1")
        other_entry = MagicMock()

        # Simulate session_store._entries with two sessions
        # Session key format: "platform:chat_id"
        entries = {
            "whatsapp:chat1": MagicMock(),   # current session
            "whatsapp:chat2": other_entry,   # another session
        }
        runner = _make_runner(session_entries=entries)

        # _session_key_for_source must return the key that matches the current session
        runner.session_store._generate_session_key = MagicMock(
            return_value="whatsapp:chat1"
        )

        event = _make_event("/persona", source=current_source)

        with patch("gateway.run._hermes_home", tmp_path):
            result = await runner._handle_persona_command(event)

        # Adapter.send should have been called once (for chat2, not chat1)
        adapter = runner.adapters[Platform.WHATSAPP]
        assert adapter.send.call_count == 1
        call_args = adapter.send.call_args
        assert call_args[0][0] == "chat2"  # chat_id
        assert "Persona reloaded" in call_args[0][1]
        assert "notified 1 other chat(s)" in result

    @pytest.mark.asyncio
    async def test_persona_empty_cache(self, tmp_path):
        """Works correctly when no agents are in the cache yet."""
        runner = _make_runner()
        # _agent_cache is empty — no agents loaded
        event = _make_event("/persona")

        with patch("gateway.run._hermes_home", tmp_path):
            result = await runner._handle_persona_command(event)

        assert "Invalidated 0 cached session(s)" in result

    @pytest.mark.asyncio
    async def test_persona_missing_invalidate_method_is_safe(self, tmp_path):
        """Agents without _invalidate_system_prompt are silently skipped."""
        runner = _make_runner()

        # Agent stored directly (not as tuple), missing the invalidate method
        agent_no_method = object()  # plain object, no _invalidate_system_prompt
        runner._agent_cache = {"whatsapp:chat1": agent_no_method}

        event = _make_event("/persona")

        with patch("gateway.run._hermes_home", tmp_path):
            result = await runner._handle_persona_command(event)

        # Should not raise; invalidated count stays 0
        assert "Invalidated 0 cached session(s)" in result

    @pytest.mark.asyncio
    async def test_persona_returns_soul_found_status(self, tmp_path):
        """Returns 'SOUL.md found' when SOUL.md exists."""
        soul_file = tmp_path / "SOUL.md"
        soul_file.write_text("# My Agent\nYou are Hermes.")

        runner = _make_runner()
        event = _make_event("/persona")

        with patch("gateway.run._hermes_home", tmp_path):
            result = await runner._handle_persona_command(event)

        assert "SOUL.md found" in result

    @pytest.mark.asyncio
    async def test_persona_returns_soul_missing_warning(self, tmp_path):
        """Returns a warning when SOUL.md does not exist."""
        runner = _make_runner()
        event = _make_event("/persona")

        with patch("gateway.run._hermes_home", tmp_path):
            result = await runner._handle_persona_command(event)

        assert "SOUL.md not found" in result

    @pytest.mark.asyncio
    async def test_persona_counts_correct_with_multiple_agents(self, tmp_path):
        """Summary counts match number of agents invalidated."""
        runner = _make_runner()

        agents = [_make_agent() for _ in range(3)]
        runner._agent_cache = {
            f"whatsapp:chat{i}": (a,) for i, a in enumerate(agents)
        }

        event = _make_event("/persona")

        with patch("gateway.run._hermes_home", tmp_path):
            result = await runner._handle_persona_command(event)

        for a in agents:
            a._invalidate_system_prompt.assert_called_once()
        assert "Invalidated 3 cached session(s)" in result

    @pytest.mark.asyncio
    async def test_persona_current_session_not_notified_via_send(self, tmp_path):
        """The current chat is NOT notified via adapter.send (its response is the return value)."""
        current_source = _make_source(chat_id="chat1")
        entries = {
            "whatsapp:chat1": MagicMock(),
        }
        runner = _make_runner(session_entries=entries)
        runner.session_store._generate_session_key = MagicMock(
            return_value="whatsapp:chat1"
        )

        event = _make_event("/persona", source=current_source)

        with patch("gateway.run._hermes_home", tmp_path):
            result = await runner._handle_persona_command(event)

        adapter = runner.adapters[Platform.WHATSAPP]
        adapter.send.assert_not_called()
        assert "notified 0 other chat(s)" in result

    @pytest.mark.asyncio
    async def test_persona_no_cache_lock_skips_gracefully(self, tmp_path):
        """When _agent_cache_lock is missing, command still completes without error."""
        runner = _make_runner()
        del runner._agent_cache_lock  # simulate missing lock

        event = _make_event("/persona")

        with patch("gateway.run._hermes_home", tmp_path):
            result = await runner._handle_persona_command(event)

        assert "Invalidated 0 cached session(s)" in result
