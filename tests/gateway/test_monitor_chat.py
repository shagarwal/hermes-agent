"""Tests for the monitor_chat feature (gateway/run.py helpers)."""
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Import helpers under test
# ---------------------------------------------------------------------------

from gateway.run import _load_monitor_chat_target, _source_label


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_source(chat_id: str):
    source = MagicMock()
    source.chat_id = chat_id
    return source


def _make_adapters(platform_value: str, adapter=None):
    """Return a dict keyed by a Platform-like object."""
    platform = MagicMock()
    platform.value = platform_value
    # Make it hashable and comparable
    platform.__hash__ = lambda self: hash(platform_value)
    platform.__eq__ = lambda self, other: getattr(other, "value", None) == platform_value
    return {platform: adapter or MagicMock()}


# ---------------------------------------------------------------------------
# _source_label tests
# ---------------------------------------------------------------------------

class TestSourceLabel:
    def test_dm_lid(self):
        source = _make_source("13181372117007@lid")
        assert _source_label(source) == "DM"

    def test_dm_whatsapp_net(self):
        source = _make_source("15551234567@s.whatsapp.net")
        assert _source_label(source) == "DM"

    def test_group(self):
        source = _make_source("12345678abcdef@g.us")
        label = _source_label(source)
        assert label.startswith("group:")
        # Should use first 8 chars of chat_id
        assert label == "group:12345678"

    def test_empty_chat_id(self):
        source = _make_source("")
        assert _source_label(source) == "DM"

    def test_none_chat_id(self):
        source = MagicMock()
        source.chat_id = None
        assert _source_label(source) == "DM"


# ---------------------------------------------------------------------------
# _load_monitor_chat_target tests
# ---------------------------------------------------------------------------

class TestLoadMonitorChatTarget:
    def test_empty_config_returns_none_none(self):
        with patch("gateway.run._load_gateway_config", return_value={}):
            adapter, chat_id = _load_monitor_chat_target({})
        assert adapter is None
        assert chat_id is None

    def test_missing_monitor_chat_key_returns_none_none(self):
        cfg = {"agent": {"max_turns": 90}}
        with patch("gateway.run._load_gateway_config", return_value=cfg):
            adapter, chat_id = _load_monitor_chat_target({})
        assert adapter is None
        assert chat_id is None

    def test_empty_monitor_chat_string_returns_none_none(self):
        cfg = {"agent": {"monitor_chat": ""}}
        with patch("gateway.run._load_gateway_config", return_value=cfg):
            adapter, chat_id = _load_monitor_chat_target({})
        assert adapter is None
        assert chat_id is None

    def test_invalid_format_no_colon_returns_none_none(self):
        cfg = {"agent": {"monitor_chat": "whatsapp13181372117007"}}
        with patch("gateway.run._load_gateway_config", return_value=cfg):
            adapter, chat_id = _load_monitor_chat_target({})
        assert adapter is None
        assert chat_id is None

    def test_platform_adapter_not_found_returns_none_none(self):
        cfg = {"agent": {"monitor_chat": "whatsapp:13181372117007@lid"}}
        # Empty adapters dict — platform not present
        with patch("gateway.run._load_gateway_config", return_value=cfg):
            adapter, chat_id = _load_monitor_chat_target({})
        assert adapter is None
        assert chat_id is None

    def test_valid_config_returns_adapter_and_chat_id(self):
        from gateway.config import Platform
        mock_adapter = MagicMock()
        cfg = {"agent": {"monitor_chat": "whatsapp:13181372117007@lid"}}
        adapters = {Platform.WHATSAPP: mock_adapter}
        with patch("gateway.run._load_gateway_config", return_value=cfg):
            adapter, chat_id = _load_monitor_chat_target(adapters)
        assert adapter is mock_adapter
        assert chat_id == "13181372117007@lid"

    def test_exception_in_config_load_returns_none_none(self):
        with patch("gateway.run._load_gateway_config", side_effect=RuntimeError("disk error")):
            adapter, chat_id = _load_monitor_chat_target({})
        assert adapter is None
        assert chat_id is None


# ---------------------------------------------------------------------------
# Monitor redirect logic (unit-level simulation)
# ---------------------------------------------------------------------------

class TestMonitorRedirectLogic:
    """Simulate the logic that runs at the top of _run_agent after loading monitor config."""

    def _compute_monitor_state(self, source_chat_id, monitor_chat_id, monitor_adapter):
        """Replicate the in-function logic for testing."""
        _monitor_adapter = monitor_adapter
        _monitor_chat_id = monitor_chat_id
        _is_monitor_source = _monitor_chat_id and source_chat_id == _monitor_chat_id
        _monitor_active = bool(_monitor_adapter and _monitor_chat_id and not _is_monitor_source)
        return _monitor_active, _is_monitor_source

    def test_source_is_monitor_chat_no_redirect(self):
        """When source == monitor_chat, monitor should NOT be active."""
        monitor_adapter = MagicMock()
        monitor_chat_id = "13181372117007@lid"
        source_chat_id = "13181372117007@lid"

        _monitor_active, _is_monitor_source = self._compute_monitor_state(
            source_chat_id, monitor_chat_id, monitor_adapter
        )

        assert _is_monitor_source is True
        assert _monitor_active is False

    def test_source_different_from_monitor_redirect_active(self):
        """When source != monitor_chat, monitor SHOULD be active."""
        monitor_adapter = MagicMock()
        monitor_chat_id = "13181372117007@lid"
        source_chat_id = "99999999@s.whatsapp.net"

        _monitor_active, _is_monitor_source = self._compute_monitor_state(
            source_chat_id, monitor_chat_id, monitor_adapter
        )

        assert _is_monitor_source is False
        assert _monitor_active is True

    def test_no_monitor_config_not_active(self):
        """When no monitor is configured, monitor should NOT be active."""
        monitor_adapter = None
        monitor_chat_id = None
        source_chat_id = "99999999@s.whatsapp.net"

        _monitor_active, _is_monitor_source = self._compute_monitor_state(
            source_chat_id, monitor_chat_id, monitor_adapter
        )

        # _is_monitor_source uses short-circuit: None (falsy) when monitor_chat_id is None
        assert not _is_monitor_source
        assert _monitor_active is False

    def test_status_adapter_redirected_when_monitor_active(self):
        """_status_adapter should point to monitor adapter when active."""
        source_adapter = MagicMock(name="source_adapter")
        monitor_adapter = MagicMock(name="monitor_adapter")
        monitor_chat_id = "13181372117007@lid"
        source_chat_id = "99999999@s.whatsapp.net"

        _monitor_active, _ = self._compute_monitor_state(
            source_chat_id, monitor_chat_id, monitor_adapter
        )

        # Simulate the redirect logic
        _status_adapter = source_adapter
        _status_chat_id = source_chat_id
        if _monitor_active:
            _status_adapter = monitor_adapter
            _status_chat_id = monitor_chat_id

        assert _status_adapter is monitor_adapter
        assert _status_chat_id == monitor_chat_id

    def test_status_adapter_not_redirected_when_source_is_monitor(self):
        """_status_adapter should stay as source adapter when source == monitor chat."""
        source_adapter = MagicMock(name="source_adapter")
        monitor_adapter = MagicMock(name="monitor_adapter")
        monitor_chat_id = "13181372117007@lid"
        source_chat_id = "13181372117007@lid"  # Same as monitor

        _monitor_active, _ = self._compute_monitor_state(
            source_chat_id, monitor_chat_id, monitor_adapter
        )

        _status_adapter = source_adapter
        _status_chat_id = source_chat_id
        if _monitor_active:
            _status_adapter = monitor_adapter
            _status_chat_id = monitor_chat_id

        assert _status_adapter is source_adapter
        assert _status_chat_id == source_chat_id


# ---------------------------------------------------------------------------
# Approval message format tests
# ---------------------------------------------------------------------------

class TestApprovalMessageFormat:
    def test_approval_message_includes_session_key_when_monitor_active(self):
        """When monitor is active, approval instructions include the session key."""
        _monitor_active = True
        _approval_session_key = "whatsapp:99999999@s.whatsapp.net"
        _src_label = "DM"

        cmd_preview = "rm -rf /tmp/test"
        desc = "Dangerous deletion"

        if _monitor_active:
            _approve_instructions = (
                f"Reply `/approve {_approval_session_key}` to execute or "
                f"`/deny {_approval_session_key}` to cancel."
            )
        else:
            _approve_instructions = (
                "Reply `/approve` to execute, `/approve session` to approve this pattern "
                "for the session, `/approve always` to approve permanently, or `/deny` to cancel."
            )

        _monitor_prefix = f"[{_src_label}] " if _monitor_active else ""
        msg = (
            f"{_monitor_prefix}⚠️ **Dangerous command requires approval:**\n"
            f"```\n{cmd_preview}\n```\n"
            f"Reason: {desc}\n\n"
            f"{_approve_instructions}"
        )

        assert _approval_session_key in msg
        assert "/approve whatsapp:99999999@s.whatsapp.net" in msg
        assert "/deny whatsapp:99999999@s.whatsapp.net" in msg
        assert "[DM]" in msg

    def test_approval_message_normal_when_monitor_not_active(self):
        """When monitor is NOT active, approval instructions use original format."""
        _monitor_active = False
        _approval_session_key = "whatsapp:99999999@s.whatsapp.net"
        _src_label = "DM"

        cmd_preview = "rm -rf /tmp/test"
        desc = "Dangerous deletion"

        if _monitor_active:
            _approve_instructions = (
                f"Reply `/approve {_approval_session_key}` to execute or "
                f"`/deny {_approval_session_key}` to cancel."
            )
        else:
            _approve_instructions = (
                "Reply `/approve` to execute, `/approve session` to approve this pattern "
                "for the session, `/approve always` to approve permanently, or `/deny` to cancel."
            )

        _monitor_prefix = f"[{_src_label}] " if _monitor_active else ""
        msg = (
            f"{_monitor_prefix}⚠️ **Dangerous command requires approval:**\n"
            f"```\n{cmd_preview}\n```\n"
            f"Reason: {desc}\n\n"
            f"{_approve_instructions}"
        )

        assert "/approve` to execute" in msg
        assert "/approve session`" in msg
        assert "[DM]" not in msg
        assert _approval_session_key not in msg
