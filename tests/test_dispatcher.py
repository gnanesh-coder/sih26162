"""Tests for outbound alert dispatch.

Dispatch is the one part of this system that reaches outside the host and cannot
be recalled. The properties worth defending are therefore about restraint rather
than throughput: it must not fire by accident, must not claim a delivery it did
not make, and must not let a caller bypass suppression.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.alerting import dispatcher as dsp
from src.alerting.dispatcher import (
    AlertDispatcher,
    DispatchStatus,
    EmailChannel,
    SmsChannel,
    WebhookChannel,
    format_alert_text,
)


def _alert(**over):
    base = {
        "incident_id": 1,
        "priority": "P0_EMERGENCY",
        "state": "ACCIDENTAL_FIRE",
        "latitude": 22.4,
        "longitude": 70.05,
        "frp": 35.3,
        "acq_date": "2026-03-07",
        "facility_name": "Test Refinery",
        "n_30d": 2,
        "z_frp": 6.1,
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------
# Dispatch must not happen by accident
# --------------------------------------------------------------------------

def test_dispatch_is_off_by_default(monkeypatch):
    """An unset environment must never deliver.

    A pipeline replay over a year of archived detections would otherwise page a
    control room for every historical fire.
    """
    monkeypatch.delenv("ALERT_DISPATCH_ENABLED", raising=False)
    assert dsp.dispatch_enabled() is False


@pytest.mark.parametrize("value", ["", "0", "false", "no", "maybe", "TRUE-ish"])
def test_only_an_explicit_affirmative_enables_dispatch(monkeypatch, value):
    monkeypatch.setenv("ALERT_DISPATCH_ENABLED", value)
    assert dsp.dispatch_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes"])
def test_explicit_affirmatives_enable_dispatch(monkeypatch, value):
    monkeypatch.setenv("ALERT_DISPATCH_ENABLED", value)
    assert dsp.dispatch_enabled() is True


def test_configured_channel_dry_runs_when_disabled(monkeypatch):
    """A configured webhook still must not send while dispatch is off."""
    monkeypatch.delenv("ALERT_DISPATCH_ENABLED", raising=False)
    sent = {"n": 0}
    monkeypatch.setattr(dsp.requests, "post",
                        lambda *a, **k: sent.__setitem__("n", sent["n"] + 1))

    result = WebhookChannel(url="https://example.test/hook").send(_alert())

    assert result.status is DispatchStatus.DRY_RUN
    assert result.delivered is False
    assert sent["n"] == 0


# --------------------------------------------------------------------------
# A channel must never claim a delivery it did not make
# --------------------------------------------------------------------------

def test_sms_never_claims_delivery(monkeypatch):
    """SMS must never report delivery it cannot account for.

    This test predates the provider implementation, when the channel was a
    deliberate stub returning NOT_IMPLEMENTED. The adapter now sends for real,
    so the expected status changed -- but the rule the stub existed to protect
    did not: with no gateway configured the channel reports NOT_CONFIGURED and
    `delivered` stays False. Nothing here may return a delivery receipt for a
    message no provider was asked to carry.
    """
    monkeypatch.setenv("ALERT_DISPATCH_ENABLED", "true")
    for key in ("ALERT_SMS_PROVIDER", "ALERT_SMS_TO", "ALERT_SMS_API_KEY",
                "ALERT_SMS_FROM", "ALERT_SMS_URL"):
        monkeypatch.delenv(key, raising=False)
    result = SmsChannel().send(_alert())

    assert result.status is DispatchStatus.NOT_CONFIGURED
    assert result.delivered is False
    assert result.status is not DispatchStatus.SENT


def test_unconfigured_channel_reports_not_configured(monkeypatch):
    monkeypatch.setenv("ALERT_DISPATCH_ENABLED", "true")
    result = WebhookChannel(url="", secret="").send(_alert())

    assert result.status is DispatchStatus.NOT_CONFIGURED
    assert result.delivered is False


def test_transport_failure_is_reported_not_swallowed(monkeypatch):
    """A network error must surface as FAILED, never as success or a crash."""
    monkeypatch.setenv("ALERT_DISPATCH_ENABLED", "true")

    def boom(*a, **k):
        raise ConnectionError("name resolution failed")

    monkeypatch.setattr(dsp.requests, "post", boom)
    result = WebhookChannel(url="https://example.test/hook").send(_alert())

    assert result.status is DispatchStatus.FAILED
    assert result.delivered is False
    assert "name resolution failed" in result.detail


def test_non_2xx_response_is_a_failure(monkeypatch):
    monkeypatch.setenv("ALERT_DISPATCH_ENABLED", "true")

    class Resp:
        status_code = 500
        text = "internal error"

    monkeypatch.setattr(dsp.requests, "post", lambda *a, **k: Resp())
    result = WebhookChannel(url="https://example.test/hook").send(_alert())

    assert result.status is DispatchStatus.FAILED
    assert "500" in result.detail


def test_successful_webhook_reports_sent_and_signs_the_body(monkeypatch):
    monkeypatch.setenv("ALERT_DISPATCH_ENABLED", "true")
    captured = {}

    class Resp:
        status_code = 202
        text = ""

    def fake_post(url, data=None, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers or {}
        captured["data"] = data
        return Resp()

    monkeypatch.setattr(dsp.requests, "post", fake_post)
    result = WebhookChannel(url="https://example.test/hook", secret="s3cret").send(_alert())

    assert result.status is DispatchStatus.SENT
    assert result.delivered is True
    assert "X-SIH-Signature" in captured["headers"]
    assert len(captured["headers"]["X-SIH-Signature"]) == 64  # HMAC-SHA256 hex


def test_webhook_is_unsigned_when_no_secret_is_set(monkeypatch):
    monkeypatch.setenv("ALERT_DISPATCH_ENABLED", "true")
    captured = {}

    class Resp:
        status_code = 200
        text = ""

    monkeypatch.setattr(dsp.requests, "post",
                        lambda url, data=None, headers=None, timeout=None:
                        (captured.update(headers=headers or {}), Resp())[1])
    WebhookChannel(url="https://example.test/hook", secret="").send(_alert())

    assert "X-SIH-Signature" not in captured["headers"]


# --------------------------------------------------------------------------
# Suppression cannot be bypassed by calling dispatch directly
# --------------------------------------------------------------------------

@pytest.mark.parametrize("priority", ["SUPPRESSED", "NON_ALERT"])
def test_suppressed_priorities_are_refused(monkeypatch, priority):
    """The alert-fatigue mandate is enforced at dispatch, not only upstream."""
    monkeypatch.setenv("ALERT_DISPATCH_ENABLED", "true")
    report = AlertDispatcher().dispatch(_alert(priority=priority))

    assert report["any_delivered"] is False
    assert report["channels"][0]["status"] == DispatchStatus.REFUSED_PRIORITY.value


def test_unknown_priority_is_refused_rather_than_guessed(monkeypatch):
    monkeypatch.setenv("ALERT_DISPATCH_ENABLED", "true")
    report = AlertDispatcher().dispatch(_alert(priority="SOMETHING_NEW"))

    assert report["any_delivered"] is False
    assert report["channels"][0]["status"] == DispatchStatus.REFUSED_PRIORITY.value
    assert "refusing to guess" in report["channels"][0]["detail"]


def test_routing_matches_priority():
    p0 = AlertDispatcher().dispatch(_alert(priority="P0_EMERGENCY"))
    p2 = AlertDispatcher().dispatch(_alert(incident_id=2, priority="P2_ADVISORY"))

    assert {c["channel"] for c in p0["channels"]} == {"sms", "email", "webhook"}
    assert {c["channel"] for c in p2["channels"]} == {"webhook"}


def test_the_same_incident_is_not_dispatched_twice():
    d = AlertDispatcher()
    d.dispatch(_alert())
    second = d.dispatch(_alert())

    assert second["channels"][0]["status"] == DispatchStatus.DUPLICATE.value


# --------------------------------------------------------------------------
# The report and the message body
# --------------------------------------------------------------------------

def test_report_always_states_whether_dispatch_was_enabled(monkeypatch):
    monkeypatch.delenv("ALERT_DISPATCH_ENABLED", raising=False)
    report = AlertDispatcher().dispatch(_alert())

    assert report["dispatch_enabled"] is False
    assert report["any_delivered"] is False


def test_alert_text_carries_the_unconfirmed_caveat():
    """An operator reading this at 3am must know it is model-derived."""
    body = format_alert_text(_alert())

    assert "not" in body.lower() and "confirmed" in body.lower()
    assert "35.3" in body


def test_alert_text_names_an_unmapped_site_rather_than_leaving_it_blank():
    body = format_alert_text(_alert(facility_name=None))
    assert "not on any mapped polygon" in body


def test_email_requires_host_sender_and_recipients(monkeypatch):
    for var in ("ALERT_SMTP_HOST", "ALERT_EMAIL_FROM", "ALERT_EMAIL_TO"):
        monkeypatch.delenv(var, raising=False)
    assert EmailChannel().is_configured() is False

    monkeypatch.setenv("ALERT_SMTP_HOST", "smtp.test")
    monkeypatch.setenv("ALERT_EMAIL_FROM", "ops@test")
    monkeypatch.setenv("ALERT_EMAIL_TO", "duty@test, chief@test")
    channel = EmailChannel()
    assert channel.is_configured() is True
    assert channel.recipients == ["duty@test", "chief@test"]
