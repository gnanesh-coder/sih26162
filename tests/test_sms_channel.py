"""Guards the SMS channel.

This class was a deliberate stub for most of the project's life: it reported
NOT_IMPLEMENTED and refused to send, because a stub returning SENT would be a
fabricated delivery receipt for a message nobody received. It now sends for
real, and the rule the stub protected has to survive that change --
**a result is SENT only when the gateway said so.**

The other thing under test is segment arithmetic. An SMS segment is 160
characters in GSM-7 and only 70 in UCS-2, and a single character outside the
GSM-7 alphabet promotes the whole message. That is not pedantry: the first
version used a typographic ellipsis (U+2026) as its truncation marker, so every
body trimmed to "fit one segment" was silently promoted to UCS-2 and would have
been billed as three.
"""

from unittest import mock

import pytest

import src.alerting.dispatcher as D

ALERT = {
    "priority": "P0_EMERGENCY",
    "state": "ACCIDENT_SUSPECTED",
    "latitude": 23.7512,
    "longitude": 86.4231,
    "facility_name": "Jharia coal washery",
    "frp": 42.1,
}

DEVANAGARI_NAME = "झरिया कोयला खदान"


@pytest.fixture
def httpsms(monkeypatch):
    monkeypatch.setenv("ALERT_DISPATCH_ENABLED", "true")
    monkeypatch.setenv("ALERT_SMS_PROVIDER", "httpsms")
    monkeypatch.setenv("ALERT_SMS_TO", "+919876543210,+919812345678")
    monkeypatch.setenv("ALERT_SMS_FROM", "+919000000000")
    monkeypatch.setenv("ALERT_SMS_API_KEY", "test-key")
    return D.SmsChannel()


# ---------------------------------------------------------------------------
# Only the gateway can say SENT
# ---------------------------------------------------------------------------

def test_all_delivered_reports_sent(httpsms):
    with mock.patch.object(D.requests, "post") as post:
        post.return_value = mock.Mock(status_code=200, text="ok")
        assert httpsms.send(ALERT).status is D.DispatchStatus.SENT


def test_some_delivered_reports_partial_not_sent(httpsms):
    """A half-delivered page is neither SENT nor FAILED.

    Collapsing it into either misinforms the operator about whether responders
    were actually notified.
    """
    with mock.patch.object(D.requests, "post") as post:
        post.side_effect = [
            mock.Mock(status_code=200, text="ok"),
            mock.Mock(status_code=402, text="insufficient credit"),
        ]
        result = httpsms.send(ALERT)
    assert result.status is D.DispatchStatus.PARTIAL
    assert "1 of 2 delivered" in result.detail


def test_none_delivered_reports_failed(httpsms):
    with mock.patch.object(D.requests, "post") as post:
        post.return_value = mock.Mock(status_code=500, text="gateway down")
        assert httpsms.send(ALERT).status is D.DispatchStatus.FAILED


def test_one_exception_does_not_stop_the_other_recipient(httpsms):
    """A crew that can be reached must still be reached."""
    with mock.patch.object(D.requests, "post") as post:
        post.side_effect = [ConnectionError("dns"), mock.Mock(status_code=200, text="ok")]
        assert httpsms.send(ALERT).status is D.DispatchStatus.PARTIAL


def test_unconfigured_is_not_configured_rather_than_not_implemented(monkeypatch):
    """The adapter exists now, so the honest failure is missing credentials."""
    for key in ("ALERT_SMS_PROVIDER", "ALERT_SMS_TO", "ALERT_SMS_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    assert D.SmsChannel().send(ALERT).status is D.DispatchStatus.NOT_CONFIGURED


def test_dispatch_disabled_is_a_dry_run_not_a_send(monkeypatch, httpsms):
    """A replay over a year of archived detections must never page anyone."""
    monkeypatch.setenv("ALERT_DISPATCH_ENABLED", "false")
    with mock.patch.object(D.requests, "post") as post:
        result = D.SmsChannel().send(ALERT)
    assert result.status is D.DispatchStatus.DRY_RUN
    post.assert_not_called()


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

def test_httpsms_sends_the_api_key_as_a_header(httpsms):
    with mock.patch.object(D.requests, "post") as post:
        post.return_value = mock.Mock(status_code=200, text="ok")
        httpsms.send(ALERT)
    assert post.call_args.args[0] == D.SmsChannel.HTTPSMS_URL
    assert post.call_args.kwargs["headers"]["x-api-key"] == "test-key"


def test_twilio_uses_basic_auth_and_the_account_url(monkeypatch):
    monkeypatch.setenv("ALERT_DISPATCH_ENABLED", "true")
    monkeypatch.setenv("ALERT_SMS_PROVIDER", "twilio")
    monkeypatch.setenv("ALERT_SMS_TO", "+919876543210")
    monkeypatch.setenv("ALERT_SMS_FROM", "+15550001111")
    monkeypatch.setenv("ALERT_TWILIO_ACCOUNT_SID", "ACfake")
    monkeypatch.setenv("ALERT_TWILIO_AUTH_TOKEN", "tok")
    with mock.patch.object(D.requests, "post") as post:
        post.return_value = mock.Mock(status_code=201, text="ok")
        D.SmsChannel().send(ALERT)
    assert "ACfake" in post.call_args.args[0]
    assert post.call_args.kwargs["auth"] == ("ACfake", "tok")


def test_generic_provider_fills_the_template(monkeypatch):
    """A gateway this project has never heard of needs config, not code."""
    monkeypatch.setenv("ALERT_DISPATCH_ENABLED", "true")
    monkeypatch.setenv("ALERT_SMS_PROVIDER", "generic")
    monkeypatch.setenv("ALERT_SMS_TO", "+919876543210")
    monkeypatch.setenv("ALERT_SMS_URL", "https://gateway.test/send")
    monkeypatch.setenv("ALERT_SMS_AUTH_HEADER", "authorization: tok")
    monkeypatch.setenv("ALERT_SMS_BODY_TEMPLATE", '{"numbers":"{to}","message":"{text}"}')
    with mock.patch.object(D.requests, "post") as post:
        post.return_value = mock.Mock(status_code=200, text="ok")
        D.SmsChannel().send(ALERT)
    body = post.call_args.kwargs["data"].decode("utf-8")
    assert "+919876543210" in body
    assert post.call_args.kwargs["headers"]["authorization"] == "tok"


# ---------------------------------------------------------------------------
# Segment arithmetic
# ---------------------------------------------------------------------------

def test_truncation_marker_stays_inside_gsm7():
    """The regression: U+2026 promoted the whole message to UCS-2."""
    body = D.format_alert_sms({**ALERT, "facility_name": "A" * 300})
    assert "…" not in body
    assert D.is_gsm7(body)


def test_ascii_alert_fits_one_gsm7_segment():
    body = D.format_alert_sms({**ALERT, "facility_name": "B" * 300})
    assert len(body) <= D.SMS_SEGMENT_CHARS


def test_non_gsm7_name_is_budgeted_at_the_ucs2_limit():
    """OSM facility names are frequently in an Indic script."""
    body = D.format_alert_sms({**ALERT, "facility_name": DEVANAGARI_NAME * 4})
    assert len(body) <= D.SMS_SEGMENT_CHARS_UCS2


def test_coordinates_survive_truncation():
    """A crew can find a fire from coordinates; the name is a convenience."""
    body = D.format_alert_sms({**ALERT, "facility_name": "C" * 500})
    assert "23.7512,86.4231" in body


def test_unverified_warning_survives_truncation():
    """What stops a crew being committed on a model output is not a footnote."""
    body = D.format_alert_sms({**ALERT, "facility_name": "D" * 500})
    assert body.endswith("-unverified sat detection")


def test_gsm7_detection_and_limits():
    assert D.is_gsm7("P0 23.7512,86.4231 42MW")
    assert not D.is_gsm7(DEVANAGARI_NAME)
    assert D.sms_segment_limit("plain ascii") == D.SMS_SEGMENT_CHARS
    assert D.sms_segment_limit(DEVANAGARI_NAME) == D.SMS_SEGMENT_CHARS_UCS2


# ---------------------------------------------------------------------------
# The audit trail must not become a phone directory
# ---------------------------------------------------------------------------

def test_recipient_numbers_are_masked_in_reports(httpsms):
    with mock.patch.object(D.requests, "post") as post:
        post.return_value = mock.Mock(status_code=200, text="ok")
        result = httpsms.send(ALERT)
    combined = (result.target or "") + (result.detail or "")
    assert "+919876543210" not in combined
    assert "...3210" in result.target


def test_masking_handles_a_short_number():
    assert D._mask_number("12") == "..."
    assert D._mask_number("+91 98765 43210") == "...3210"
