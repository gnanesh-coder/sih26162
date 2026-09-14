"""Guards the plain-text webhook body.

Why this exists: the problem statement asks for SMS delivery, and this project
has no commercial gateway. `SmsChannel` therefore reports NOT_IMPLEMENTED and
always will, because a stub returning SENT would be a fabricated delivery
receipt for a message nobody received.

Text mode gives the system a real phone-delivery path over the channel that
already works -- push services such as ntfy render the POST body as the message
a human reads on a lock screen, and a wall of raw JSON there is not readable.
It does NOT make the SMS channel real, and the tests below pin that distinction
so a demo cannot quietly turn a push notification into a claimed SMS.
"""

from unittest import mock

import pytest

from src.alerting.dispatcher import (
    DispatchStatus,
    SmsChannel,
    WebhookChannel,
    format_alert_text,
)

ALERT = {
    "priority": "P0_EMERGENCY",
    "state": "ACCIDENT_SUSPECTED",
    "latitude": 23.75,
    "longitude": 86.42,
    "facility_name": "coal washery",
    "frp": 42.1,
    "n_30d": 3,
    "z_frp": 4.2,
    "timestamp_utc": "2026-09-14T07:48:00Z",
    "rationale": "No established baseline; strong thermal onset.",
}


def _deliver(body_format):
    ch = WebhookChannel(url="https://ntfy.sh/t", secret="s3cr3t", body_format=body_format)
    with mock.patch("src.alerting.dispatcher.requests.post") as post:
        post.return_value = mock.Mock(status_code=200, text="ok")
        result = ch._deliver(ALERT)
    return result, post.call_args.kwargs


def test_json_remains_the_default():
    ch = WebhookChannel(url="https://example.test/hook")
    assert ch.body_format == "json"


def test_text_mode_sends_the_human_readable_body():
    _, kw = _deliver("text")
    body = kw["data"].decode("utf-8")
    assert body == format_alert_text(ALERT)
    assert "P0_EMERGENCY" in body
    assert kw["headers"]["Content-Type"].startswith("text/plain")


def test_json_mode_still_sends_json():
    import json

    _, kw = _deliver("json")
    parsed = json.loads(kw["data"].decode("utf-8"))
    assert parsed["facility_name"] == "coal washery"
    assert kw["headers"]["Content-Type"] == "application/json"


def test_signature_covers_whichever_body_was_sent():
    """The HMAC must authenticate the bytes actually transmitted."""
    import hashlib
    import hmac as _hmac

    for fmt in ("json", "text"):
        _, kw = _deliver(fmt)
        expected = _hmac.new(b"s3cr3t", kw["data"], hashlib.sha256).hexdigest()
        assert kw["headers"]["X-SIH-Signature"] == expected


def test_push_headers_are_set_for_notification_services():
    _, kw = _deliver("text")
    h = kw["headers"]
    assert h["Title"].startswith("P0_EMERGENCY")
    assert "coal washery" in h["Title"]
    assert h["Priority"] == "5", "a P0 must arrive at the top push priority"


def test_non_p0_uses_a_lower_push_priority():
    ch = WebhookChannel(url="https://ntfy.sh/t", body_format="text")
    with mock.patch("src.alerting.dispatcher.requests.post") as post:
        post.return_value = mock.Mock(status_code=200, text="ok")
        ch._deliver({**ALERT, "priority": "P2_ADVISORY"})
    assert post.call_args.kwargs["headers"]["Priority"] == "3"


def test_title_is_bounded():
    """A push title is truncated by the service; truncate it deliberately."""
    ch = WebhookChannel(url="https://ntfy.sh/t", body_format="text")
    with mock.patch("src.alerting.dispatcher.requests.post") as post:
        post.return_value = mock.Mock(status_code=200, text="ok")
        ch._deliver({**ALERT, "facility_name": "x" * 400})
    assert len(post.call_args.kwargs["headers"]["Title"]) <= 120


def test_text_mode_does_not_make_the_sms_channel_real():
    """The whole point. A push notification is not an SMS."""
    sms = SmsChannel()
    assert sms.is_configured() is False
    assert sms.send(ALERT).status is DispatchStatus.NOT_IMPLEMENTED


def test_target_description_names_the_format():
    """An operator reading the dispatch report should see which body went."""
    ch = WebhookChannel(url="https://ntfy.sh/t", body_format="text")
    assert "text" in ch.target_description()
