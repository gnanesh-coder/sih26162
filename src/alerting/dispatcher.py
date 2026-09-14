"""Outbound alert dispatch: webhook, email, and a deliberately unwired SMS adapter.

The problem statement requires that a confirmed accidental industrial fire reach
responders through "secure SMS gateways, encrypted email, and API webhooks to
NTRO, civil defense, and regional first responders". Until now the system
classified detections and stopped: nothing left the building, while
`src/alerting/__init__.py` described "webhook dispatch modules" that did not
exist.

Three properties drive the design, in this order:

1. **Dispatch never happens by accident.** Sending is outward-facing and cannot
   be recalled. It is OFF unless `ALERT_DISPATCH_ENABLED` is explicitly true, so
   a demo, a test run, or a pipeline replay over a year of archived detections
   cannot page a control room. The default is DRY_RUN: the payload is built,
   logged and returned, and no packet leaves the host.

2. **A channel never silently succeeds.** Every attempt returns an explicit
   status. An unconfigured channel says NOT_CONFIGURED; the SMS adapter says
   NOT_IMPLEMENTED because no provider is wired; a transport failure says FAILED
   and carries the error. There is no code path that returns success without a
   delivery having been attempted.

3. **Suppression is enforced here too.** The whole point of the recurrence state
   machine is that routine flaring does not reach an operator. Dispatch refuses
   SUPPRESSED and NON_ALERT priorities outright, so a caller cannot bypass the
   alert-fatigue mandate by calling the dispatcher directly.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import smtplib
import ssl
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from enum import Enum
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("dispatcher")

DISPATCH_TIMEOUT_S = 10.0


class DispatchStatus(str, Enum):
    """Outcome of a single delivery attempt. Only SENT means a packet left."""

    SENT = "SENT"
    # Some recipients reached, others not. Neither SENT nor FAILED is true of a
    # half-delivered page, and collapsing it into either would misinform the
    # operator about whether responders were notified.
    PARTIAL = "PARTIAL"
    DRY_RUN = "DRY_RUN"                    # built and logged, deliberately not sent
    NOT_CONFIGURED = "NOT_CONFIGURED"      # channel has no credentials/endpoint
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"    # adapter exists, no provider wired
    FAILED = "FAILED"                      # transport or remote error
    REFUSED_PRIORITY = "REFUSED_PRIORITY"  # suppressed class; must not be dispatched
    DUPLICATE = "DUPLICATE"                # already dispatched in this session


@dataclass
class DispatchResult:
    """What happened on one channel for one alert."""

    channel: str
    status: DispatchStatus
    detail: str = ""
    target: str = ""
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def delivered(self) -> bool:
        """True only when a packet actually left the host."""
        return self.status is DispatchStatus.SENT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "status": self.status.value,
            "delivered": self.delivered,
            "detail": self.detail,
            "target": self.target,
            "timestamp_utc": self.timestamp_utc,
        }


def dispatch_enabled() -> bool:
    """Whether real delivery is permitted. False unless explicitly turned on."""
    return os.getenv("ALERT_DISPATCH_ENABLED", "").strip().lower() in {"1", "true", "yes"}


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------


def _mask_number(number: str) -> str:
    """Last four digits only.

    Dispatch reports are written to the audit log, which is retained as
    regulatory evidence. That log should record that a responder was paged, not
    accumulate into a directory of responders' personal numbers.
    """
    digits = "".join(ch for ch in number if ch.isdigit())
    return f"...{digits[-4:]}" if len(digits) >= 4 else "..."


class DispatchChannel(ABC):
    """One delivery route. Must report honestly and never raise at the caller."""

    name: str = "channel"

    @abstractmethod
    def is_configured(self) -> bool:
        """Whether this channel has everything it needs to deliver."""

    @abstractmethod
    def _deliver(self, alert: Dict[str, Any]) -> DispatchResult:
        """Perform the actual delivery. Only called when enabled and configured."""

    def send(self, alert: Dict[str, Any]) -> DispatchResult:
        """Deliver `alert`, or explain precisely why it was not delivered."""
        if not self.is_configured():
            return DispatchResult(
                self.name, DispatchStatus.NOT_CONFIGURED,
                detail=f"{self.name}: no endpoint or credentials in the environment.",
            )
        if not dispatch_enabled():
            return DispatchResult(
                self.name, DispatchStatus.DRY_RUN, target=self.target_description(),
                detail=("Payload built and validated; not sent. Set "
                        "ALERT_DISPATCH_ENABLED=true to deliver."),
            )
        try:
            return self._deliver(alert)
        except Exception as exc:  # noqa: BLE001 - a channel must never take the run down
            logger.warning("%s dispatch failed: %s", self.name, exc)
            return DispatchResult(
                self.name, DispatchStatus.FAILED, target=self.target_description(),
                detail=f"{type(exc).__name__}: {exc}",
            )

    def target_description(self) -> str:
        """Where this channel would send, with secrets redacted."""
        return ""


class WebhookChannel(DispatchChannel):
    """POSTs the alert to a configured endpoint, as JSON or as plain text.

    When ALERT_WEBHOOK_SECRET is set the body is signed with HMAC-SHA256 and the
    digest travels in `X-SIH-Signature`, so a receiver can verify the alert came
    from this system rather than from anyone who learned the URL.

    **Two body formats, because the receiver decides what is useful.** A machine
    endpoint wants JSON. A push service that puts a notification on a phone --
    ntfy, Gotify, and similar -- renders the body as the message a human reads,
    and a wall of raw JSON on a lock screen at 3am is not readable.

    Setting ALERT_WEBHOOK_FORMAT=text sends `format_alert_text()` instead, with
    a `Title` header those services use as the notification headline. That gives
    the system a phone-delivery path over the channel that already exists,
    without wiring a commercial SMS gateway or pretending one is wired: the
    SmsChannel still reports NOT_IMPLEMENTED, because a notification is not an
    SMS and claiming otherwise would be a fabricated delivery receipt.
    """

    name = "webhook"

    def __init__(
        self,
        url: Optional[str] = None,
        secret: Optional[str] = None,
        body_format: Optional[str] = None,
    ) -> None:
        self.url = os.getenv("ALERT_WEBHOOK_URL", "") if url is None else url
        self.secret = os.getenv("ALERT_WEBHOOK_SECRET", "") if secret is None else secret
        fmt = os.getenv("ALERT_WEBHOOK_FORMAT", "json") if body_format is None else body_format
        self.body_format = fmt.strip().lower() or "json"

    def is_configured(self) -> bool:
        return bool(self.url.strip())

    def target_description(self) -> str:
        return f"{self.url} ({self.body_format})"

    def _deliver(self, alert: Dict[str, Any]) -> DispatchResult:
        if self.body_format == "text":
            body = format_alert_text(alert).encode("utf-8")
            headers = {"Content-Type": "text/plain; charset=utf-8"}
            # Consumed by ntfy/Gotify as the notification headline; ignored by
            # any receiver that does not know them.
            priority = str(alert.get("priority", "ALERT"))
            facility = alert.get("facility_name") or "unmapped location"
            headers["Title"] = f"{priority}: {facility}"[:120]
            headers["Priority"] = "5" if priority.startswith("P0") else "3"
            headers["Tags"] = "fire"
        else:
            body = json.dumps(alert, default=str).encode("utf-8")
            headers = {"Content-Type": "application/json"}

        if self.secret:
            headers["X-SIH-Signature"] = hmac.new(
                self.secret.encode("utf-8"), body, hashlib.sha256
            ).hexdigest()

        resp = requests.post(self.url, data=body, headers=headers,
                             timeout=DISPATCH_TIMEOUT_S)
        if 200 <= resp.status_code < 300:
            return DispatchResult(self.name, DispatchStatus.SENT, target=self.url,
                                  detail=f"HTTP {resp.status_code}")
        return DispatchResult(
            self.name, DispatchStatus.FAILED, target=self.url,
            detail=f"HTTP {resp.status_code}: {resp.text[:200]}",
        )


class EmailChannel(DispatchChannel):
    """Sends the alert over SMTP with STARTTLS.

    Credentials come from the environment and are never logged. The connection
    is upgraded to TLS before authentication; an SMTP server that refuses
    STARTTLS produces a FAILED result rather than a silent plaintext login.
    """

    name = "email"

    def __init__(self) -> None:
        self.host = os.getenv("ALERT_SMTP_HOST", "").strip()
        self.port = int(os.getenv("ALERT_SMTP_PORT", "587") or 587)
        self.user = os.getenv("ALERT_SMTP_USER", "").strip()
        self.password = os.getenv("ALERT_SMTP_PASSWORD", "")
        self.sender = os.getenv("ALERT_EMAIL_FROM", "").strip()
        self.recipients = [
            r.strip() for r in os.getenv("ALERT_EMAIL_TO", "").split(",") if r.strip()
        ]

    def is_configured(self) -> bool:
        return bool(self.host and self.sender and self.recipients)

    def target_description(self) -> str:
        return ", ".join(self.recipients)

    def _build_message(self, alert: Dict[str, Any]) -> EmailMessage:
        msg = EmailMessage()
        msg["Subject"] = (
            f"[{alert.get('priority', 'ALERT')}] {alert.get('state', 'THERMAL ANOMALY')} "
            f"at {alert.get('facility_name') or 'unmapped site'}"
        )
        msg["From"] = self.sender
        msg["To"] = ", ".join(self.recipients)
        msg.set_content(format_alert_text(alert))
        return msg

    def _deliver(self, alert: Dict[str, Any]) -> DispatchResult:
        context = ssl.create_default_context()
        with smtplib.SMTP(self.host, self.port, timeout=DISPATCH_TIMEOUT_S) as smtp:
            smtp.starttls(context=context)
            if self.user:
                smtp.login(self.user, self.password)
            smtp.send_message(self._build_message(alert))
        return DispatchResult(self.name, DispatchStatus.SENT,
                              target=self.target_description(),
                              detail=f"{len(self.recipients)} recipient(s)")


class SmsChannel(DispatchChannel):
    """Sends the alert as SMS through a configured gateway.

    Three providers, because the sensible choice depends on what the operator
    has rather than on what this project prefers:

    * ``httpsms``  -- httpsms.com turns an ordinary Android handset into the
      gateway. Free, no commercial account, and the messages leave from a real
      Indian number, which matters: a transactional SMS from an unknown foreign
      shortcode is the one most likely to be ignored at 3am.
    * ``twilio``   -- the commercial default, if an account exists.
    * ``generic``  -- any gateway that accepts an HTTP POST. The body is a
      template with ``{to}``, ``{from}`` and ``{text}`` placeholders, so a
      provider this project has never heard of needs configuration, not code.

    **Unconfigured is NOT_CONFIGURED, not NOT_IMPLEMENTED.** The distinction
    matters and is the reason this class was a deliberate stub until now: the
    previous version could not send at all, and said so. It can now, so the only
    honest remaining failure is "no credentials", which is the operator's to
    fix rather than the code's.

    What has not changed is the rule the stub existed to protect: **a result is
    SENT only when the gateway said so.** Per-recipient outcomes are tracked
    separately, a partial delivery reports PARTIAL with the counts, and a total
    failure reports FAILED. Nothing here returns a delivery receipt for a
    message the provider did not acknowledge.
    """

    name = "sms"

    HTTPSMS_URL = "https://api.httpsms.com/v1/messages/send"
    TWILIO_URL = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"

    def __init__(self) -> None:
        self.provider = os.getenv("ALERT_SMS_PROVIDER", "").strip().lower()
        self.recipients = [
            r.strip() for r in os.getenv("ALERT_SMS_TO", "").split(",") if r.strip()
        ]
        self.sender = os.getenv("ALERT_SMS_FROM", "").strip()

        # httpsms
        self.api_key = os.getenv("ALERT_SMS_API_KEY", "")
        # twilio
        self.twilio_sid = os.getenv("ALERT_TWILIO_ACCOUNT_SID", "").strip()
        self.twilio_token = os.getenv("ALERT_TWILIO_AUTH_TOKEN", "")
        # generic
        self.generic_url = os.getenv("ALERT_SMS_URL", "").strip()
        self.generic_template = os.getenv(
            "ALERT_SMS_BODY_TEMPLATE",
            '{{"to": "{to}", "from": "{from}", "text": "{text}"}}',
        )
        self.generic_auth = os.getenv("ALERT_SMS_AUTH_HEADER", "")

    def is_configured(self) -> bool:
        if not self.recipients:
            return False
        if self.provider == "httpsms":
            return bool(self.api_key and self.sender)
        if self.provider == "twilio":
            return bool(self.twilio_sid and self.twilio_token and self.sender)
        if self.provider == "generic":
            return bool(self.generic_url)
        return False

    def target_description(self) -> str:
        # Numbers are partially masked. A dispatch report is written to the
        # audit log, and an audit trail should record that a responder was
        # paged without becoming a directory of their phone numbers.
        return f"{self.provider}: " + ", ".join(_mask_number(n) for n in self.recipients)

    def _send_one(self, number: str, text: str) -> tuple:
        """Returns (ok, detail) for one recipient."""
        if self.provider == "httpsms":
            resp = requests.post(
                self.HTTPSMS_URL,
                json={"content": text, "from": self.sender, "to": number},
                headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
                timeout=DISPATCH_TIMEOUT_S,
            )
        elif self.provider == "twilio":
            resp = requests.post(
                self.TWILIO_URL.format(sid=self.twilio_sid),
                data={"From": self.sender, "To": number, "Body": text},
                auth=(self.twilio_sid, self.twilio_token),
                timeout=DISPATCH_TIMEOUT_S,
            )
        else:  # generic
            headers = {"Content-Type": "application/json"}
            if self.generic_auth and ":" in self.generic_auth:
                key, _, value = self.generic_auth.partition(":")
                headers[key.strip()] = value.strip()
            body = (
                self.generic_template
                .replace("{to}", number)
                .replace("{from}", self.sender)
                .replace("{text}", text.replace('"', '\\"'))
            )
            resp = requests.post(
                self.generic_url, data=body.encode("utf-8"),
                headers=headers, timeout=DISPATCH_TIMEOUT_S,
            )

        if 200 <= resp.status_code < 300:
            return True, f"HTTP {resp.status_code}"
        # The body can echo the API key on some gateways; only the status and a
        # short prefix are kept, and never the request.
        return False, f"HTTP {resp.status_code}: {resp.text[:120]}"

    def _deliver(self, alert: Dict[str, Any]) -> DispatchResult:
        text = format_alert_sms(alert)
        sent, failed = [], []
        for number in self.recipients:
            try:
                ok, detail = self._send_one(number, text)
            except Exception as exc:  # noqa: BLE001 - one number must not take the rest down
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            (sent if ok else failed).append((_mask_number(number), detail))

        target = self.target_description()
        if failed and sent:
            return DispatchResult(
                self.name, DispatchStatus.PARTIAL, target=target,
                detail=(f"{len(sent)} of {len(self.recipients)} delivered; "
                        f"failed: " + "; ".join(f"{n} {d}" for n, d in failed)),
            )
        if failed:
            return DispatchResult(
                self.name, DispatchStatus.FAILED, target=target,
                detail="; ".join(f"{n} {d}" for n, d in failed),
            )
        return DispatchResult(
            self.name, DispatchStatus.SENT, target=target,
            detail=(f"{len(sent)} recipient(s), {len(text)} chars, 1 segment "
                    f"({'GSM-7' if is_gsm7(text) else 'UCS-2'})"),
        )


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

# Which channels carry which priority, mirroring the alerting table in the
# problem-statement research: an accidental industrial fire goes out on every
# route, an operational anomaly reaches the duty desk, an advisory is logged for
# systems that care. Suppressed and non-alert classes appear here as empty
# tuples because they are the alert-fatigue mandate, not an oversight.
PRIORITY_ROUTING: Dict[str, tuple] = {
    "P0_EMERGENCY": ("sms", "email", "webhook"),
    "P1_ALERT": ("email", "webhook"),
    "P2_ADVISORY": ("webhook",),
    "SUPPRESSED": (),
    "NON_ALERT": (),
}


def format_alert_text(alert: Dict[str, Any]) -> str:
    """Plain-text body for an alert. Readable on a phone at 3am."""
    lines = [
        f"{alert.get('priority', 'ALERT')} -- {alert.get('state', 'THERMAL ANOMALY')}",
        "",
        f"Location   : {alert.get('latitude')}, {alert.get('longitude')}",
        f"Facility   : {alert.get('facility_name') or 'not on any mapped polygon'}",
        f"Detected   : {alert.get('timestamp_utc') or alert.get('acq_date')}",
        f"FRP        : {alert.get('frp')} MW",
        f"Recurrence : n_30d={alert.get('n_30d')}, z={alert.get('z_frp')}",
    ]
    if alert.get("dnbr") is not None:
        lines.append(
            f"Optical    : dNBR {alert.get('dnbr')} ({alert.get('dnbr_severity')})"
        )
    if alert.get("rationale"):
        lines += ["", f"Why: {alert['rationale']}"]
    lines += [
        "",
        "Classification is model-derived from satellite thermal data and has not",
        "been confirmed on the ground. Verify before committing responders.",
    ]
    return "\n".join(lines)


# An SMS segment is 160 GSM-7 characters; anything longer is split and billed
# per segment, and a gateway may truncate rather than split. The long-form body
# is written to be read on a phone at 3am, not to fit in one segment, so SMS
# gets its own format: the facts a responder needs to move, in one segment.
SMS_SEGMENT_CHARS = 160      # GSM-7
SMS_SEGMENT_CHARS_UCS2 = 70  # anything outside GSM-7 forces this

# The GSM 03.38 basic alphabet, plus the characters that live in its extension
# table. A message containing anything outside this set is encoded UCS-2 by the
# gateway, and a single segment then holds 70 characters instead of 160.
#
# This caught a real bug. The truncation marker was a typographic ellipsis
# (U+2026), which is not in GSM-7 -- so a body trimmed to "fit one segment" was
# silently promoted to UCS-2 and billed as three. The marker is now three ASCII
# dots, and the limit follows the encoding the text actually requires.
# Built from code points rather than a literal, so the set cannot be corrupted
# by whatever tooling edits this file next -- which is exactly how it broke the
# first time.
_GSM7 = (
    set("@\u00a3$\u00a5\u00e8\u00e9\u00f9\u00ec\u00f2\u00c7")
    | set("\n\r ")
    | set("\u00d8\u00f8\u00c5\u00e5\u0394_\u03a6\u0393\u039b\u03a9")
    | set("\u03a0\u03a8\u03a3\u0398\u039e\u00c6\u00e6\u00df\u00c9")
    | set("!\"#\u00a4%&'()*+,-./:;<=>?")
    | set("0123456789")
    | set("\u00a1ABCDEFGHIJKLMNOPQRSTUVWXYZ\u00c4\u00d6\u00d1\u00dc\u00a7")
    | set("\u00bfabcdefghijklmnopqrstuvwxyz\u00e4\u00f6\u00f1\u00fc\u00e0")
    # GSM-7 extension table: each of these actually costs two septets, so a
    # message full of them fits fewer than 160. Treated as in-alphabet here,
    # which keeps the limit honest for the alerts this system sends -- they are
    # coordinates, numbers and facility names, not braces and euro signs.
    | set("^{}\\[~]|\u20ac")
)


def is_gsm7(text: str) -> bool:
    """Whether `text` fits the GSM-7 alphabet, and so the 160-char segment."""
    return all(ch in _GSM7 for ch in text)


def sms_segment_limit(text: str) -> int:
    """Characters available in ONE segment for this text's required encoding."""
    return SMS_SEGMENT_CHARS if is_gsm7(text) else SMS_SEGMENT_CHARS_UCS2


def format_alert_sms(alert: Dict[str, Any], limit: Optional[int] = None) -> str:
    """One-segment SMS body, truncated deliberately rather than split.

    Ordering is by what a responder acts on: priority, where, how hot, what it
    is. The coordinate keeps four decimal places -- roughly 11 m, finer than
    the 375 m VIIRS pixel the detection came from, so nothing useful is lost.

    OSM facility names are frequently in Devanagari or another Indic script,
    none of which is GSM-7. Rather than mangling the name, the limit drops to
    the 70 characters a UCS-2 segment actually holds, so the message still
    leaves as one segment and the coordinates -- the part a crew needs -- are
    never the thing that gets cut.
    """
    lat = alert.get("latitude")
    lon = alert.get("longitude")
    where = (
        f"{lat:.4f},{lon:.4f}"
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float))
        else "?"
    )
    facility = (alert.get("facility_name") or "unmapped").strip()

    frp = alert.get("frp")
    frp_txt = f" {frp:.0f}MW" if isinstance(frp, (int, float)) else ""

    head = f"{alert.get('priority', 'ALERT')} {alert.get('state', 'THERMAL')}"
    body = f"{head} {where}{frp_txt} {facility}"

    # Unverified is not a footnote on an SMS: it is what stops a crew being
    # committed on a model output. Kept even when the name is cut.
    tail = " -unverified sat detection"

    if limit is None:
        limit = sms_segment_limit(body + tail)

    room = limit - len(tail)
    if len(body) > room:
        # Three ASCII dots, not U+2026: the typographic ellipsis is outside
        # GSM-7 and would push the whole message into UCS-2.
        body = body[: max(0, room - 3)].rstrip() + "..."
    return (body + tail)[:limit]


class AlertDispatcher:
    """Routes an alert to the channels its priority calls for.

    Deduplicates within the life of the instance: an incident already dispatched
    is not sent again, so a pipeline re-run over the same corpus cannot page the
    same control room twice for the same fire.
    """

    def __init__(self, channels: Optional[List[DispatchChannel]] = None) -> None:
        self.channels: Dict[str, DispatchChannel] = {
            c.name: c for c in (channels or [SmsChannel(), EmailChannel(), WebhookChannel()])
        }
        self._dispatched: set = set()

    def dispatch(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Deliver one alert. Returns a per-channel report, never raises."""
        priority = str(alert.get("priority", "NON_ALERT")).upper()
        routes = PRIORITY_ROUTING.get(priority)

        if routes is None:
            return self._report(alert, priority, [DispatchResult(
                "router", DispatchStatus.REFUSED_PRIORITY,
                detail=f"Unknown priority {priority!r}; refusing to guess a route.",
            )])

        if not routes:
            return self._report(alert, priority, [DispatchResult(
                "router", DispatchStatus.REFUSED_PRIORITY,
                detail=(f"{priority} is suppressed by design. Routine industrial "
                        "operation must not reach an operator."),
            )])

        key = self._dedupe_key(alert)
        if key in self._dispatched:
            return self._report(alert, priority, [DispatchResult(
                "router", DispatchStatus.DUPLICATE,
                detail=f"Already dispatched in this session ({key}).",
            )])
        self._dispatched.add(key)

        results = [self.channels[n].send(alert) for n in routes if n in self.channels]
        return self._report(alert, priority, results)

    @staticmethod
    def _dedupe_key(alert: Dict[str, Any]) -> str:
        return "|".join(str(alert.get(k, "")) for k in
                        ("incident_id", "latitude", "longitude", "acq_date"))

    def _report(self, alert: Dict[str, Any], priority: str,
                results: List[DispatchResult]) -> Dict[str, Any]:
        delivered = [r for r in results if r.delivered]
        report = {
            "incident_id": alert.get("incident_id"),
            "priority": priority,
            "dispatch_enabled": dispatch_enabled(),
            "channels": [r.to_dict() for r in results],
            "delivered_count": len(delivered),
            "any_delivered": bool(delivered),
        }
        logger.info(
            "Dispatch %s: %s -> %s",
            alert.get("incident_id"), priority,
            ", ".join(f"{r.channel}={r.status.value}" for r in results) or "no routes",
        )
        return report
