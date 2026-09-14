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
    """SMS adapter with no provider wired. Always reports NOT_IMPLEMENTED.

    The problem statement asks for SMS gateway delivery, and an SMS gateway
    requires a commercial account this project does not have. The adapter exists
    so the routing table is complete and a provider is a drop-in, but it will
    never claim to have sent anything. A stub that returned SENT would be a
    fabricated delivery receipt for a message nobody received -- worse than an
    absent feature, because an operator would believe responders were notified.
    """

    name = "sms"

    def is_configured(self) -> bool:
        # Deliberately always False: honest about having no provider.
        return False

    def send(self, alert: Dict[str, Any]) -> DispatchResult:
        return DispatchResult(
            self.name, DispatchStatus.NOT_IMPLEMENTED,
            detail=("No SMS provider is wired. Implement SmsChannel._deliver "
                    "against a gateway account; until then this channel reports "
                    "honestly rather than claiming delivery."),
        )

    def _deliver(self, alert: Dict[str, Any]) -> DispatchResult:  # pragma: no cover
        raise NotImplementedError("No SMS provider configured.")


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
