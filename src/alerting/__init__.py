"""Recurrence state machine, SitRep generation, and outbound alert dispatch.

`dispatcher` delivers over webhook and SMTP email. The SMS adapter is present
but has no provider wired and reports NOT_IMPLEMENTED; it never claims delivery.
Dispatch is off unless ALERT_DISPATCH_ENABLED is explicitly set.
"""
