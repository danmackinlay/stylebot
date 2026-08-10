"""TLS trust fix-up for the tinker SDK's Rust HTTP transport.

Both the trainer and the inference path talk to tinker, and both must apply the
patch before their first SDK call; it lives here so neither has to import the
other. Nothing is imported at module scope — the patch itself is best-effort and
does its own `tinker` import inside the try.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def patch_pyqwest_tls_trust() -> None:
    """Make the tinker SDK's Rust transport trust the system cert store.

    tinker (0.23.x) routes HTTP through pyqwest (reqwest/rustls) constructed
    with `tls_include_system_certs=False`, whose bundled roots reject the
    Google Trust Services chain tinker.thinkingmachines.dev serves
    (`invalid peer certificate: UnknownIssuer` on the very first auth-token
    call). Rebuilding the transport with system certs included fixes it; there
    is no supported env/config knob, so this patches the SDK's private transport
    factory. Best-effort: if the internals move, log and let the SDK try its own
    default.
    """
    try:
        import pyqwest
        import tinker._base_client as bc
        from pyqwest.httpx import AsyncPyqwestTransport

        def _with_system_certs():
            return AsyncPyqwestTransport(
                transport=pyqwest.HTTPTransport(tls_include_system_certs=True)
            )

        bc._default_pyqwest_transport = _with_system_certs
    except Exception:  # pragma: no cover - only reachable on SDK refactors
        logger.warning(
            "could not patch pyqwest TLS trust; if the run fails with "
            "'invalid peer certificate: UnknownIssuer', the tinker SDK "
            "internals have moved — see patch_pyqwest_tls_trust"
        )
