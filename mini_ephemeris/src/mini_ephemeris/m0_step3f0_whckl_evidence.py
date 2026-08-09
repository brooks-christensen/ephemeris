from __future__ import annotations


WHCKL_SHORTCUT_FINDING = {
    "id": "whckl_shortcut",
    "source": "REBOUND 4.6.0 rebound/simulation.py:600-624",
    "symbol": "Simulation.integrator setter",
    "finding": "The WHCKL shortcut selects WHFast with the lazy kernel and first symplectic corrector order 17.",
    "historical_applicability": "This is a concrete candidate for a future physical-only lane; it cannot be selected in a native variation/MEGNO lane because the nonstandard kernel is rejected.",
}
