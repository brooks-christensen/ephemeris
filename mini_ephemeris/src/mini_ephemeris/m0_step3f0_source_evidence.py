from __future__ import annotations


PYTHON_ACCESSOR_FINDING = {
    "id": "python_accessor_semantics",
    "source": "REBOUND 4.6.0 rebound/simulation.py:885-905, 956-1038, 1218-1229",
    "symbol": "Simulation.megno; Simulation.lyapunov; Simulation.particles; Simulation.orbits; Simulation.energy",
    "finding": "The Python accessors read current C state or particle arrays and do not call the WHFast synchronization routine.",
    "historical_applicability": "Historical scientific samples are nevertheless synchronized because the runner calls integrate for every target and REBOUND synchronizes before integrate returns.",
}
