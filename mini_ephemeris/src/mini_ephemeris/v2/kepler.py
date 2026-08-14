"""Isolated bound-elliptic canonical Kepler flow and analytic tangent map.

The primitive advances one relative Jacobi pair ``(q,P)``.  Velocity is only
the internal value ``P/reduced_mass_kg``.  There is no center-of-mass drift,
multi-pair composition, force callback, observer, or integration loop here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import re
import sys
from typing import Any, Iterable, Mapping, Sequence, Tuple

from .canonical import canonical_json_bytes, finite_binary64_hex, sha256_hex
from .errors import InvalidModel, InvalidState, LayoutMismatch, V2FoundationError
from .jacobi import JacobiTransformPlan
from .model import BodyId, PhysicalModel


Vector3 = Tuple[float, float, float]

PLAN_SCHEMA = "v2.canonical_kepler_pair_plan/1"
STATE_SCHEMA = "v2.canonical_kepler_pair_state/1"
TANGENT_SCHEMA = "v2.canonical_kepler_pair_tangent/1"
DIAGNOSTIC_SCHEMA = "v2.kepler_solver_diagnostics/1"
MAXIMUM_QUALIFIED_ECCENTRICITY = 0.92
MAXIMUM_QUALIFIED_PHASE_RAD = 0.999 * 2.0 * math.pi
UNIT_ROUNDOFF = sys.float_info.epsilon
_SMALLEST_NORMAL = sys.float_info.min
_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_INV_FACTORIAL = tuple(1.0 / math.factorial(index) for index in range(35))


class KeplerDomainError(V2FoundationError):
    """Raised when a pair, state, or duration is outside the qualified domain."""


class KeplerConvergenceError(V2FoundationError):
    """Raised when the deterministic universal-anomaly solver does not converge."""


def _finite_positive(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InvalidModel(f"{field} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise InvalidModel(f"{field} must be finite and positive")
    return result


def _bounded_int(value: int, field: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidModel(f"{field} must be an integer")
    if value < 1 or value > maximum:
        raise InvalidModel(f"{field} must be in [1,{maximum}]")
    return value


def _fingerprint(value: str, field: str) -> str:
    if not isinstance(value, str) or not _FINGERPRINT_PATTERN.fullmatch(value):
        raise InvalidState(f"{field} must be a lowercase SHA-256 fingerprint")
    return value


def _vector3(values: Iterable[float], field: str) -> Vector3:
    components = tuple(values)
    if len(components) != 3:
        raise InvalidState(f"{field} must have exactly three components")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in components):
        raise InvalidState(f"{field} components must be real numbers")
    result = tuple(float(value) for value in components)
    if not all(math.isfinite(value) for value in result):
        raise InvalidState(f"{field} must contain only finite binary64 values")
    return result  # type: ignore[return-value]


def _vector_payload(values: Vector3, field: str) -> list[str]:
    return [finite_binary64_hex(value, field) for value in values]


def _dot(first: Vector3, second: Vector3) -> float:
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def _cross(first: Vector3, second: Vector3) -> Vector3:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _norm(values: Vector3) -> float:
    return math.sqrt(_dot(values, values))


@dataclass(frozen=True, init=False)
class CanonicalKeplerPairPlan:
    """Immutable model-bound parameters for one relative Jacobi Kepler pair."""

    pair_index: int
    outer_body_id: BodyId
    inner_body_ids: Tuple[BodyId, ...]
    layout_fingerprint: str
    model_fingerprint: str
    jacobi_plan_fingerprint: str
    inner_mass_kg: float
    outer_mass_kg: float
    reduced_mass_kg: float
    gravitational_parameter_m3_s2: float
    minimum_periapsis_m: float
    maximum_eccentricity: float
    maximum_absolute_phase_rad: float
    newton_max_iterations: int
    quartic_max_iterations: int
    bisection_max_iterations: int
    unit_system_id: str
    schema: str
    fingerprint: str

    def __init__(
        self,
        *,
        model: PhysicalModel,
        jacobi_plan: JacobiTransformPlan,
        pair_index: int,
        minimum_periapsis_m: float,
        maximum_eccentricity: float = MAXIMUM_QUALIFIED_ECCENTRICITY,
        maximum_absolute_phase_rad: float = MAXIMUM_QUALIFIED_PHASE_RAD,
        newton_max_iterations: int = 32,
        quartic_max_iterations: int = 64,
        bisection_max_iterations: int = 96,
    ) -> None:
        if not isinstance(model, PhysicalModel):
            raise InvalidModel("Kepler pair plan requires a PhysicalModel")
        if not isinstance(jacobi_plan, JacobiTransformPlan):
            raise InvalidModel("Kepler pair plan requires a JacobiTransformPlan")
        if jacobi_plan.model_fingerprint != model.fingerprint:
            raise LayoutMismatch("Jacobi plan and model fingerprints differ")
        if jacobi_plan.layout != model.layout:
            raise LayoutMismatch("Jacobi plan and model layouts differ")
        if model.units.identifier != "si_v1":
            raise InvalidModel("Kepler pair plan requires SI units")
        if isinstance(pair_index, bool) or not isinstance(pair_index, int):
            raise InvalidModel("pair_index must be an integer")
        if pair_index < 1 or pair_index >= len(jacobi_plan.body_ids):
            raise InvalidModel("pair_index must identify one relative Jacobi pair")

        minimum_periapsis = _finite_positive(minimum_periapsis_m, "minimum_periapsis_m")
        maximum_eccentricity_value = _finite_positive(
            maximum_eccentricity, "maximum_eccentricity"
        )
        if maximum_eccentricity_value > MAXIMUM_QUALIFIED_ECCENTRICITY:
            raise InvalidModel("maximum_eccentricity exceeds the qualified envelope")
        maximum_phase = _finite_positive(
            maximum_absolute_phase_rad, "maximum_absolute_phase_rad"
        )
        if maximum_phase > MAXIMUM_QUALIFIED_PHASE_RAD:
            raise InvalidModel("maximum phase exceeds the qualified envelope")

        inner_mass = jacobi_plan.cumulative_masses_kg[pair_index - 1]
        outer_mass = jacobi_plan.masses_kg[pair_index]
        total_mass = inner_mass + outer_mass
        reduced_mass = inner_mass * outer_mass / total_mass
        gravitational_parameter = model.gravitational_constant_si * total_mass
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (inner_mass, outer_mass, reduced_mass, gravitational_parameter)
        ):
            raise InvalidModel("derived Kepler pair parameters must be finite and positive")

        object.__setattr__(self, "pair_index", pair_index)
        object.__setattr__(self, "outer_body_id", jacobi_plan.body_ids[pair_index])
        object.__setattr__(self, "inner_body_ids", jacobi_plan.body_ids[:pair_index])
        object.__setattr__(self, "layout_fingerprint", jacobi_plan.layout_fingerprint)
        object.__setattr__(self, "model_fingerprint", model.fingerprint)
        object.__setattr__(self, "jacobi_plan_fingerprint", jacobi_plan.fingerprint)
        object.__setattr__(self, "inner_mass_kg", inner_mass)
        object.__setattr__(self, "outer_mass_kg", outer_mass)
        object.__setattr__(self, "reduced_mass_kg", reduced_mass)
        object.__setattr__(
            self, "gravitational_parameter_m3_s2", gravitational_parameter
        )
        object.__setattr__(self, "minimum_periapsis_m", minimum_periapsis)
        object.__setattr__(self, "maximum_eccentricity", maximum_eccentricity_value)
        object.__setattr__(self, "maximum_absolute_phase_rad", maximum_phase)
        object.__setattr__(
            self,
            "newton_max_iterations",
            _bounded_int(newton_max_iterations, "newton_max_iterations", 32),
        )
        object.__setattr__(
            self,
            "quartic_max_iterations",
            _bounded_int(quartic_max_iterations, "quartic_max_iterations", 64),
        )
        object.__setattr__(
            self,
            "bisection_max_iterations",
            _bounded_int(bisection_max_iterations, "bisection_max_iterations", 96),
        )
        object.__setattr__(self, "unit_system_id", "si_v1")
        object.__setattr__(self, "schema", PLAN_SCHEMA)
        object.__setattr__(self, "fingerprint", sha256_hex(self.canonical_bytes()))

    def canonical_payload(self) -> Mapping[str, Any]:
        """Return all material pair, domain, and solver fields deterministically."""

        return {
            "bisection_max_iterations": self.bisection_max_iterations,
            "gravitational_parameter_m3_s2_hex": finite_binary64_hex(
                self.gravitational_parameter_m3_s2,
                "gravitational_parameter_m3_s2",
            ),
            "inner_body_ids": [body.value for body in self.inner_body_ids],
            "inner_mass_kg_hex": finite_binary64_hex(self.inner_mass_kg, "inner_mass_kg"),
            "jacobi_plan_fingerprint": self.jacobi_plan_fingerprint,
            "layout_fingerprint": self.layout_fingerprint,
            "maximum_absolute_phase_rad_hex": finite_binary64_hex(
                self.maximum_absolute_phase_rad, "maximum_absolute_phase_rad"
            ),
            "maximum_eccentricity_hex": finite_binary64_hex(
                self.maximum_eccentricity, "maximum_eccentricity"
            ),
            "minimum_periapsis_m_hex": finite_binary64_hex(
                self.minimum_periapsis_m, "minimum_periapsis_m"
            ),
            "model_fingerprint": self.model_fingerprint,
            "newton_max_iterations": self.newton_max_iterations,
            "outer_body_id": self.outer_body_id.value,
            "outer_mass_kg_hex": finite_binary64_hex(self.outer_mass_kg, "outer_mass_kg"),
            "pair_index": self.pair_index,
            "quartic_max_iterations": self.quartic_max_iterations,
            "reduced_mass_kg_hex": finite_binary64_hex(
                self.reduced_mass_kg, "reduced_mass_kg"
            ),
            "schema": self.schema,
            "unit_system_id": self.unit_system_id,
        }

    def canonical_bytes(self) -> bytes:
        """Serialize this plan without its derived fingerprint."""

        return canonical_json_bytes(self.canonical_payload())


def build_kepler_pair_plan(
    model: PhysicalModel,
    jacobi_plan: JacobiTransformPlan,
    pair_index: int,
    *,
    minimum_periapsis_m: float,
    maximum_eccentricity: float = MAXIMUM_QUALIFIED_ECCENTRICITY,
    maximum_absolute_phase_rad: float = MAXIMUM_QUALIFIED_PHASE_RAD,
    newton_max_iterations: int = 32,
    quartic_max_iterations: int = 64,
    bisection_max_iterations: int = 96,
) -> CanonicalKeplerPairPlan:
    """Build one immutable plan from a qualified fixed-mass Jacobi plan."""

    return CanonicalKeplerPairPlan(
        model=model,
        jacobi_plan=jacobi_plan,
        pair_index=pair_index,
        minimum_periapsis_m=minimum_periapsis_m,
        maximum_eccentricity=maximum_eccentricity,
        maximum_absolute_phase_rad=maximum_absolute_phase_rad,
        newton_max_iterations=newton_max_iterations,
        quartic_max_iterations=quartic_max_iterations,
        bisection_max_iterations=bisection_max_iterations,
    )


@dataclass(frozen=True, init=False)
class CanonicalKeplerPairState:
    """Immutable plan-bound canonical relative state ``(q,P)`` in SI units."""

    q_m: Vector3
    p_kg_m_per_s: Vector3
    unit_system_id: str
    layout_fingerprint: str
    model_fingerprint: str
    pair_plan_fingerprint: str

    def __init__(
        self,
        *,
        q_m: Iterable[float],
        p_kg_m_per_s: Iterable[float],
        unit_system_id: str,
        layout_fingerprint: str,
        model_fingerprint: str,
        pair_plan_fingerprint: str,
    ) -> None:
        if unit_system_id != "si_v1":
            raise InvalidState("Kepler pair state requires unit_system_id='si_v1'")
        object.__setattr__(self, "q_m", _vector3(q_m, "q_m"))
        object.__setattr__(self, "p_kg_m_per_s", _vector3(p_kg_m_per_s, "p_kg_m_per_s"))
        object.__setattr__(self, "unit_system_id", unit_system_id)
        object.__setattr__(
            self, "layout_fingerprint", _fingerprint(layout_fingerprint, "layout_fingerprint")
        )
        object.__setattr__(
            self, "model_fingerprint", _fingerprint(model_fingerprint, "model_fingerprint")
        )
        object.__setattr__(
            self,
            "pair_plan_fingerprint",
            _fingerprint(pair_plan_fingerprint, "pair_plan_fingerprint"),
        )

    def canonical_payload(self) -> Mapping[str, Any]:
        """Return exact deterministic plan-bound state fields."""

        return {
            "layout_fingerprint": self.layout_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "p_kg_m_per_s_hex": _vector_payload(self.p_kg_m_per_s, "p_kg_m_per_s"),
            "pair_plan_fingerprint": self.pair_plan_fingerprint,
            "q_m_hex": _vector_payload(self.q_m, "q_m"),
            "schema": STATE_SCHEMA,
            "unit_system_id": self.unit_system_id,
        }

    def canonical_bytes(self) -> bytes:
        """Serialize the immutable state deterministically."""

        return canonical_json_bytes(self.canonical_payload())


@dataclass(frozen=True, init=False)
class CanonicalKeplerPairTangent:
    """Immutable plan-bound canonical tangent ``(delta_q,delta_P)``."""

    delta_q_m: Vector3
    delta_p_kg_m_per_s: Vector3
    unit_system_id: str
    layout_fingerprint: str
    model_fingerprint: str
    pair_plan_fingerprint: str

    def __init__(
        self,
        *,
        delta_q_m: Iterable[float],
        delta_p_kg_m_per_s: Iterable[float],
        unit_system_id: str,
        layout_fingerprint: str,
        model_fingerprint: str,
        pair_plan_fingerprint: str,
    ) -> None:
        if unit_system_id != "si_v1":
            raise InvalidState("Kepler pair tangent requires unit_system_id='si_v1'")
        object.__setattr__(self, "delta_q_m", _vector3(delta_q_m, "delta_q_m"))
        object.__setattr__(
            self,
            "delta_p_kg_m_per_s",
            _vector3(delta_p_kg_m_per_s, "delta_p_kg_m_per_s"),
        )
        object.__setattr__(self, "unit_system_id", unit_system_id)
        object.__setattr__(
            self, "layout_fingerprint", _fingerprint(layout_fingerprint, "layout_fingerprint")
        )
        object.__setattr__(
            self, "model_fingerprint", _fingerprint(model_fingerprint, "model_fingerprint")
        )
        object.__setattr__(
            self,
            "pair_plan_fingerprint",
            _fingerprint(pair_plan_fingerprint, "pair_plan_fingerprint"),
        )

    def canonical_payload(self) -> Mapping[str, Any]:
        """Return exact deterministic plan-bound tangent fields."""

        return {
            "delta_p_kg_m_per_s_hex": _vector_payload(
                self.delta_p_kg_m_per_s, "delta_p_kg_m_per_s"
            ),
            "delta_q_m_hex": _vector_payload(self.delta_q_m, "delta_q_m"),
            "layout_fingerprint": self.layout_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "pair_plan_fingerprint": self.pair_plan_fingerprint,
            "schema": TANGENT_SCHEMA,
            "unit_system_id": self.unit_system_id,
        }

    def canonical_bytes(self) -> bytes:
        """Serialize the immutable tangent deterministically."""

        return canonical_json_bytes(self.canonical_payload())


@dataclass(frozen=True)
class KeplerSolverDiagnostics:
    """Deterministic diagnostics for one accepted universal-anomaly solve."""

    branch: str
    iterations: int
    residual_s: float
    update_x_s_per_m: float
    converged: bool
    universal_anomaly_x_s_per_m: float
    beta_m2_per_s2: float
    phase_advance_rad: float
    schema: str = DIAGNOSTIC_SCHEMA


@dataclass(frozen=True)
class KeplerDriftResult:
    """Detached physical result and the diagnostics for its exact root."""

    state: CanonicalKeplerPairState
    diagnostics: KeplerSolverDiagnostics


@dataclass(frozen=True)
class KeplerTangentResult:
    """Physical and directional tangent results sharing one solver root."""

    state: CanonicalKeplerPairState
    tangent: CanonicalKeplerPairTangent
    diagnostics: KeplerSolverDiagnostics


@dataclass(frozen=True)
class _DomainValues:
    velocity_m_per_s: Vector3
    radius_m: float
    specific_energy_m2_per_s2: float
    semimajor_axis_m: float
    eccentricity: float
    phase_advance_rad: float


@dataclass(frozen=True)
class _RootSolution:
    x_s_per_m: float
    gs3: Tuple[float, float, float, float]
    radius_m: float
    diagnostics: KeplerSolverDiagnostics


@dataclass(frozen=True)
class _PhysicalEvaluation:
    state: CanonicalKeplerPairState
    f_hat: float
    g_s: float
    fdot_per_s: float
    gdot_hat: float
    velocity0_m_per_s: Vector3


def _require_plan_state(
    plan: CanonicalKeplerPairPlan, state: CanonicalKeplerPairState
) -> None:
    if not isinstance(plan, CanonicalKeplerPairPlan):
        raise InvalidModel("plan must be CanonicalKeplerPairPlan")
    if not isinstance(state, CanonicalKeplerPairState):
        raise InvalidState("Kepler drift requires CanonicalKeplerPairState")
    if state.unit_system_id != plan.unit_system_id:
        raise LayoutMismatch("state and plan unit systems differ")
    if state.layout_fingerprint != plan.layout_fingerprint:
        raise LayoutMismatch("state and plan layout fingerprints differ")
    if state.model_fingerprint != plan.model_fingerprint:
        raise LayoutMismatch("state and plan model fingerprints differ")
    if state.pair_plan_fingerprint != plan.fingerprint:
        raise LayoutMismatch("state and Kepler pair plan fingerprints differ")


def _require_plan_tangent(
    plan: CanonicalKeplerPairPlan, tangent: CanonicalKeplerPairTangent
) -> None:
    if not isinstance(tangent, CanonicalKeplerPairTangent):
        raise InvalidState("Kepler tangent action requires CanonicalKeplerPairTangent")
    if tangent.unit_system_id != plan.unit_system_id:
        raise LayoutMismatch("tangent and plan unit systems differ")
    if tangent.layout_fingerprint != plan.layout_fingerprint:
        raise LayoutMismatch("tangent and plan layout fingerprints differ")
    if tangent.model_fingerprint != plan.model_fingerprint:
        raise LayoutMismatch("tangent and plan model fingerprints differ")
    if tangent.pair_plan_fingerprint != plan.fingerprint:
        raise LayoutMismatch("tangent and Kepler pair plan fingerprints differ")


def _duration(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise KeplerDomainError("duration_s must be a finite binary64 number")
    result = float(value)
    if not math.isfinite(result):
        raise KeplerDomainError("duration_s must be finite")
    return result


def _domain_values(
    plan: CanonicalKeplerPairPlan,
    state: CanonicalKeplerPairState,
    duration_s: float,
) -> _DomainValues:
    _require_plan_state(plan, state)
    duration = _duration(duration_s)
    radius = _norm(state.q_m)
    if not math.isfinite(radius) or radius <= 0.0:
        raise KeplerDomainError("radius must be finite and nonzero")
    reduced_mass = plan.reduced_mass_kg
    gravitational_parameter = plan.gravitational_parameter_m3_s2
    velocity = tuple(value / reduced_mass for value in state.p_kg_m_per_s)
    speed_squared = _dot(velocity, velocity)  # type: ignore[arg-type]
    specific_energy = 0.5 * speed_squared - gravitational_parameter / radius
    if not math.isfinite(specific_energy) or specific_energy >= 0.0:
        raise KeplerDomainError("state must be a bound elliptic conic")
    semimajor_axis = -gravitational_parameter / (2.0 * specific_energy)
    angular_momentum = _cross(state.q_m, velocity)  # type: ignore[arg-type]
    angular_momentum_norm = _norm(angular_momentum)
    angular_scale = math.sqrt(gravitational_parameter * semimajor_axis)
    if angular_momentum_norm <= 128.0 * UNIT_ROUNDOFF * angular_scale:
        raise KeplerDomainError("radial or degenerate elliptic states are unsupported")
    eccentricity_squared = (
        1.0
        + 2.0
        * specific_energy
        * angular_momentum_norm
        * angular_momentum_norm
        / (gravitational_parameter * gravitational_parameter)
    )
    eccentricity = math.sqrt(max(0.0, eccentricity_squared))
    if not math.isfinite(eccentricity) or eccentricity > plan.maximum_eccentricity:
        raise KeplerDomainError("eccentricity exceeds the qualified bound")
    periapsis = semimajor_axis * (1.0 - eccentricity)
    if not math.isfinite(periapsis) or periapsis < plan.minimum_periapsis_m:
        raise KeplerDomainError("periapsis is inside the qualified collision separation")
    mean_motion = math.sqrt(
        gravitational_parameter / (semimajor_axis * semimajor_axis * semimajor_axis)
    )
    phase = abs(duration) * mean_motion
    if not math.isfinite(phase) or phase > plan.maximum_absolute_phase_rad:
        raise KeplerDomainError("duration exceeds the qualified phase-advance bound")
    return _DomainValues(
        velocity_m_per_s=velocity,  # type: ignore[arg-type]
        radius_m=radius,
        specific_energy_m2_per_s2=specific_energy,
        semimajor_axis_m=semimajor_axis,
        eccentricity=eccentricity,
        phase_advance_rad=phase,
    )


def _stumpff_series(z_value: float, index: int) -> float:
    maximum_factorial = len(_INV_FACTORIAL) - 1
    maximum_term = (maximum_factorial - index) // 2
    value = _INV_FACTORIAL[index + 2 * maximum_term]
    for term in range(maximum_term - 1, -1, -1):
        value = _INV_FACTORIAL[index + 2 * term] - z_value * value
    return value


def _stumpff_cs3(z_original: float) -> Tuple[float, float, float, float]:
    z = z_original
    scale_count = 0
    while abs(z) > 0.1:
        z = z / 4.0
        scale_count += 1
    c0 = _stumpff_series(z, 0)
    c1 = _stumpff_series(z, 1)
    c2 = _stumpff_series(z, 2)
    c3 = _stumpff_series(z, 3)
    for _ in range(scale_count):
        c3 = (c2 + c0 * c3) * 0.25
        c2 = c1 * c1 * 0.5
        c1 = c0 * c1
        c0 = 2.0 * c0 * c0 - 1.0
    return c0, c1, c2, c3


def _stumpff_cs5(z_original: float) -> Tuple[float, float, float, float, float, float]:
    z = z_original
    scale_count = 0
    while abs(z) > 0.1:
        z = z / 4.0
        scale_count += 1
    c1 = _stumpff_series(z, 1)
    c2 = _stumpff_series(z, 2)
    c3 = _stumpff_series(z, 3)
    c4 = _stumpff_series(z, 4)
    c5 = _stumpff_series(z, 5)
    for _ in range(scale_count):
        z = z * 4.0
        c5 = (c5 + c4 + c3 * c2) * 0.0625
        c4 = (1.0 + c1) * c3 * 0.125
        c3 = _INV_FACTORIAL[3] - z * c5
        c2 = 0.5 - z * c4
        c1 = 1.0 - z * c3
    c0 = _INV_FACTORIAL[0] - z * c2
    return c0, c1, c2, c3, c4, c5


def _stiefel_gs3(beta: float, x_value: float) -> Tuple[float, float, float, float]:
    x_squared = x_value * x_value
    c0, c1, c2, c3 = _stumpff_cs3(beta * x_squared)
    return c0, c1 * x_value, c2 * x_squared, c3 * x_squared * x_value


def _stiefel_gs5(
    beta: float, x_value: float
) -> Tuple[float, float, float, float, float, float]:
    x_squared = x_value * x_value
    c0, c1, c2, c3, c4, c5 = _stumpff_cs5(beta * x_squared)
    x_cubed = x_squared * x_value
    x_fourth = x_cubed * x_value
    x_fifth = x_fourth * x_value
    return c0, c1 * x_value, c2 * x_squared, c3 * x_cubed, c4 * x_fourth, c5 * x_fifth


def _equation(
    radius0: float,
    eta0: float,
    zeta0: float,
    duration_s: float,
    x_value: float,
    gs3: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    residual = radius0 * x_value + eta0 * gs3[2] + zeta0 * gs3[3] - duration_s
    derivative = radius0 + eta0 * gs3[1] + zeta0 * gs3[2]
    return residual, derivative


def _converged(
    *,
    residual: float,
    update: float,
    duration_s: float,
    radius0: float,
    eta0: float,
    zeta0: float,
    x_value: float,
    gs3: Tuple[float, float, float, float],
) -> bool:
    residual_scale = max(
        abs(duration_s),
        abs(radius0 * x_value),
        abs(eta0 * gs3[2]),
        abs(zeta0 * gs3[3]),
        _SMALLEST_NORMAL,
    )
    update_scale = max(abs(x_value), abs(duration_s / radius0), _SMALLEST_NORMAL)
    return (
        abs(residual) <= 128.0 * UNIT_ROUNDOFF * residual_scale
        and abs(update) <= 64.0 * UNIT_ROUNDOFF * update_scale
    )


def _diagnostics(
    branch: str,
    iterations: int,
    residual: float,
    update: float,
    x_value: float,
    beta: float,
    phase: float,
) -> KeplerSolverDiagnostics:
    return KeplerSolverDiagnostics(
        branch=branch,
        iterations=iterations,
        residual_s=abs(residual),
        update_x_s_per_m=abs(update),
        converged=True,
        universal_anomaly_x_s_per_m=x_value,
        beta_m2_per_s2=beta,
        phase_advance_rad=phase,
    )


def _solve_universal_anomaly(
    plan: CanonicalKeplerPairPlan,
    state: CanonicalKeplerPairState,
    duration_s: float,
    domain: _DomainValues,
) -> _RootSolution:
    if duration_s == 0.0:
        diagnostics = _diagnostics(
            "zero_duration", 0, 0.0, 0.0, 0.0, 0.0, domain.phase_advance_rad
        )
        return _RootSolution(0.0, (1.0, 0.0, 0.0, 0.0), domain.radius_m, diagnostics)

    gravitational_parameter = plan.gravitational_parameter_m3_s2
    radius0 = domain.radius_m
    velocity0 = domain.velocity_m_per_s
    speed_squared = _dot(velocity0, velocity0)
    beta = 2.0 * gravitational_parameter / radius0 - speed_squared
    if not math.isfinite(beta) or beta <= 0.0:
        raise KeplerDomainError("universal elliptic solver requires beta>0")
    eta0 = _dot(state.q_m, velocity0)
    zeta0 = gravitational_parameter - beta * radius0
    sqrt_beta = math.sqrt(beta)
    inverse_period = sqrt_beta * beta / (2.0 * math.pi * gravitational_parameter)
    x_per_period = 2.0 * math.pi / sqrt_beta

    duration_over_radius = duration_s / radius0
    x_value = duration_over_radius * (
        1.0 - duration_over_radius * eta0 * 0.5 / radius0
    )
    gs3 = _stiefel_gs3(beta, x_value)
    residual, derivative = _equation(
        radius0, eta0, zeta0, duration_s, x_value, gs3
    )
    if not math.isfinite(derivative) or derivative <= 0.0:
        derivative = math.nan
        x_after_first = math.nan
    else:
        eta_g1_zeta_g2 = eta0 * gs3[1] + zeta0 * gs3[2]
        x_after_first = (
            x_value * eta_g1_zeta_g2
            - eta0 * gs3[2]
            - zeta0 * gs3[3]
            + duration_s
        ) / derivative
    first_update = x_after_first - x_value
    iterations = 1
    if math.isfinite(x_after_first):
        x_value = x_after_first
        gs3 = _stiefel_gs3(beta, x_value)
        residual, derivative = _equation(
            radius0, eta0, zeta0, duration_s, x_value, gs3
        )
        if _converged(
            residual=residual,
            update=first_update,
            duration_s=duration_s,
            radius0=radius0,
            eta0=eta0,
            zeta0=zeta0,
            x_value=x_value,
            gs3=gs3,
        ):
            return _RootSolution(
                x_value,
                gs3,
                derivative,
                _diagnostics(
                    "elliptic_newton",
                    iterations,
                    residual,
                    first_update,
                    x_value,
                    beta,
                    domain.phase_advance_rad,
                ),
            )

    use_quartic = not math.isfinite(first_update) or abs(first_update) > 0.01 * x_per_period
    primary_branch = "elliptic_quartic" if use_quartic else "elliptic_newton"
    if use_quartic:
        x_value = beta * duration_s / gravitational_parameter
        for _ in range(plan.quartic_max_iterations):
            gs3 = _stiefel_gs3(beta, x_value)
            residual, derivative = _equation(
                radius0, eta0, zeta0, duration_s, x_value, gs3
            )
            second_derivative = eta0 * gs3[0] + zeta0 * gs3[1]
            radicand = abs(
                16.0 * derivative * derivative
                - 20.0 * residual * second_derivative
            )
            denominator = derivative + math.sqrt(radicand)
            if denominator == 0.0 or not math.isfinite(denominator):
                break
            new_x = (x_value * denominator - 5.0 * residual) / denominator
            update = new_x - x_value
            x_value = new_x
            iterations += 1
            gs3 = _stiefel_gs3(beta, x_value)
            residual, derivative = _equation(
                radius0, eta0, zeta0, duration_s, x_value, gs3
            )
            if _converged(
                residual=residual,
                update=update,
                duration_s=duration_s,
                radius0=radius0,
                eta0=eta0,
                zeta0=zeta0,
                x_value=x_value,
                gs3=gs3,
            ):
                return _RootSolution(
                    x_value,
                    gs3,
                    derivative,
                    _diagnostics(
                        primary_branch,
                        iterations,
                        residual,
                        update,
                        x_value,
                        beta,
                        domain.phase_advance_rad,
                    ),
                )
    else:
        remaining = max(0, plan.newton_max_iterations - iterations)
        for _ in range(remaining):
            gs3 = _stiefel_gs3(beta, x_value)
            residual, derivative = _equation(
                radius0, eta0, zeta0, duration_s, x_value, gs3
            )
            if derivative <= 0.0 or not math.isfinite(derivative):
                break
            eta_g1_zeta_g2 = eta0 * gs3[1] + zeta0 * gs3[2]
            new_x = (
                x_value * eta_g1_zeta_g2
                - eta0 * gs3[2]
                - zeta0 * gs3[3]
                + duration_s
            ) / derivative
            update = new_x - x_value
            x_value = new_x
            iterations += 1
            gs3 = _stiefel_gs3(beta, x_value)
            residual, derivative = _equation(
                radius0, eta0, zeta0, duration_s, x_value, gs3
            )
            if _converged(
                residual=residual,
                update=update,
                duration_s=duration_s,
                radius0=radius0,
                eta0=eta0,
                zeta0=zeta0,
                x_value=x_value,
                gs3=gs3,
            ):
                return _RootSolution(
                    x_value,
                    gs3,
                    derivative,
                    _diagnostics(
                        primary_branch,
                        iterations,
                        residual,
                        update,
                        x_value,
                        beta,
                        domain.phase_advance_rad,
                    ),
                )

    lower = x_per_period * math.floor(duration_s * inverse_period)
    upper = lower + x_per_period
    last_residual = math.inf
    last_update = math.inf
    for _ in range(plan.bisection_max_iterations):
        x_value = (lower + upper) * 0.5
        gs3 = _stiefel_gs3(beta, x_value)
        residual, derivative = _equation(
            radius0, eta0, zeta0, duration_s, x_value, gs3
        )
        update = (upper - lower) * 0.5
        iterations += 1
        last_residual = residual
        last_update = update
        if _converged(
            residual=residual,
            update=update,
            duration_s=duration_s,
            radius0=radius0,
            eta0=eta0,
            zeta0=zeta0,
            x_value=x_value,
            gs3=gs3,
        ):
            return _RootSolution(
                x_value,
                gs3,
                derivative,
                _diagnostics(
                    "elliptic_bisection",
                    iterations,
                    residual,
                    update,
                    x_value,
                    beta,
                    domain.phase_advance_rad,
                ),
            )
        if residual >= 0.0:
            upper = x_value
        else:
            lower = x_value
    raise KeplerConvergenceError(
        "universal anomaly did not satisfy residual and update gates "
        f"after {iterations} iterations; residual_s={last_residual!r}, "
        f"update_x_s_per_m={last_update!r}"
    )


def _new_state(
    plan: CanonicalKeplerPairPlan, q_m: Vector3, p_kg_m_per_s: Vector3
) -> CanonicalKeplerPairState:
    return CanonicalKeplerPairState(
        q_m=q_m,
        p_kg_m_per_s=p_kg_m_per_s,
        unit_system_id=plan.unit_system_id,
        layout_fingerprint=plan.layout_fingerprint,
        model_fingerprint=plan.model_fingerprint,
        pair_plan_fingerprint=plan.fingerprint,
    )


def _new_tangent(
    plan: CanonicalKeplerPairPlan,
    delta_q_m: Vector3,
    delta_p_kg_m_per_s: Vector3,
) -> CanonicalKeplerPairTangent:
    return CanonicalKeplerPairTangent(
        delta_q_m=delta_q_m,
        delta_p_kg_m_per_s=delta_p_kg_m_per_s,
        unit_system_id=plan.unit_system_id,
        layout_fingerprint=plan.layout_fingerprint,
        model_fingerprint=plan.model_fingerprint,
        pair_plan_fingerprint=plan.fingerprint,
    )


def _physical_evaluation(
    plan: CanonicalKeplerPairPlan,
    state: CanonicalKeplerPairState,
    duration_s: float,
    domain: _DomainValues,
    root: _RootSolution,
) -> _PhysicalEvaluation:
    if duration_s == 0.0:
        return _PhysicalEvaluation(state, 0.0, 0.0, 0.0, 0.0, domain.velocity_m_per_s)
    gravitational_parameter = plan.gravitational_parameter_m3_s2
    radius0 = domain.radius_m
    radius = root.radius_m
    gs3 = root.gs3
    f_hat = -gravitational_parameter * gs3[2] / radius0
    g_value = duration_s - gravitational_parameter * gs3[3]
    fdot = -gravitational_parameter * gs3[1] / radius0 / radius
    gdot_hat = -gravitational_parameter * gs3[2] / radius
    velocity0 = domain.velocity_m_per_s
    q1 = tuple(
        state.q_m[index]
        + (f_hat * state.q_m[index] + g_value * velocity0[index])
        for index in range(3)
    )
    velocity1 = tuple(
        velocity0[index]
        + (fdot * state.q_m[index] + gdot_hat * velocity0[index])
        for index in range(3)
    )
    momentum1 = tuple(plan.reduced_mass_kg * value for value in velocity1)
    return _PhysicalEvaluation(
        _new_state(plan, q1, momentum1),  # type: ignore[arg-type]
        f_hat,
        g_value,
        fdot,
        gdot_hat,
        velocity0,
    )


def kepler_drift(
    plan: CanonicalKeplerPairPlan,
    state: CanonicalKeplerPairState,
    duration_s: float,
) -> KeplerDriftResult:
    """Apply the exact bound-elliptic two-body flow for one fixed duration."""

    duration = _duration(duration_s)
    domain = _domain_values(plan, state, duration)
    root = _solve_universal_anomaly(plan, state, duration, domain)
    physical = _physical_evaluation(plan, state, duration, domain, root)
    return KeplerDriftResult(physical.state, root.diagnostics)


def kepler_drift_tangent(
    plan: CanonicalKeplerPairPlan,
    state: CanonicalKeplerPairState,
    tangent: CanonicalKeplerPairTangent,
    duration_s: float,
) -> KeplerTangentResult:
    """Apply the analytic initial-state Jacobian to one canonical direction."""

    duration = _duration(duration_s)
    domain = _domain_values(plan, state, duration)
    _require_plan_tangent(plan, tangent)
    root = _solve_universal_anomaly(plan, state, duration, domain)
    physical = _physical_evaluation(plan, state, duration, domain, root)
    if duration == 0.0:
        return KeplerTangentResult(physical.state, tangent, root.diagnostics)
    if tangent.delta_q_m == (0.0, 0.0, 0.0) and tangent.delta_p_kg_m_per_s == (
        0.0,
        0.0,
        0.0,
    ):
        zero = _new_tangent(plan, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        return KeplerTangentResult(physical.state, zero, root.diagnostics)

    reduced_mass = plan.reduced_mass_kg
    gravitational_parameter = plan.gravitational_parameter_m3_s2
    q0 = state.q_m
    velocity0 = physical.velocity0_m_per_s
    delta_q0 = tangent.delta_q_m
    delta_velocity0 = tuple(value / reduced_mass for value in tangent.delta_p_kg_m_per_s)
    radius0 = domain.radius_m
    radius_inverse0 = 1.0 / radius0
    beta = root.diagnostics.beta_m2_per_s2
    eta0 = _dot(q0, velocity0)
    zeta0 = gravitational_parameter - beta * radius0
    gs = _stiefel_gs5(beta, root.x_s_per_m)
    radius_inverse = 1.0 / root.radius_m

    delta_radius0 = _dot(delta_q0, q0) * radius_inverse0
    delta_beta = (
        -2.0
        * gravitational_parameter
        * delta_radius0
        * radius_inverse0
        * radius_inverse0
        - 2.0 * _dot(delta_velocity0, velocity0)  # type: ignore[arg-type]
    )
    delta_eta0 = _dot(delta_q0, velocity0) + _dot(q0, delta_velocity0)  # type: ignore[arg-type]
    delta_zeta0 = -beta * delta_radius0 - radius0 * delta_beta
    g3_beta = 0.5 * (3.0 * gs[5] - root.x_s_per_m * gs[4])
    g2_beta = 0.5 * (2.0 * gs[4] - root.x_s_per_m * gs[3])
    g1_beta = 0.5 * (gs[3] - root.x_s_per_m * gs[2])
    t_beta = eta0 * g2_beta + zeta0 * g3_beta
    delta_x = -radius_inverse * (
        root.x_s_per_m * delta_radius0
        + gs[2] * delta_eta0
        + gs[3] * delta_zeta0
        + t_beta * delta_beta
    )
    delta_g1 = gs[0] * delta_x + g1_beta * delta_beta
    delta_g2 = gs[1] * delta_x + g2_beta * delta_beta
    delta_g3 = gs[2] * delta_x + g3_beta * delta_beta
    delta_radius = (
        delta_radius0
        + gs[1] * delta_eta0
        + gs[2] * delta_zeta0
        + eta0 * delta_g1
        + zeta0 * delta_g2
    )
    delta_f_hat = (
        gravitational_parameter
        * gs[2]
        * delta_radius0
        * radius_inverse0
        * radius_inverse0
        - gravitational_parameter * delta_g2 * radius_inverse0
    )
    delta_g = -gravitational_parameter * delta_g3
    delta_fdot = (
        -gravitational_parameter
        * delta_g1
        * radius_inverse0
        * radius_inverse
        + gravitational_parameter
        * gs[1]
        * (delta_radius0 * radius_inverse0 + delta_radius * radius_inverse)
        * radius_inverse0
        * radius_inverse
    )
    delta_gdot_hat = (
        -gravitational_parameter * delta_g2 * radius_inverse
        + gravitational_parameter
        * gs[2]
        * delta_radius
        * radius_inverse
        * radius_inverse
    )

    delta_q1 = tuple(
        delta_q0[index]
        + (
            physical.f_hat * delta_q0[index]
            + physical.g_s * delta_velocity0[index]
            + delta_f_hat * q0[index]
            + delta_g * velocity0[index]
        )
        for index in range(3)
    )
    delta_velocity1 = tuple(
        delta_velocity0[index]
        + (
            physical.fdot_per_s * delta_q0[index]
            + physical.gdot_hat * delta_velocity0[index]
            + delta_fdot * q0[index]
            + delta_gdot_hat * velocity0[index]
        )
        for index in range(3)
    )
    delta_momentum1 = tuple(reduced_mass * value for value in delta_velocity1)
    result_tangent = _new_tangent(
        plan, delta_q1, delta_momentum1  # type: ignore[arg-type]
    )
    return KeplerTangentResult(physical.state, result_tangent, root.diagnostics)
