"""Immutable, auditable physical-model descriptions for the v2 namespace."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import re
from typing import Any, Iterable, Mapping, Tuple, Union

from .canonical import canonical_json_bytes, finite_binary64_hex, sha256_hex
from .errors import InvalidModel, LayoutMismatch


_BODY_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _nonempty_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise InvalidModel(f"{field} must be a nonempty trimmed string")
    return value


@dataclass(frozen=True)
class BodyId:
    """Stable scientific body identity independent of dense array position."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not _BODY_ID_PATTERN.fullmatch(self.value):
            raise InvalidModel(
                "BodyId must match [a-z0-9][a-z0-9._-]* and is not a display label"
            )

    def __str__(self) -> str:
        """Return the stable serialized identity."""

        return self.value


BodyIdLike = Union[BodyId, str]


def as_body_id(value: BodyIdLike) -> BodyId:
    """Normalize a body identifier without consulting array position."""

    return value if isinstance(value, BodyId) else BodyId(value)


@dataclass(frozen=True, init=False)
class CompiledLayout:
    """Explicit immutable mapping from stable body IDs to dense positions.

    The ordered tuple is the only source of dense indices. Duplicate IDs and a
    missing central body are rejected before any state can be constructed.
    """

    body_ids: Tuple[BodyId, ...]
    central_body: BodyId

    def __init__(self, body_ids: Iterable[BodyIdLike], central_body: BodyIdLike) -> None:
        ordered = tuple(as_body_id(value) for value in body_ids)
        central = as_body_id(central_body)
        if not ordered:
            raise InvalidModel("layout must contain at least one body")
        if len(set(ordered)) != len(ordered):
            raise InvalidModel("layout body identifiers must be unique")
        if central not in ordered:
            raise InvalidModel("central body must be present in the explicit layout")
        object.__setattr__(self, "body_ids", ordered)
        object.__setattr__(self, "central_body", central)

    def index_of(self, body_id: BodyIdLike) -> int:
        """Return the checked dense position for a stable body identity."""

        target = as_body_id(body_id)
        try:
            return self.body_ids.index(target)
        except ValueError as exc:
            raise LayoutMismatch(f"body {target.value!r} is absent from the layout") from exc

    def canonical_payload(self) -> Mapping[str, Any]:
        """Return the deterministic serialization payload for this layout."""

        return {
            "body_ids": [body.value for body in self.body_ids],
            "central_body": self.central_body.value,
            "schema": "v2.compiled_layout/1",
        }

    @property
    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 identity for body order and center."""

        return sha256_hex(canonical_json_bytes(self.canonical_payload()))


@dataclass(frozen=True)
class UnitSystem:
    """Declared public units; no implicit conversions are performed."""

    identifier: str
    length: str
    time: str
    mass: str
    velocity: str
    momentum: str
    acceleration: str

    def __post_init__(self) -> None:
        for field in (
            "identifier",
            "length",
            "time",
            "mass",
            "velocity",
            "momentum",
            "acceleration",
        ):
            _nonempty_text(getattr(self, field), f"units.{field}")

    def canonical_payload(self) -> Mapping[str, str]:
        """Return all material unit declarations in fixed semantic fields."""

        return {
            "acceleration": self.acceleration,
            "identifier": self.identifier,
            "length": self.length,
            "mass": self.mass,
            "momentum": self.momentum,
            "time": self.time,
            "velocity": self.velocity,
        }


SI_UNITS = UnitSystem(
    identifier="si_v1",
    length="m",
    time="s",
    mass="kg",
    velocity="m/s",
    momentum="kg*m/s",
    acceleration="m/s^2",
)


@dataclass(frozen=True, init=False)
class PhysicalModel:
    """Pure immutable description of model meaning and provenance.

    This contract validates representation, identity, units, and finite positive
    constants. It does not validate that the selected equations describe nature.
    Mutable inputs are normalized to ordered immutable tuples with no retained
    writable aliases.
    """

    model_id: str
    schema_version: str
    layout: CompiledLayout
    masses_kg: Tuple[Tuple[BodyId, float], ...]
    gravitational_constant_si: float
    units: UnitSystem
    enabled_effects: Tuple[str, ...]
    provenance: Tuple[Tuple[str, str], ...]

    def __init__(
        self,
        *,
        model_id: str,
        schema_version: str,
        layout: CompiledLayout,
        masses_kg: Mapping[BodyIdLike, float],
        gravitational_constant_si: float,
        units: UnitSystem,
        enabled_effects: Iterable[str],
        provenance: Mapping[str, str],
    ) -> None:
        if not isinstance(layout, CompiledLayout):
            raise InvalidModel("layout must be a CompiledLayout")
        if not isinstance(units, UnitSystem):
            raise InvalidModel("units must be a UnitSystem")
        normalized_masses = {}
        for raw_id, raw_mass in masses_kg.items():
            body_id = as_body_id(raw_id)
            if body_id in normalized_masses:
                raise InvalidModel(f"duplicate mass identity {body_id.value!r}")
            if isinstance(raw_mass, bool) or not isinstance(raw_mass, Real):
                raise InvalidModel("mass values must be finite positive binary64 numbers")
            mass = float(raw_mass)
            if not math.isfinite(mass) or mass <= 0.0:
                raise InvalidModel("mass values must be finite and positive")
            normalized_masses[body_id] = mass
        expected = set(layout.body_ids)
        actual = set(normalized_masses)
        if actual != expected:
            missing = sorted(body.value for body in expected - actual)
            extra = sorted(body.value for body in actual - expected)
            raise InvalidModel(f"mass identities mismatch; missing={missing}, extra={extra}")
        if isinstance(gravitational_constant_si, bool) or not isinstance(gravitational_constant_si, Real):
            raise InvalidModel("gravitational constant must be a finite positive number")
        constant = float(gravitational_constant_si)
        if not math.isfinite(constant) or constant <= 0.0:
            raise InvalidModel("gravitational constant must be finite and positive")
        effects = tuple(sorted({_nonempty_text(value, "enabled effect") for value in enabled_effects}))
        if not effects:
            raise InvalidModel("at least one physical effect must be declared")
        provenance_items = tuple(
            sorted(
                (
                    _nonempty_text(key, "provenance key"),
                    _nonempty_text(value, "provenance value"),
                )
                for key, value in provenance.items()
            )
        )
        if not provenance_items:
            raise InvalidModel("model provenance must not be empty")
        object.__setattr__(self, "model_id", _nonempty_text(model_id, "model_id"))
        object.__setattr__(self, "schema_version", _nonempty_text(schema_version, "schema_version"))
        object.__setattr__(self, "layout", layout)
        object.__setattr__(
            self,
            "masses_kg",
            tuple((body, normalized_masses[body]) for body in layout.body_ids),
        )
        object.__setattr__(self, "gravitational_constant_si", constant)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "enabled_effects", effects)
        object.__setattr__(self, "provenance", provenance_items)

    def mass_kg(self, body_id: BodyIdLike) -> float:
        """Return the declared mass for a stable identity."""

        target = as_body_id(body_id)
        for body, mass in self.masses_kg:
            if body == target:
                return mass
        raise LayoutMismatch(f"body {target.value!r} is absent from the model")

    def canonical_payload(self) -> Mapping[str, Any]:
        """Return the complete deterministic material-model payload."""

        return {
            "enabled_effects": list(self.enabled_effects),
            "gravitational_constant_si_hex": finite_binary64_hex(
                self.gravitational_constant_si, "gravitational_constant_si"
            ),
            "layout": self.layout.canonical_payload(),
            "masses_kg_hex": [
                {"body_id": body.value, "value": finite_binary64_hex(mass, "mass")}
                for body, mass in self.masses_kg
            ],
            "model_id": self.model_id,
            "provenance": {key: value for key, value in self.provenance},
            "schema": "v2.physical_model/1",
            "schema_version": self.schema_version,
            "units": self.units.canonical_payload(),
        }

    def canonical_bytes(self) -> bytes:
        """Serialize every material field deterministically as UTF-8 JSON."""

        return canonical_json_bytes(self.canonical_payload())

    @property
    def fingerprint(self) -> str:
        """Return the SHA-256 model fingerprint over canonical bytes."""

        return sha256_hex(self.canonical_bytes())
