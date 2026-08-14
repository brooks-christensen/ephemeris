"""Typed threshold-applicability contracts carried from Step 3g0 finding G0-002."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

from .errors import ThresholdScopeMismatch


class ComparisonClass(str, Enum):
    """Scientific comparison classes whose thresholds are not interchangeable."""

    IMPLEMENTATION_EQUIVALENCE = "implementation_equivalence"
    SAME_MAP_REPRODUCIBILITY = "same_map_reproducibility"
    TIMESTEP_CONVERGENCE = "timestep_convergence"
    DIFFERENT_MAP_PHYSICAL = "different_map_physical"


@dataclass(frozen=True)
class ThresholdUseContext:
    """Complete use identity required before a numerical threshold is applied."""

    map_id: str
    trajectory_id: str
    tangent_seed_id: str
    normalization_id: str
    coordinate_id: str
    rescaling_history_id: str
    timestamps_id: str
    comparison_class: ComparisonClass

    def __post_init__(self) -> None:
        for field in (
            "map_id",
            "trajectory_id",
            "tangent_seed_id",
            "normalization_id",
            "coordinate_id",
            "rescaling_history_id",
            "timestamps_id",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ThresholdScopeMismatch(f"{field} must be explicit and nonempty")
        if not isinstance(self.comparison_class, ComparisonClass):
            raise ThresholdScopeMismatch("comparison_class must be typed")


@dataclass(frozen=True)
class ThresholdApplicability:
    """Evidence scope for a threshold; no numeric threshold is invented here."""

    provenance_id: str
    allowed_contexts: Tuple[ThresholdUseContext, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.provenance_id, str) or not self.provenance_id.strip():
            raise ThresholdScopeMismatch("provenance_id must be explicit")
        contexts = tuple(self.allowed_contexts)
        if not contexts or any(not isinstance(value, ThresholdUseContext) for value in contexts):
            raise ThresholdScopeMismatch("allowed_contexts must contain typed contexts")
        if len(set(contexts)) != len(contexts):
            raise ThresholdScopeMismatch("allowed threshold contexts must be unique")
        object.__setattr__(self, "allowed_contexts", contexts)

    def accepts(self, use: ThresholdUseContext) -> bool:
        """Return true only for an exactly declared evidence-use identity."""

        return isinstance(use, ThresholdUseContext) and use in self.allowed_contexts

    def require_compatible(self, use: ThresholdUseContext) -> None:
        """Reject use outside map, trajectory, tangent, time, and purpose scope."""

        if not self.accepts(use):
            raise ThresholdScopeMismatch(
                f"threshold {self.provenance_id!r} is incompatible with the requested context"
            )
