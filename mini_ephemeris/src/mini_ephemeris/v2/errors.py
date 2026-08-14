"""Typed errors for the isolated v2 foundation contracts."""


class V2FoundationError(ValueError):
    """Base class for rejected v2 foundation inputs or contracts."""


class InvalidModel(V2FoundationError):
    """Raised when an immutable model description is incomplete or invalid."""


class LayoutMismatch(V2FoundationError):
    """Raised when body identity or ordering differs across an API boundary."""


class InvalidState(V2FoundationError):
    """Raised when a public state violates shape, unit, or finiteness rules."""


class InvalidTimebase(V2FoundationError):
    """Raised when exact control-time identity cannot be represented safely."""


class KernelContractError(V2FoundationError):
    """Raised when a force or JVP provider violates its semantic contract."""


class ThresholdScopeMismatch(V2FoundationError):
    """Raised when a threshold is applied outside its declared evidence scope."""
