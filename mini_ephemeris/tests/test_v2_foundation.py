"""Focused, non-dynamical tests for the isolated Step 3g1a v2 foundation."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from mini_ephemeris.v2 import (
    AccountingDomain,
    AccountingEvent,
    BodyId,
    CanonicalJacobiState,
    CanonicalJacobiTangentState,
    CartesianAcceleration,
    CartesianAccelerationJVP,
    CartesianPositionTangent,
    ComparisonClass,
    CompiledLayout,
    ExactSeconds,
    ForceEvaluationContext,
    InertialCartesianState,
    InvalidModel,
    InvalidState,
    InvalidTimebase,
    KernelContractError,
    LayoutMismatch,
    MacroTimebase,
    PhysicalModel,
    SI_UNITS,
    ThresholdApplicability,
    ThresholdScopeMismatch,
    ThresholdUseContext,
    UnitSystem,
    capture_observer_snapshot,
    evaluate_force,
    evaluate_jvp,
    observe,
    require_canonical_tangent_compatible,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST22 = ROOT / "ephemeris_experiment_runner/manifests/22_m0_step3g1a_v2_foundation_v1.json"
V2_ROOT = ROOT / "mini_ephemeris/src/mini_ephemeris/v2"


def sample_layout(order=("sun", "planet"), central="sun"):
    return CompiledLayout(order, central)


def sample_model(**overrides):
    layout = overrides.pop("layout", sample_layout())
    values = {
        "model_id": "synthetic_foundation_model",
        "schema_version": "1",
        "layout": layout,
        "masses_kg": {body.value: float(index + 1) for index, body in enumerate(layout.body_ids)},
        "gravitational_constant_si": 6.67430e-11,
        "units": SI_UNITS,
        "enabled_effects": ["synthetic_linear_fixture"],
        "provenance": {"source": "unit-test", "version": "1"},
    }
    values.update(overrides)
    return PhysicalModel(**values)


def sample_state(layout=None, unit_system_id="si_v1"):
    layout = layout or sample_layout()
    return InertialCartesianState(
        layout,
        ((1.0, 2.0, 3.0), (-2.0, 4.0, 0.5)),
        ((0.1, 0.2, 0.3), (0.0, -0.4, 0.2)),
        unit_system_id,
    )


def sample_direction(layout=None, scale=1.0, unit_system_id="si_v1"):
    layout = layout or sample_layout()
    return CartesianPositionTangent(
        layout,
        ((scale, -2.0 * scale, 0.5 * scale), (3.0 * scale, scale, -scale)),
        unit_system_id,
    )


class SyntheticLinearProvider:
    """Stateless analytic fixture; this is not a physical force provider."""

    __slots__ = ()

    def evaluate(self, model, state, context):
        del context
        values = tuple(tuple(-2.0 * value for value in row) for row in state.positions_m)
        return CartesianAcceleration(model.layout, values, model.units.identifier)

    def jvp(self, model, state, direction, context):
        del state, context
        values = tuple(
            tuple(0.0 if value == 0.0 else -2.0 * value for value in row)
            for row in direction.delta_positions_m
        )
        return CartesianAccelerationJVP(model.layout, values, model.units.identifier)


class ModelContractTests(unittest.TestCase):
    def test_body_identity_and_explicit_order(self):
        layout = sample_layout(("planet", "sun"), "sun")
        self.assertEqual(layout.index_of("planet"), 0)
        self.assertEqual(layout.index_of(BodyId("sun")), 1)
        self.assertNotEqual(layout.fingerprint, sample_layout().fingerprint)

    def test_duplicate_missing_and_unknown_body_rejected(self):
        with self.assertRaises(InvalidModel):
            CompiledLayout(("sun", "sun"), "sun")
        with self.assertRaises(InvalidModel):
            CompiledLayout(("planet",), "sun")
        with self.assertRaises(LayoutMismatch):
            sample_layout().index_of("missing")

    def test_model_is_immutable_and_defensively_copies_inputs(self):
        masses = {"sun": 1.0, "planet": 2.0}
        effects = ["synthetic_linear_fixture"]
        provenance = {"source": "unit-test", "version": "1"}
        model = sample_model(masses_kg=masses, enabled_effects=effects, provenance=provenance)
        masses["sun"] = 99.0
        effects.append("changed")
        provenance["source"] = "changed"
        self.assertEqual(model.mass_kg("sun"), 1.0)
        self.assertEqual(model.enabled_effects, ("synthetic_linear_fixture",))
        self.assertIn(("source", "unit-test"), model.provenance)
        with self.assertRaises(FrozenInstanceError):
            model.model_id = "changed"

    def test_invalid_ids_masses_constants_units_and_provenance_rejected(self):
        for invalid in ("", "Sun", "has space", " leading"):
            with self.subTest(invalid=invalid), self.assertRaises(InvalidModel):
                BodyId(invalid)
        for invalid in (0.0, -1.0, math.inf, math.nan, True, "1.0"):
            with self.subTest(mass=invalid), self.assertRaises(InvalidModel):
                sample_model(masses_kg={"sun": invalid, "planet": 2.0})
        for invalid in (0.0, -1.0, math.inf, math.nan, True, "1.0"):
            with self.subTest(constant=invalid), self.assertRaises(InvalidModel):
                sample_model(gravitational_constant_si=invalid)
        with self.assertRaises(InvalidModel):
            sample_model(masses_kg={"sun": 1.0})
        with self.assertRaises(InvalidModel):
            sample_model(enabled_effects=[])
        with self.assertRaises(InvalidModel):
            sample_model(provenance={})
        with self.assertRaises(InvalidModel):
            sample_model(provenance={"source": object()})
        with self.assertRaises(InvalidModel):
            UnitSystem("", "m", "s", "kg", "m/s", "kg*m/s", "m/s^2")

    def test_canonical_serialization_ignores_mapping_and_collection_order(self):
        first = sample_model(
            masses_kg={"sun": 1.0, "planet": 2.0},
            enabled_effects=["z_effect", "a_effect"],
            provenance={"z": "last", "a": "first"},
        )
        second = sample_model(
            masses_kg={"planet": 2.0, "sun": 1.0},
            enabled_effects=["a_effect", "z_effect"],
            provenance={"a": "first", "z": "last"},
        )
        self.assertEqual(first.canonical_bytes(), second.canonical_bytes())
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertIn(b"0x1.", first.canonical_bytes())

    def test_fingerprint_sensitive_to_every_material_field(self):
        base = sample_model()
        variants = [
            sample_model(model_id="other"),
            sample_model(schema_version="2"),
            sample_model(
                layout=sample_layout(("planet", "sun"), "sun"),
                masses_kg={"sun": 1.0, "planet": 2.0},
            ),
            sample_model(layout=sample_layout(("sun", "planet"), "planet")),
            sample_model(masses_kg={"sun": 2.0, "planet": 2.0}),
            sample_model(masses_kg={"sun": 1.0, "planet": 3.0}),
            sample_model(gravitational_constant_si=6.7e-11),
            sample_model(
                units=UnitSystem("other", "m", "s", "kg", "m/s", "kg*m/s", "m/s^2")
            ),
            sample_model(enabled_effects=["another_effect"]),
            sample_model(provenance={"source": "unit-test", "version": "2"}),
        ]
        unit_fields = (
            ("identifier", "other"),
            ("length", "km"),
            ("time", "day"),
            ("mass", "g"),
            ("velocity", "km/s"),
            ("momentum", "g*cm/s"),
            ("acceleration", "km/s^2"),
        )
        for field, value in unit_fields:
            unit_values = SI_UNITS.canonical_payload()
            unit_values[field] = value
            variants.append(sample_model(units=UnitSystem(**unit_values)))
        for variant in variants:
            with self.subTest(variant=variant.canonical_payload()):
                self.assertNotEqual(base.fingerprint, variant.fingerprint)


class StateContractTests(unittest.TestCase):
    def test_coordinate_types_are_semantically_distinct(self):
        layout = sample_layout()
        inertial = sample_state(layout)
        canonical = CanonicalJacobiState(
            layout,
            inertial.positions_m,
            ((10.0, 20.0, 30.0), (40.0, 50.0, 60.0)),
            "si_v1",
        )
        tangent = CanonicalJacobiTangentState(
            layout,
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ((0.0, 0.0, 1.0), (1.0, 1.0, 0.0)),
            "si_v1",
        )
        self.assertIsNot(type(inertial), type(canonical))
        self.assertIsNot(type(canonical), type(tangent))
        self.assertEqual(canonical.p_kg_m_per_s[0], (10.0, 20.0, 30.0))
        self.assertEqual(tangent.delta_p_kg_m_per_s[1], (1.0, 1.0, 0.0))

    def test_canonical_state_tangent_layout_compatibility(self):
        layout = sample_layout()
        state = CanonicalJacobiState(
            layout,
            ((0.0, 0.0, 0.0),) * 2,
            ((1.0, 0.0, 0.0),) * 2,
            "si_v1",
        )
        matching = CanonicalJacobiTangentState(
            layout,
            ((0.0, 1.0, 0.0),) * 2,
            ((0.0, 0.0, 1.0),) * 2,
            "si_v1",
        )
        self.assertIsNone(require_canonical_tangent_compatible(state, matching))
        reordered = CanonicalJacobiTangentState(
            sample_layout(("planet", "sun"), "sun"),
            ((0.0, 1.0, 0.0),) * 2,
            ((0.0, 0.0, 1.0),) * 2,
            "si_v1",
        )
    def test_shape_nonfinite_and_unit_validation(self):
        layout = sample_layout()
        with self.assertRaises(InvalidState):
            InertialCartesianState(layout, ((1.0, 2.0), (0.0, 0.0, 0.0)), ((0, 0, 0),) * 2, "si_v1")
        with self.assertRaises(InvalidState):
            InertialCartesianState(layout, ((0, 0, 0),), ((0, 0, 0),), "si_v1")
        with self.assertRaises(InvalidState):
            InertialCartesianState(layout, ((math.nan, 0, 0), (0, 0, 0)), ((0, 0, 0),) * 2, "si_v1")
        with self.assertRaises(InvalidState):
            InertialCartesianState(layout, ((True, 0, 0), (0, 0, 0)), ((0, 0, 0),) * 2, "si_v1")
        with self.assertRaises(InvalidState):
            InertialCartesianState(layout, (("1", 0, 0), (0, 0, 0)), ((0, 0, 0),) * 2, "si_v1")
        with self.assertRaises(InvalidState):
            InertialCartesianState(layout, ((0, 0, 0),) * 2, ((0, 0, 0),) * 2, "")

    def test_numpy_inputs_are_detached_and_public_storage_is_immutable(self):
        layout = sample_layout()
        positions = np.arange(6.0).reshape(2, 3)
        velocities = np.ones((2, 3))
        state = InertialCartesianState(layout, positions, velocities, "si_v1")
        positions[:] = -999.0
        velocities[:] = -888.0
        self.assertEqual(state.positions_m[0], (0.0, 1.0, 2.0))
        self.assertEqual(state.velocities_m_per_s[0], (1.0, 1.0, 1.0))
        with self.assertRaises(TypeError):
            state.positions_m[0][0] = 2.0
        with self.assertRaises(FrozenInstanceError):
            state.unit_system_id = "changed"


class KernelProtocolTests(unittest.TestCase):
    def setUp(self):
        self.model = sample_model()
        self.state = sample_state()
        self.direction = sample_direction()
        self.context = ForceEvaluationContext(
            MacroTimebase(ExactSeconds(0), ExactSeconds(1, 10)).at(3),
            AccountingDomain.MAP_STAGE,
            "synthetic-3",
        )
        self.provider = SyntheticLinearProvider()

    def test_force_is_deterministic_and_inputs_remain_bitwise_unchanged(self):
        before_model = self.model.canonical_bytes()
        before_state = (self.state.positions_m, self.state.velocities_m_per_s)
        first = evaluate_force(self.provider, self.model, self.state, self.context)
        second = evaluate_force(self.provider, self.model, self.state, self.context)
        self.assertEqual(first, second)
        self.assertEqual(self.model.canonical_bytes(), before_model)
        self.assertEqual((self.state.positions_m, self.state.velocities_m_per_s), before_state)
        self.assertFalse(hasattr(self.provider, "__dict__"))

    def test_jvp_linearity_and_exact_zero_direction(self):
        u = sample_direction(scale=0.25)
        v = sample_direction(scale=-1.5)
        a = 2.0
        b = -0.5
        combined_rows = tuple(
            tuple(a * ux + b * vx for ux, vx in zip(urow, vrow))
            for urow, vrow in zip(u.delta_positions_m, v.delta_positions_m)
        )
        combined = CartesianPositionTangent(self.model.layout, combined_rows, "si_v1")
        lhs = evaluate_jvp(self.provider, self.model, self.state, combined, self.context)
        ju = evaluate_jvp(self.provider, self.model, self.state, u, self.context)
        jv = evaluate_jvp(self.provider, self.model, self.state, v, self.context)
        rhs = tuple(
            tuple(a * ux + b * vx for ux, vx in zip(urow, vrow))
            for urow, vrow in zip(ju.values_m_per_s2, jv.values_m_per_s2)
        )
        self.assertEqual(lhs.values_m_per_s2, rhs)
        zero = CartesianPositionTangent(self.model.layout, ((0.0, 0.0, 0.0),) * 2, "si_v1")
        zero_result = evaluate_jvp(
            self.provider, self.model, self.state, zero, self.context
        ).values_m_per_s2
        self.assertEqual(zero_result, ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
        self.assertTrue(all(value.hex() == "0x0.0p+0" for row in zero_result for value in row))

    def test_layout_and_unit_mismatches_are_rejected_before_provider(self):
        other_layout = sample_layout(("sun", "other"), "sun")
        with self.assertRaises(LayoutMismatch):
            evaluate_force(self.provider, self.model, sample_state(other_layout), self.context)
        with self.assertRaises(InvalidState):
            sample_state(unit_system_id="other")
        other_units = UnitSystem(
            "other", "km", "day", "kg", "km/day", "kg*km/day", "km/day^2"
        )
        with self.assertRaises(LayoutMismatch):
            evaluate_force(self.provider, sample_model(units=other_units), self.state, self.context)
        with self.assertRaises(LayoutMismatch):
            evaluate_jvp(
                self.provider,
                self.model,
                self.state,
                sample_direction(other_layout),
                self.context,
            )

    def test_wrong_output_semantics_are_rejected(self):
        class WrongProvider:
            def evaluate(self, model, state, context):
                del model, state, context
                return object()

            def jvp(self, model, state, direction, context):
                del model, state, direction, context
                return object()

        with self.assertRaises(KernelContractError):
            evaluate_force(WrongProvider(), self.model, self.state, self.context)
        with self.assertRaises(KernelContractError):
            evaluate_jvp(WrongProvider(), self.model, self.state, self.direction, self.context)


class TimebaseTests(unittest.TestCase):
    def test_exact_roundtrip_equality_and_hashing(self):
        original = MacroTimebase(ExactSeconds(-3, 2), ExactSeconds(2, 6), 10**12)
        restored = MacroTimebase.from_canonical_payload(original.canonical_payload())
        self.assertEqual(original, restored)
        self.assertEqual(hash(original), hash(restored))
        self.assertEqual(original.canonical_bytes(), restored.canonical_bytes())
        self.assertEqual(original.fingerprint, restored.fingerprint)
        self.assertEqual(original.interval, ExactSeconds(1, 3))

    def test_negative_and_very_large_direct_indices(self):
        timebase = MacroTimebase(ExactSeconds(1, 7), ExactSeconds(1, 10), 2**62)
        self.assertEqual(timebase.at(-3).seconds, ExactSeconds(-11, 70))
        large = timebase.at(2**60)
        self.assertEqual(large.step_index, 2**60)
        self.assertEqual(
            large.seconds,
            ExactSeconds(10 + 7 * (2**60), 70),
        )

    def test_invalid_intervals_denominators_indices_and_bounds(self):
        with self.assertRaises(InvalidTimebase):
            ExactSeconds(1, 0)
        with self.assertRaises(InvalidTimebase):
            MacroTimebase(ExactSeconds(0), ExactSeconds(0))
        with self.assertRaises(InvalidTimebase):
            MacroTimebase(ExactSeconds(0), ExactSeconds(-1))
        with self.assertRaises(InvalidTimebase):
            MacroTimebase(ExactSeconds(0), ExactSeconds(1), -1)
        timebase = MacroTimebase(ExactSeconds(0), ExactSeconds(1), 10)
        with self.assertRaises(InvalidTimebase):
            timebase.at(11)
        with self.assertRaises(InvalidTimebase):
            timebase.at(1.0)

    def test_binary64_boundary_and_accumulation_drift(self):
        timebase = MacroTimebase(ExactSeconds(0), ExactSeconds(1, 10), 2_000_000)
        direct = timebase.at(1_000_000).seconds.to_binary64()
        accumulated = 0.0
        for _ in range(1_000_000):
            accumulated += 0.1
        self.assertEqual(direct, 100_000.0)
        self.assertNotEqual(direct, accumulated)


class OwnershipAndAccountingTests(unittest.TestCase):
    def test_snapshot_detaches_from_mutable_source(self):
        positions = np.arange(6.0).reshape(2, 3)
        velocities = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        snapshot = capture_observer_snapshot(
            layout=sample_layout(),
            positions_m=positions,
            velocities_m_per_s=velocities,
            unit_system_id="si_v1",
            time=MacroTimebase(ExactSeconds(0), ExactSeconds(1)).at(4),
            metadata={"purpose": "synthetic-observer"},
        )
        positions[:] = -9.0
        velocities[0][0] = -8.0
        self.assertEqual(snapshot.state.positions_m[0], (0.0, 1.0, 2.0))
        self.assertEqual(snapshot.state.velocities_m_per_s[0], (1.0, 2.0, 3.0))
        with self.assertRaises(FrozenInstanceError):
            snapshot.time = MacroTimebase(ExactSeconds(0), ExactSeconds(1)).at(5)
        with self.assertRaises(InvalidState):
            capture_observer_snapshot(
                layout=sample_layout(),
                positions_m=((0.0, 0.0, 0.0),) * 2,
                velocities_m_per_s=((0.0, 0.0, 0.0),) * 2,
                unit_system_id="si_v1",
                time=MacroTimebase(ExactSeconds(0), ExactSeconds(1)).at(0),
                metadata={"identity": object()},
            )

    def test_observer_only_receives_immutable_snapshot(self):
        snapshot = capture_observer_snapshot(
            layout=sample_layout(),
            positions_m=((0.0, 0.0, 0.0),) * 2,
            velocities_m_per_s=((0.0, 0.0, 0.0),) * 2,
            unit_system_id="si_v1",
            time=MacroTimebase(ExactSeconds(0), ExactSeconds(1)).at(0),
            metadata={"purpose": "observer-boundary"},
        )

        def observer(value):
            self.assertFalse(hasattr(value, "live_state"))
            with self.assertRaises(FrozenInstanceError):
                value.state = sample_state()
            return value.state.positions_m

        self.assertEqual(observe(snapshot, observer), ((0.0, 0.0, 0.0),) * 2)
        with self.assertRaises(TypeError):
            observe(object(), observer)

    def test_accounting_domains_are_typed_and_disjoint(self):
        events = {
            AccountingEvent(domain, "synthetic")
            for domain in AccountingDomain
        }
        self.assertEqual(len(events), 4)
        with self.assertRaises(TypeError):
            AccountingEvent("map_stage", "synthetic")


class HighFindingContractTests(unittest.TestCase):
    def test_threshold_scope_rejects_mismatched_contexts(self):
        base = ThresholdUseContext(
            map_id="same-map-v1",
            trajectory_id="trajectory-a",
            tangent_seed_id="seed-12345",
            normalization_id="euclidean-v1",
            coordinate_id="canonical-jacobi-v1",
            rescaling_history_id="none",
            timestamps_id="ticks-0-10",
            comparison_class=ComparisonClass.IMPLEMENTATION_EQUIVALENCE,
        )
        applicability = ThresholdApplicability("manifest10-direction", (base,))
        self.assertTrue(applicability.accepts(base))
        applicability.require_compatible(base)
        replacements = {
            "map_id": "different-map",
            "trajectory_id": "trajectory-b",
            "tangent_seed_id": "other-seed",
            "normalization_id": "other-norm",
            "coordinate_id": "other-coordinates",
            "rescaling_history_id": "rescaled",
            "timestamps_id": "other-ticks",
            "comparison_class": ComparisonClass.DIFFERENT_MAP_PHYSICAL,
        }
        for field, value in replacements.items():
            with self.subTest(field=field), self.assertRaises(ThresholdScopeMismatch):
                applicability.require_compatible(replace(base, **{field: value}))

    def test_high_finding_angle_requirement_is_explicit(self):
        manifest = json.loads(MANIFEST22.read_text(encoding="utf-8"))
        requirement = manifest["implementation_requirements"]["V2-DIAG-ANGLE-001"]
        self.assertIn("atan2", requirement)
        self.assertIn("chord", requirement)
        self.assertIn("G0-001", manifest["high_finding_traceability"])

    def test_both_high_findings_have_named_tests_and_requirements(self):
        manifest = json.loads(MANIFEST22.read_text(encoding="utf-8"))
        traced = manifest["high_finding_traceability"]
        self.assertEqual(set(traced), {"G0-001", "G0-002"})
        self.assertEqual(traced["G0-001"]["requirement"], "V2-DIAG-ANGLE-001")
        self.assertEqual(traced["G0-002"]["requirement"], "V2-THRESH-SCOPE-001")
        self.assertTrue(all(item["acceptance_tests"] for item in traced.values()))


class IsolationAndFreshProcessTests(unittest.TestCase):
    def _run_probe(self, hash_seed, locale_name):
        code = """
import json
from mini_ephemeris.v2 import CompiledLayout, ExactSeconds, MacroTimebase, PhysicalModel, SI_UNITS
layout = CompiledLayout(['sun','planet'], 'sun')
model = PhysicalModel(model_id='probe', schema_version='1', layout=layout,
    masses_kg={'planet':2.0,'sun':1.0}, gravitational_constant_si=6.67430e-11,
    units=SI_UNITS, enabled_effects={'effect-b','effect-a'},
    provenance={'z':'last','a':'first'})
timebase = MacroTimebase(ExactSeconds(-1,3), ExactSeconds(1,10), 2**62)
print(json.dumps({'model':model.fingerprint,'time':timebase.fingerprint,
    'target':timebase.at(2**60).canonical_payload()}, sort_keys=True, separators=(',',':')))
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "mini_ephemeris/src")
        env["PYTHONHASHSEED"] = str(hash_seed)
        env["LC_ALL"] = locale_name
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout

    def test_fresh_process_hash_seed_and_locale_consistency(self):
        first = self._run_probe(1, "C")
        second = self._run_probe(8675309, "C.UTF-8")
        self.assertEqual(first, second)

    def test_v2_imports_no_rebound_and_exposes_no_primitive_entrypoint(self):
        code = """
import json, sys
import mini_ephemeris.v2 as v2
print(json.dumps({'rebound': 'rebound' in sys.modules,
                  'reboundx': 'reboundx' in sys.modules,
                  'exports': sorted(v2.__all__)}, sort_keys=True))
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "mini_ephemeris/src")
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertFalse(payload["rebound"])
        self.assertFalse(payload["reboundx"])
        prohibited_exports = {"integrate", "step", "kepler", "kick", "lazy", "corrector", "whckl", "megno", "lcn"}
        self.assertTrue(prohibited_exports.isdisjoint(name.lower() for name in payload["exports"]))

    def test_source_has_no_hidden_dynamics_or_dependency_imports(self):
        forbidden_calls = {"integrate", "step", "kepler", "kick", "lazy", "corrector", "whckl", "megno", "lcn"}
        for path in sorted(V2_ROOT.glob("*.py")):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            imports = {
                alias.name.split(".")[0]
                for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names
            }
            self.assertTrue({"rebound", "reboundx"}.isdisjoint(imports), path)
            executable_names = {
                node.name.lower()
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            }
            self.assertTrue(forbidden_calls.isdisjoint(executable_names), path)


    def test_public_api_docstrings_and_annotations(self):
        def assert_signature(node, path):
            arguments = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
            for argument in arguments:
                if argument.arg not in {"self", "cls"}:
                    self.assertIsNotNone(
                        argument.annotation,
                        f"{path}:{node.lineno} missing annotation for {argument.arg}",
                    )
            self.assertIsNotNone(
                node.returns,
                f"{path}:{node.lineno} missing return annotation for {node.name}",
            )

        for path in sorted(V2_ROOT.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                    with self.subTest(path=path.name, api=node.name):
                        self.assertIsNotNone(ast.get_docstring(node))
                        assert_signature(node, path)
                elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                    with self.subTest(path=path.name, api=node.name):
                        self.assertIsNotNone(ast.get_docstring(node))
                    for method in node.body:
                        if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            continue
                        if method.name == "__init__" or not method.name.startswith("_"):
                            with self.subTest(path=path.name, api=f"{node.name}.{method.name}"):
                                if method.name != "__init__":
                                    self.assertIsNotNone(ast.get_docstring(method))
                                assert_signature(method, path)
if __name__ == "__main__":
    unittest.main()
