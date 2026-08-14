from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
import tempfile
import unittest

from mini_ephemeris.m0_step3g0_reporting import DOCUMENTATION_DIRECTORY, STATIC_ARTIFACTS, generate
from mini_ephemeris.m0_step3g0_verification import sha256_file


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "ephemeris_experiment_runner/manifests/21_m0_step3g0_verification_architecture_audit_v1.json"
DOCS = ROOT / DOCUMENTATION_DIRECTORY
SUMMARY = DOCS / "m0_step3g0_verification_architecture_audit_summary.json"
REPORT = DOCS / "m0_step3g0_verification_architecture_audit_report.md"
FINDINGS = DOCS / "code_review_findings.json"
THRESHOLDS = DOCS / "threshold_provenance.csv"
LITERATURE = DOCS / "literature_novelty_matrix.csv"


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


class Step3g0ArtifactTests(unittest.TestCase):
    def test_manifest_status_and_integrity_contract(self) -> None:
        manifest = load_json(MANIFEST)
        self.assertEqual(manifest["initial_integrity_baseline"]["head"], "997e5713aedb6e489c886e060af8fc6010b5c3b4")
        self.assertEqual(
            manifest["allowed_final_statuses"],
            [
                "STEP3G0_VERIFICATION_ARCHITECTURE_AUDIT_COMPLETE",
                "STEP3G0_VERIFICATION_ARCHITECTURE_AUDIT_INCONCLUSIVE",
                "BLOCKED",
            ],
        )
        self.assertEqual(
            manifest["allowed_primary_findings"],
            ["V2_CORE_SPECIFICATION_READY", "V2_CORE_SPECIFICATION_NOT_READY", "MIXED_OR_INCONCLUSIVE", "NOT_EVALUATED"],
        )
        for relative, expected in manifest["protected_files"].items():
            self.assertEqual(sha256_file(ROOT / relative), expected)
        for relative, expected in manifest["historical_documents"].items():
            self.assertEqual(sha256_file(ROOT / relative), expected)

    def test_audit_modules_contain_no_timestep_call(self) -> None:
        modules = (
            ROOT / "mini_ephemeris/src/mini_ephemeris/m0_step3g0_verification.py",
            ROOT / "mini_ephemeris/src/mini_ephemeris/m0_step3g0_reporting.py",
        )
        for module in modules:
            tree = ast.parse(module.read_text(encoding="utf-8"))
            calls = [
                node.func.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"integrate", "step", "integrate_rebound_streaming"}
            ]
            self.assertEqual(calls, [], module)

    def test_findings_have_required_schema_and_no_critical(self) -> None:
        findings = load_json(FINDINGS)["findings"]
        required = {
            "finding_id", "file", "symbol", "requirement_id", "severity", "direct_evidence",
            "scientific_consequence", "existing_test_coverage", "missing_test", "scope_effect",
            "recommended_disposition",
        }
        self.assertEqual(len(findings), 11)
        self.assertEqual(len({item["finding_id"] for item in findings}), len(findings))
        for item in findings:
            self.assertTrue(required.issubset(item))
            self.assertIn(item["severity"], {"CRITICAL", "HIGH", "MEDIUM", "LOW", "INFORMATIONAL"})
        self.assertFalse(any(item["severity"] == "CRITICAL" for item in findings))

    def test_threshold_and_literature_classifications_are_closed(self) -> None:
        allowed = {
            "VALID_FOR_CURRENT_USE", "VALID_ONLY_FOR_IMPLEMENTATION_EQUIVALENCE",
            "VALID_ONLY_FOR_SAME_MAP_REPRODUCIBILITY", "ILL_CONDITIONED",
            "PHYSICALLY_UNJUSTIFIED", "PROVENANCE_INSUFFICIENT",
        }
        with THRESHOLDS.open(newline="", encoding="utf-8") as handle:
            thresholds = list(csv.DictReader(handle))
        self.assertEqual(len(thresholds), 20)
        self.assertTrue(all(item["classification"] in allowed for item in thresholds))
        with LITERATURE.open(newline="", encoding="utf-8") as handle:
            literature = list(csv.DictReader(handle))
        self.assertEqual(literature[-1]["opportunity_relevance"], "NOVEL_COMBINATION_OR_EXTENSION")
        joined = " ".join(item["title"] + " " + item["authors"] + " " + item["finding"] for item in literature)
        for topic in ("Mikkola", "WHFast", "WHCKL", "MEGNO", "differentiable", "issues and pull requests"):
            self.assertIn(topic, joined)

    def test_compact_regeneration_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_root = Path(first)
            second_root = Path(second)
            first_summary = first_root / "summary.json"
            second_summary = second_root / "summary.json"
            generate(ROOT, output_root=first_root / "output", summary_path=first_summary)
            generate(ROOT, output_root=second_root / "output", summary_path=second_summary)
            self.assertEqual(first_summary.read_bytes(), second_summary.read_bytes())
            self.assertEqual(
                (first_root / "output/audit_evidence.json").read_bytes(),
                (second_root / "output/audit_evidence.json").read_bytes(),
            )

    def test_summary_report_and_artifact_inventory_agree(self) -> None:
        summary = load_json(SUMMARY)
        report = REPORT.read_text(encoding="utf-8")
        self.assertEqual(summary["final_status"], "STEP3G0_VERIFICATION_ARCHITECTURE_AUDIT_COMPLETE")
        self.assertEqual(summary["primary_finding"], "V2_CORE_SPECIFICATION_READY")
        self.assertEqual(summary["verification_envelope"], "VERIFIED_WITHIN_DOCUMENTED_MODEL_AND_NUMERICAL_ENVELOPE")
        self.assertEqual(summary["callback_classification"], "CALLBACK_ACCOUNTING_EXACTLY_RECONCILED")
        self.assertEqual(summary["callback_accounting"]["source_schedule_total"], 29_226_432)
        self.assertFalse(summary["production_qualified"])
        for value in (summary["final_status"], summary["primary_finding"], summary["verification_envelope"]):
            self.assertIn(value, report)
        for name in STATIC_ARTIFACTS:
            self.assertEqual(summary["artifact_inventory_sha256"][name], sha256_file(DOCS / name))

    def test_all_json_is_strict_and_all_required_outputs_exist(self) -> None:
        for path in (MANIFEST, SUMMARY, FINDINGS):
            load_json(path)
        for name in STATIC_ARTIFACTS:
            path = DOCS / name
            self.assertTrue(path.is_file() and path.stat().st_size > 0, path)


if __name__ == "__main__":
    unittest.main()
