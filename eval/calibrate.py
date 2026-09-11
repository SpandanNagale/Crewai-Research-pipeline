"""
Judge calibration: measure how well the LLM judge (eval/scorer.py) agrees
with a human rater on the same reports.

"Calibrate the judge. Hand-label ~30 reports, measure judge agreement with
your labels, report the correlation in the README." -- that hand-labeling is
real human judgment; nothing in this file invents it. Running `label` walks
you through your own saved reports (outputs/*.md) and records your 1-5
scores; `score` then runs the judge over the same reports and reports
weighted Cohen's kappa + Spearman correlation (eval/stats.py) between your
labels and the judge's. Until someone actually runs both subcommands, there
is no real number to put in the README -- see the README's Evaluation
section, which stays marked "pending calibration run" until then.

Usage:
    python -m eval.calibrate label --reports-dir outputs --out eval/calibration_labels.json
    python -m eval.calibrate score --labels eval/calibration_labels.json --out eval/calibration_results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from eval.scorer import score_report
from eval.stats import spearman_correlation, weighted_cohens_kappa

LIKERT_CATEGORIES = [1, 2, 3, 4, 5]


def bucket_score(total_0_to_10: float) -> int:
    """Maps a judge's 0-10 total onto the same 1-5 Likert scale a human rater uses."""
    bucket = round(total_0_to_10 / 2) + 1
    return max(1, min(5, bucket))


def load_labels(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_labels(path: Path, labels: Dict[str, int]) -> None:
    path.write_text(json.dumps(labels, indent=2, sort_keys=True), encoding="utf-8")


def find_reports(reports_dir: Path) -> List[Path]:
    return sorted(reports_dir.glob("*.md"))


def compute_agreement(
    labels: Dict[str, int], judge_totals: Dict[str, Optional[float]]
) -> Optional[dict]:
    """
    Both dicts are keyed by report id (filename stem). Only reports with both
    a human label and a successfully-parsed judge score are compared -- a
    judge parse failure (defect #8) is excluded, not coerced into a score.
    Returns None if fewer than 2 overlapping, comparable pairs exist.
    """
    common_ids = [rid for rid in labels if judge_totals.get(rid) is not None]
    if len(common_ids) < 2:
        return None

    human = [labels[rid] for rid in common_ids]
    judge_raw = [judge_totals[rid] for rid in common_ids]
    judge_bucketed = [bucket_score(t) for t in judge_raw]

    return {
        "n": len(common_ids),
        "weighted_kappa": weighted_cohens_kappa(human, judge_bucketed, LIKERT_CATEGORIES),
        "spearman_correlation": spearman_correlation(human, judge_raw),
    }


def _cmd_label(args: argparse.Namespace) -> int:
    reports_dir = Path(args.reports_dir)
    out_path = Path(args.out)
    labels = load_labels(out_path)

    reports = find_reports(reports_dir)
    if not reports:
        print(f"No .md reports found in {reports_dir}. Run some research first.")
        return 1

    for report_path in reports:
        report_id = report_path.stem
        if report_id in labels:
            continue
        print("\n" + "=" * 80)
        print(report_path.read_text(encoding="utf-8")[:3000])
        print("=" * 80)
        while True:
            raw = input(f"[{report_id}] Your score (1-5, or 's' to skip, 'q' to quit): ").strip().lower()
            if raw == "q":
                save_labels(out_path, labels)
                print(f"Saved {len(labels)} labels to {out_path}.")
                return 0
            if raw == "s":
                break
            if raw in {"1", "2", "3", "4", "5"}:
                labels[report_id] = int(raw)
                save_labels(out_path, labels)
                break
            print("Enter 1-5, 's' to skip, or 'q' to quit.")

    print(f"Labeled {len(labels)} reports. Saved to {out_path}.")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    labels_path = Path(args.labels)
    reports_dir = Path(args.reports_dir)
    labels = load_labels(labels_path)
    if not labels:
        print(f"No labels found at {labels_path}. Run `label` first.")
        return 1

    judge_totals: Dict[str, Optional[float]] = {}
    for report_id in labels:
        report_path = reports_dir / f"{report_id}.md"
        if not report_path.exists():
            continue
        result = score_report(report_path.read_text(encoding="utf-8"), topic=report_id)
        judge_totals[report_id] = result.total  # None on a parse failure -- excluded, not coerced

    agreement = compute_agreement(labels, judge_totals)
    if agreement is None:
        print("Fewer than 2 labeled reports had a parseable judge score -- nothing to compute yet.")
        return 1

    Path(args.out).write_text(json.dumps(agreement, indent=2), encoding="utf-8")
    print(json.dumps(agreement, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Judge calibration: hand-label reports, then measure agreement.")
    sub = parser.add_subparsers(dest="command", required=True)

    label_parser = sub.add_parser("label", help="Blind-label saved reports 1-5.")
    label_parser.add_argument("--reports-dir", default="outputs")
    label_parser.add_argument("--out", default="eval/calibration_labels.json")
    label_parser.set_defaults(func=_cmd_label)

    score_parser = sub.add_parser("score", help="Run the judge over labeled reports and compute agreement.")
    score_parser.add_argument("--labels", default="eval/calibration_labels.json")
    score_parser.add_argument("--reports-dir", default="outputs")
    score_parser.add_argument("--out", default="eval/calibration_results.json")
    score_parser.set_defaults(func=_cmd_score)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    import sys
    sys.exit(main())
