"""
Calibration analysis for Phase 1 P1 compaction results.

Reads p1_compaction_results.json (no model calls) and computes per-arm:
  - Brier score (proper scoring rule)
  - ECE (expected calibration error, adaptive bins)
  - Risk-coverage table (accuracy at each confidence threshold)
  - Abstention analysis (confidence==0 cases)

Usage:
    python -m graphrag.eval.calibration_analysis [--results PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
from collections import defaultdict
from typing import Any

RESULTS_DEFAULT = pathlib.Path(
    "graphrag/eval/external/mediq/p1_compaction_results.json"
)
OUT_DEFAULT = pathlib.Path("graphrag/eval/data/calibration_metrics.json")

ARMS = [
    "full-transcript",
    "truncation-headtail",
    "freetext-summary",
    "structured-handoff",
    "handoff-plus-sources",
]

THRESHOLDS = [0.0, 0.5, 0.7, 0.8, 0.9]


def brier_score(pairs: list[tuple[float, int]]) -> float:
    return statistics.mean((c - y) ** 2 for c, y in pairs)


def ece(pairs: list[tuple[float, int]], n_bins: int = 5) -> float:
    """Equal-frequency binned ECE. Skips bins with <2 samples."""
    if not pairs:
        return float("nan")
    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    n = len(sorted_pairs)
    bin_size = max(1, n // n_bins)
    bins: list[list[tuple[float, int]]] = []
    for i in range(0, n, bin_size):
        chunk = sorted_pairs[i : i + bin_size]
        if bins and len(bins[-1]) < bin_size // 2:
            bins[-1].extend(chunk)
        else:
            bins.append(chunk)

    ece_val = 0.0
    for b in bins:
        if len(b) < 2:
            continue
        mean_conf = statistics.mean(c for c, _ in b)
        mean_acc = statistics.mean(y for _, y in b)
        ece_val += len(b) / n * abs(mean_conf - mean_acc)
    return ece_val


def risk_coverage(
    pairs: list[tuple[float, int]], threshold: float
) -> dict[str, Any]:
    answered = [(c, y) for c, y in pairs if c >= threshold]
    if not answered:
        return {"coverage": 0.0, "accuracy": None, "n_answered": 0}
    return {
        "coverage": len(answered) / len(pairs),
        "accuracy": statistics.mean(y for _, y in answered),
        "n_answered": len(answered),
    }


def arm_stats(
    pairs: list[tuple[float, int]]
) -> dict[str, Any]:
    abstain = [(c, y) for c, y in pairs if c == 0.0]
    active = [(c, y) for c, y in pairs if c > 0.0]

    abstain_correct = sum(y for _, y in abstain)
    abstain_wrong = len(abstain) - abstain_correct

    rc: dict[float, dict] = {}
    for t in THRESHOLDS:
        rc[t] = risk_coverage(pairs, t)

    # calibration on active (non-abstained) pairs only
    active_ece = ece(active) if len(active) >= 5 else float("nan")
    active_brier = brier_score(active) if active else float("nan")

    # full Brier (confidence=0 counts as 0-probability prediction)
    full_brier = brier_score(pairs)

    return {
        "n_total": len(pairs),
        "n_abstain": len(abstain),
        "abstain_rate": len(abstain) / len(pairs),
        "abstain_correct": abstain_correct,
        "abstain_wrong": abstain_wrong,
        "n_active": len(active),
        "accuracy_all": statistics.mean(y for _, y in pairs),
        "accuracy_active": statistics.mean(y for _, y in active) if active else None,
        "mean_confidence_active": statistics.mean(c for c, _ in active) if active else None,
        "brier_score_full": full_brier,
        "brier_score_active": active_brier,
        "ece_active": active_ece,
        "risk_coverage": {str(t): v for t, v in rc.items()},
    }


def load_pairs(results: list[dict]) -> dict[str, list[tuple[float, int]]]:
    arm_pairs: dict[str, list] = defaultdict(list)
    for r in results:
        for arm in ARMS:
            diag = r.get("diagnoses", {}).get(arm)
            if diag is None:
                continue
            conf = diag.get("confidence")
            correct = diag.get("correct")
            if conf is None or correct is None:
                continue
            arm_pairs[arm].append((float(conf), int(correct)))
    return dict(arm_pairs)


def print_summary(stats: dict[str, dict]) -> None:
    header = f"{'Arm':<28} {'n':>4} {'Acc':>6} {'Abst%':>6} {'Brier':>6} {'ECE':>6}"
    print(header)
    print("-" * len(header))
    for arm in ARMS:
        s = stats.get(arm)
        if s is None:
            continue
        acc = f"{s['accuracy_all']:.3f}"
        abst = f"{s['abstain_rate']*100:.1f}%"
        brier = f"{s['brier_score_full']:.3f}"
        ece_v = f"{s['ece_active']:.3f}" if not math.isnan(s["ece_active"]) else " nan"
        print(f"{arm:<28} {s['n_total']:>4} {acc:>6} {abst:>6} {brier:>6} {ece_v:>6}")

    print()
    print("Risk-coverage table (accuracy at confidence threshold, all arms):")
    hdr = f"{'Threshold':<12}" + "".join(f"  {a[:8]:>10}" for a in ARMS)
    print(hdr)
    print("-" * len(hdr))
    for t in THRESHOLDS:
        row = f"{t:<12.1f}"
        for arm in ARMS:
            s = stats.get(arm)
            if s is None:
                row += f"  {'—':>10}"
                continue
            rc = s["risk_coverage"][str(float(t))]
            acc = rc["accuracy"]
            cov = rc["coverage"]
            cell = f"{acc:.3f}({cov:.0%})" if acc is not None else "—"
            row += f"  {cell:>10}"
        print(row)

    print()
    print("Calibration on active (confidence>0) cases:")
    hdr2 = f"{'Arm':<28} {'n_active':>8} {'mean_conf':>9} {'Acc_active':>10} {'Brier':>6} {'ECE':>6}"
    print(hdr2)
    print("-" * len(hdr2))
    for arm in ARMS:
        s = stats.get(arm)
        if s is None:
            continue
        mc = f"{s['mean_confidence_active']:.3f}" if s["mean_confidence_active"] else "—"
        aa = f"{s['accuracy_active']:.3f}" if s["accuracy_active"] else "—"
        brier = f"{s['brier_score_active']:.3f}" if not math.isnan(s["brier_score_active"]) else " nan"
        ece_v = f"{s['ece_active']:.3f}" if not math.isnan(s["ece_active"]) else " nan"
        print(f"{arm:<28} {s['n_active']:>8} {mc:>9} {aa:>10} {brier:>6} {ece_v:>6}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=pathlib.Path, default=RESULTS_DEFAULT)
    parser.add_argument("--out", type=pathlib.Path, default=OUT_DEFAULT)
    args = parser.parse_args()

    data = json.loads(args.results.read_text())
    pairs = load_pairs(data["results"])

    stats: dict[str, dict] = {}
    for arm in ARMS:
        if arm in pairs:
            stats[arm] = arm_stats(pairs[arm])

    print_summary(stats)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "source": str(args.results),
        "dataset_fingerprint": data.get("dataset_fingerprint"),
        "n_cases": len(data["results"]),
        "arms": stats,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nMetrics saved → {args.out}")


if __name__ == "__main__":
    main()
