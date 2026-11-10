"""
Paired significance test of LNR vs. SMOTE on the KEEL results of src/binary_keel.py.

For every dataset x metric x classifier, the 50 per-round scores of LNR and SMOTE are paired by
round (both methods use the same train/test split in a round) and compared with a two-sided
Wilcoxon signed-rank test (normal approximation with tie correction, rounds with zero
difference dropped, no continuity correction; no correction for multiple testing).

Run from the project root; --run_dir (the folder with rounds.csv) is required, e.g.
    python src/keel_Wilcoxon_test.py --run_dir results/keel_<time>

Writes to the run folder:
    significance_lnr_vs_smote.csv   one row per dataset x metric x classifier
    significance_summary.txt        counts of significant differences, overall and per metric
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

METRICS = ['F1-score', 'G-mean', 'ROC-AUC']


def wilcoxon_signed_rank(x: list[float], y: list[float]) -> tuple[int, float, float, float]:
    """
    Two-sided Wilcoxon signed-rank test for paired samples x and y.

    Returns (n, W_plus, z, p): n = number of non-zero differences, W_plus = sum of the ranks of the
    positive differences (x > y), z = standardised W_plus, p = two-sided p-value from the normal
    approximation. Zero differences are dropped; tied absolute differences get average ranks and
    the variance is corrected for ties.
    """
    d = [a - b for a, b in zip(x, y) if a != b]
    n = len(d)
    if n == 0:
        return 0, 0.0, 0.0, 1.0

    # Average ranks of |d| (ties share the mean of their ranks)
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    tie_term = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        t = j - i + 1
        tie_term += t ** 3 - t
        i = j + 1

    w_plus = sum(r for r, v in zip(ranks, d) if v > 0)
    mean = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - tie_term / 48
    if var <= 0:
        return n, w_plus, 0.0, 1.0
    z = (w_plus - mean) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2))  # two-sided: 2 * (1 - Phi(|z|))
    return n, w_plus, z, p


def main() -> None:
    parser = argparse.ArgumentParser(description="Wilcoxon signed-rank test of LNR vs. SMOTE on KEEL results.")
    parser.add_argument('--run_dir', type=Path, required=True,
                        help="Folder with rounds.csv from src/binary_keel.py (required), e.g. results/keel_<time>.")
    parser.add_argument('--alpha', type=float, default=0.05, help="Significance level (default: 0.05).")
    args = parser.parse_args()

    # scores[(dataset, metric, classifier)][method][round] = score
    scores = defaultdict(lambda: defaultdict(dict))
    with open(args.run_dir / "rounds.csv", newline='') as f:
        for row in csv.DictReader(f):
            for metric in METRICS:
                scores[(row['dataset'], metric, row['classifier'])][row['method']][int(row['round'])] = float(row[metric])

    out_rows = []
    for (dataset, metric, clf), by_method in sorted(scores.items()):
        rounds = sorted(set(by_method['LNR']) & set(by_method['SMOTE']))
        lnr = [by_method['LNR'][r] for r in rounds]
        smote = [by_method['SMOTE'][r] for r in rounds]
        n, w_plus, z, p = wilcoxon_signed_rank(lnr, smote)
        mean_diff = sum(a - b for a, b in zip(lnr, smote)) / len(rounds)
        if p < args.alpha:
            result = 'LNR better' if z > 0 else 'SMOTE better'
        else:
            result = 'not significant'
        out_rows.append({
            'dataset': dataset, 'metric': metric, 'classifier': clf, 'rounds': len(rounds),
            'n_nonzero': n, 'mean_diff_lnr_minus_smote': f"{100 * mean_diff:.4f}",
            'W_plus': f"{w_plus:.1f}", 'z': f"{z:.4f}", 'p_value': f"{p:.6f}", 'result': result,
        })

    out_csv = args.run_dir / "significance_lnr_vs_smote.csv"
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)

    lines = [f"Two-sided Wilcoxon signed-rank test, LNR vs. SMOTE, paired over rounds, alpha = {args.alpha}",
             f"(normal approximation with tie correction, zero differences dropped, no correction for multiple testing)",
             f"Run folder: {args.run_dir.name}", ""]
    for metric in METRICS + ['All']:
        sel = [r for r in out_rows if metric == 'All' or r['metric'] == metric]
        count = {k: sum(r['result'] == k for r in sel) for k in ('LNR better', 'SMOTE better', 'not significant')}
        lines.append(f"{metric:8s}: LNR better {count['LNR better']}/{len(sel)}, "
                     f"SMOTE better {count['SMOTE better']}/{len(sel)}, "
                     f"not significant {count['not significant']}/{len(sel)}")
    summary = "\n".join(lines) + "\n"
    (args.run_dir / "significance_summary.txt").write_text(summary)
    print(summary)
    print(f"Saved {out_csv} and {args.run_dir / 'significance_summary.txt'}")


if __name__ == "__main__":
    main()
