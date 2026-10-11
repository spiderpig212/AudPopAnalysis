"""
Near-diagonal homogeneity check for the pairwise-decoder RSA matrices.

final_figures_dprime.py's synthetic-sounds figure collapses each session's
near-diagonal (adjacent-stimulus) RSA entries down to a single mean per
session, on the assumption that near-diagonal entries are exchangeable
enough to pool. This script checks that assumption directly: for a single
RSA-matrix "cell" (one brain region / region-pair, one response range, one
stimulus type), it plots each near-diagonal stimulus pair's value as its own
box across sessions, and runs a statistical test across the near-diagonal
pairs to see whether their distributions actually differ.

Test choice -- Friedman rather than a plain one-way ANOVA or a
repeated-measures ANOVA:
  - Every session contributes exactly one point to every near-diagonal pair
    (each pair is read off that session's own complete matrix), so the pairs
    are repeated measures on the same sessions, not independent groups. A
    standard one-way ANOVA assumes independent groups and would be the wrong
    test here.
  - Decoding accuracy is bounded in [0, 1] and can pile up near ceiling, so
    the normality/sphericity assumptions a repeated-measures ANOVA needs
    aren't safe to take for granted. Friedman only ranks within each
    session's own row, so it is the more conservative choice: it won't
    manufacture significance out of a violated distributional assumption.

Restricted to AM and pureTones, the two stimulus types for which
final_figures_dprime.py actually uses the near-diagonal average (naturalSound
stimulus indices are grouped by source, so "neighboring stimulus" has no
equivalent meaning there).
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend.
import matplotlib.pyplot as plt
from scipy import stats

from analysis_class import FiringRateAnalysis
from d_prime_RSA_visualization import (
    load_rsa_data,
    raw_rows,
    cca_target_rows,
    STRIP_ALPHA_THRESHOLD,
)

response_ranges = ["onset", "sustained", "offset"]
NEARDIAG_STIM_TYPES = ("AM", "pureTones")

METRIC_TAGS = {"accuracy_matrix": "accuracy", "J_matrix": "J",
              "dprime_matrix": "dprime"}
METRIC_YLABELS = {"accuracy_matrix": "Decoding accuracy",
                  "J_matrix": "Fisher's J", "dprime_matrix": "d-prime"}


def collect_neardiag_by_session(records, data_source, row_value, row_field,
                                response_range, metric_key):
    """
    Build the (n_sessions, n_pairs) matrix of near-diagonal values for one
    RSA-matrix cell: one row per session, one column per near-diagonal
    (adjacent-stimulus) pair.

    This is a complete block design by construction: a session either has a
    full matrix for this cell or none at all, so every session included has
    a value for every pair (required for the Friedman test below).

    Returns (session_ids, pair_matrix). pair_matrix has shape (0, 0) if no
    sessions matched.
    """
    session_ids, rows = [], []
    for r in records:
        if (r["data_source"] == data_source
                and r["response_range"] == response_range
                and r[row_field] == row_value
                and r[metric_key] is not None):
            mat = r[metric_key]
            idx = np.arange(mat.shape[0] - 1)
            rows.append(mat[idx, idx + 1])
            session_ids.append(r["session"])

    if len(rows) == 0:
        return [], np.zeros((0, 0))
    return session_ids, np.array(rows)  # (n_sessions, n_pairs)


def plot_and_test_neardiag(pair_matrix, pair_labels, title, ylabel, out_path):
    """
    Box + strip plot across near-diagonal pairs (x-axis), one point per
    session; each session's own points are connected by a thin line so the
    matched (repeated-measures) structure the Friedman test relies on is
    visible, not just implied by the statistics.

    Returns the Friedman test result dict, or None if it couldn't be run
    (too few pairs or sessions).
    """
    n_sessions, n_pairs = pair_matrix.shape
    fig, ax = plt.subplots(figsize=(2 + 1.1 * max(n_pairs, 1), 5))

    if n_pairs == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return None

    positions = np.arange(n_pairs)
    box_data = [pair_matrix[:, k] for k in range(n_pairs)]
    bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                    showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4C72B0")
        patch.set_alpha(0.5)
    for median in bp["medians"]:
        median.set_color("black")

    if n_sessions <= STRIP_ALPHA_THRESHOLD:
        jitter = (np.random.rand(n_sessions) - 0.5) * 0.15
        for s in range(n_sessions):
            ax.plot(positions + jitter[s], pair_matrix[s, :],
                   color="gray", alpha=0.3, linewidth=0.8, zorder=2)
            ax.scatter(positions + jitter[s], pair_matrix[s, :],
                      color="black", s=10, alpha=0.4, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(pair_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Near-diagonal stimulus pair")

    result = None
    if n_pairs >= 3 and n_sessions >= 2:
        stat, p_val = stats.friedmanchisquare(*box_data)
        result = {"statistic": float(stat), "p_value": float(p_val),
                 "n_sessions": n_sessions, "n_pairs": n_pairs}
        subtitle = (f"Friedman χ²({n_pairs - 1}, n={n_sessions}) = "
                   f"{stat:.2f}, p = {p_val:.3g}")
    else:
        subtitle = (f"n_sessions={n_sessions}, n_pairs={n_pairs} "
                   "(too few for Friedman)")

    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return result


def make_neardiag_consistency_figures(stimulus, data_source, records,
                                      uniq_stims, out_dir, metric_key):
    """One figure per (brain region / region-pair, response range)."""
    if data_source == "raw":
        rows, row_field = raw_rows, "brain_region"
        row_label = lambda r: r
    else:
        rows, row_field = cca_target_rows, "target_region"
        row_label = lambda r: f"Primary -> {r.replace(' auditory area', '')}"

    pair_labels = [f"{uniq_stims[i]}-{uniq_stims[i + 1]}"
                  for i in range(len(uniq_stims) - 1)]
    ylabel = METRIC_YLABELS[metric_key]
    metric_tag = METRIC_TAGS[metric_key]

    os.makedirs(out_dir, exist_ok=True)
    summary_rows = []
    for row_value in rows:
        for response_range in response_ranges:
            session_ids, pair_matrix = collect_neardiag_by_session(
                records, data_source, row_value, row_field, response_range,
                metric_key)

            # Raw per-session values alongside the figure, for traceability.
            if pair_matrix.size > 0:
                raw_df = pd.DataFrame(pair_matrix, index=session_ids,
                                      columns=pair_labels)
                raw_df.index.name = "session"
                raw_df.to_csv(
                    f"{out_dir}/neardiag_{stimulus}_{data_source}_"
                    f"{row_value.replace(' ', '')}_{response_range}_"
                    f"{metric_tag}_raw.csv")

            title = (f"{stimulus} — {data_source.upper()} — "
                    f"{row_label(row_value)} — {response_range}")
            out_path = (f"{out_dir}/neardiag_{stimulus}_{data_source}_"
                       f"{row_value.replace(' ', '')}_{response_range}_"
                       f"{metric_tag}.png")
            result = plot_and_test_neardiag(
                pair_matrix, pair_labels, title, ylabel, out_path)
            if result is not None:
                summary_rows.append({
                    "stimulus": stimulus, "data_source": data_source,
                    "row_value": row_value, "response_range": response_range,
                    **result,
                })

    return summary_rows


def _add_fdr_correction(summary_df):
    """
    Benjamini-Hochberg FDR correction across every panel tested. Many
    Friedman tests are run here (one per region/response-range cell), so
    raw p-values alone would overstate how many panels show real
    near-diagonal heterogeneity.
    """
    df = summary_df.sort_values("p_value").reset_index(drop=True)
    m = len(df)
    ranks = np.arange(1, m + 1)
    adj = np.minimum.accumulate((df["p_value"].values * m / ranks)[::-1])[::-1]
    df["p_value_fdr_bh"] = np.clip(adj, 0, 1)
    return df


def main(metric_key="accuracy_matrix"):
    fr_db = FiringRateAnalysis(db_suffix="coords_updated")
    file_path = fr_db.figdata_path
    rsa_dir = f"{file_path}/dprime_RSA"
    out_dir = f"{rsa_dir}/figures/neardiag_consistency"

    all_summary = []
    for stimulus in NEARDIAG_STIM_TYPES:
        records = load_rsa_data(rsa_dir, stimulus)
        if len(records) == 0:
            print(f"No records for {stimulus}, skipping.")
            continue

        stim_arrays = fr_db.return_arrays(stimulus)
        uniq_stims = np.unique(stim_arrays["stimArray"][0, :])

        for data_source in ("raw", "cca"):
            all_summary.extend(make_neardiag_consistency_figures(
                stimulus, data_source, records, uniq_stims, out_dir,
                metric_key))

    if all_summary:
        summary_df = _add_fdr_correction(pd.DataFrame(all_summary))
        metric_tag = METRIC_TAGS[metric_key]
        summary_path = f"{out_dir}/neardiag_friedman_summary_{metric_tag}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved Friedman summary to {summary_path}")


if __name__ == "__main__":
    main(metric_key="accuracy_matrix")
    # main(metric_key="J_matrix")
    # main(metric_key="dprime_matrix")
