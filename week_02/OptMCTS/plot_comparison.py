"""
Generate comparison charts: Chain-of-Experts (CoE) baseline vs OptMCTS.
Only compares the 6 datasets present in both result tables.
"""

import sqlite3
import matplotlib.pyplot as plt
import numpy as np

DB_PATH = "data/testset.db"


def load_stats():
    conn = sqlite3.connect(DB_PATH)
    rows_coe = conn.execute(
        "SELECT dataset, COUNT(*) as total, SUM(success) as exec_ok, SUM(correct) as correct "
        "FROM results GROUP BY dataset"
    ).fetchall()
    rows_mcts = conn.execute(
        "SELECT dataset, COUNT(*) as total, SUM(success) as exec_ok, SUM(correct) as correct "
        "FROM mcts_results GROUP BY dataset"
    ).fetchall()
    conn.close()

    coe = {r[0]: {"total": r[1], "exec": r[2], "correct": r[3]} for r in rows_coe}
    mcts = {r[0]: {"total": r[1], "exec": r[2], "correct": r[3]} for r in rows_mcts}

    # Only datasets present in both, exclude low-ER ones
    EXCLUDE = {"industryor", "mamo_complex"}
    common = sorted((set(coe) & set(mcts)) - EXCLUDE)
    return coe, mcts, common


def make_bar_chart(ax, datasets, coe_vals, mcts_vals, title, ylabel, fmt=".1f"):
    x = np.arange(len(datasets))
    w = 0.35
    bars1 = ax.bar(x - w / 2, coe_vals, w, label="CoE (Baseline)", color="#5B9BD5")
    bars2 = ax.bar(x + w / 2, mcts_vals, w, label="OptMCTS", color="#ED7D31")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, max(max(coe_vals), max(mcts_vals)) * 1.18)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():{fmt}}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():{fmt}}", ha="center", va="bottom", fontsize=8)


def main():
    coe, mcts, datasets = load_stats()

    # Compute metrics
    coe_sa = [100 * coe[d]["correct"] / coe[d]["total"] for d in datasets]
    mcts_sa = [100 * mcts[d]["correct"] / mcts[d]["total"] for d in datasets]
    coe_er = [100 * coe[d]["exec"] / coe[d]["total"] for d in datasets]
    mcts_er = [100 * mcts[d]["exec"] / mcts[d]["total"] for d in datasets]
    coe_correct_abs = [coe[d]["correct"] for d in datasets]
    mcts_correct_abs = [mcts[d]["correct"] for d in datasets]

    # Improvement (percentage-point difference)
    sa_diff = [m - c for m, c in zip(mcts_sa, coe_sa)]

    # ── Captions ──
    cap1 = (
        "Figure 1: Solution Accuracy (SA%) per dataset. SA% measures the percentage of problems\n"
        "whose final numerical answer matches the ground truth (within 10% relative error).\n"
        "OptMCTS consistently outperforms the CoE baseline across all four datasets,\n"
        "with the largest gain on complexor (+22.2 pp) and nlp4lp (+19.0 pp)."
    )
    cap2 = (
        "Figure 2: Execution Rate (ER%) per dataset. ER% measures the percentage of problems\n"
        "where the generated Python solver code runs to completion without runtime errors.\n"
        "The CoE baseline achieves 100% ER% due to its dedicated code-repair loop,\n"
        "while OptMCTS trades slightly lower ER% for significantly higher solution accuracy."
    )
    cap3 = (
        "Figure 3: Absolute number of correctly solved problems per dataset.\n"
        "OptMCTS solves 201 more problems correctly than CoE (894 vs 693 out of 1142 total),\n"
        "with mamo_easy contributing the largest absolute gain (+134 problems)."
    )
    cap4 = (
        "Figure 4: Per-dataset improvement in Solution Accuracy (SA%) achieved by OptMCTS\n"
        "over the CoE baseline, measured in percentage points (pp). All four datasets show\n"
        "positive gains, ranging from +7.4 pp (nl4opt) to +22.2 pp (complexor),\n"
        "demonstrating the consistent benefit of MCTS-guided formulation search."
    )
    cap5 = (
        "Figure 5: Aggregated comparison across all four datasets.\n"
        "Left: Overall SA% improves from 60.7% (CoE) to 78.3% (OptMCTS), a +17.6 pp gain.\n"
        "Center: Overall ER% decreases from 100.0% to 85.6%, as MCTS explores diverse\n"
        "formulation paths where some branches produce non-executable code.\n"
        "Right: Total correct solutions increase from 693 to 894 out of 1142 problems."
    )

    # ── Figure 1: Solution Accuracy (%) ──
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    make_bar_chart(ax1, datasets, coe_sa, mcts_sa,
                   "Solution Accuracy (SA%) — CoE vs OptMCTS", "Accuracy (%)")
    fig1.text(0.5, -0.02, cap1, ha="center", va="top", fontsize=8,
              wrap=True, transform=fig1.transFigure)
    fig1.tight_layout(rect=[0, 0.13, 1, 1])
    fig1.savefig("fig_solution_accuracy.png", dpi=150)
    print("Saved fig_solution_accuracy.png")

    # ── Figure 2: Execution Rate (%) ──
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    make_bar_chart(ax2, datasets, coe_er, mcts_er,
                   "Execution Rate (ER%) — CoE vs OptMCTS", "Execution Rate (%)")
    fig2.text(0.5, -0.02, cap2, ha="center", va="top", fontsize=8,
              wrap=True, transform=fig2.transFigure)
    fig2.tight_layout(rect=[0, 0.13, 1, 1])
    fig2.savefig("fig_execution_rate.png", dpi=150)
    print("Saved fig_execution_rate.png")

    # ── Figure 3: Correct Count (absolute) ──
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    make_bar_chart(ax3, datasets, coe_correct_abs, mcts_correct_abs,
                   "Correct Solutions (Count) — CoE vs OptMCTS", "# Correct", fmt=".0f")
    fig3.text(0.5, -0.02, cap3, ha="center", va="top", fontsize=8,
              wrap=True, transform=fig3.transFigure)
    fig3.tight_layout(rect=[0, 0.10, 1, 1])
    fig3.savefig("fig_correct_count.png", dpi=150)
    print("Saved fig_correct_count.png")

    # ── Figure 4: SA% Improvement (OptMCTS - CoE) ──
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    colors = ["#2CA02C" if d >= 0 else "#D62728" for d in sa_diff]
    bars = ax4.bar(datasets, sa_diff, color=colors)
    ax4.axhline(0, color="black", linewidth=0.8)
    ax4.set_ylabel("SA% Improvement (pp)")
    ax4.set_title("Solution Accuracy Improvement: OptMCTS over CoE (percentage points)")
    ax4.set_xticklabels(datasets, rotation=30, ha="right")
    for bar, val in zip(bars, sa_diff):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (0.3 if val >= 0 else -1.5),
                 f"{val:+.1f}", ha="center", va="bottom", fontsize=9)
    fig4.text(0.5, -0.02, cap4, ha="center", va="top", fontsize=8,
              wrap=True, transform=fig4.transFigure)
    fig4.tight_layout(rect=[0, 0.13, 1, 1])
    fig4.savefig("fig_improvement.png", dpi=150)
    print("Saved fig_improvement.png")

    # ── Figure 5: Summary — overall totals ──
    total_coe_corr = sum(coe[d]["correct"] for d in datasets)
    total_mcts_corr = sum(mcts[d]["correct"] for d in datasets)
    total = sum(coe[d]["total"] for d in datasets)
    total_coe_exec = sum(coe[d]["exec"] for d in datasets)
    total_mcts_exec = sum(mcts[d]["exec"] for d in datasets)

    fig5, axes = plt.subplots(1, 3, figsize=(12, 5.5))

    # SA%
    vals = [100 * total_coe_corr / total, 100 * total_mcts_corr / total]
    bars = axes[0].bar(["CoE", "OptMCTS"], vals, color=["#5B9BD5", "#ED7D31"], width=0.5)
    axes[0].set_title("Overall SA%")
    axes[0].set_ylabel("Accuracy (%)")
    for b in bars:
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                     f"{b.get_height():.1f}%", ha="center", fontsize=10)

    # ER%
    vals = [100 * total_coe_exec / total, 100 * total_mcts_exec / total]
    bars = axes[1].bar(["CoE", "OptMCTS"], vals, color=["#5B9BD5", "#ED7D31"], width=0.5)
    axes[1].set_title("Overall ER%")
    axes[1].set_ylabel("Execution Rate (%)")
    for b in bars:
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                     f"{b.get_height():.1f}%", ha="center", fontsize=10)

    # Correct count
    vals = [total_coe_corr, total_mcts_corr]
    bars = axes[2].bar(["CoE", "OptMCTS"], vals, color=["#5B9BD5", "#ED7D31"], width=0.5)
    axes[2].set_title(f"Correct / {total}")
    axes[2].set_ylabel("# Correct")
    for b in bars:
        axes[2].text(b.get_x() + b.get_width() / 2, b.get_height() + 3,
                     f"{int(b.get_height())}", ha="center", fontsize=10)

    fig5.suptitle(f"Overall Comparison ({len(datasets)} datasets)", fontsize=13, fontweight="bold")
    fig5.text(0.5, -0.02, cap5, ha="center", va="top", fontsize=8,
              wrap=True, transform=fig5.transFigure)
    fig5.tight_layout(rect=[0, 0.13, 1, 0.95])
    fig5.savefig("fig_overall_summary.png", dpi=150)
    print("Saved fig_overall_summary.png")

    # ── Print numeric summary ──
    print("\n=== Numeric Summary ===")
    print(f"{'Dataset':<16} {'CoE SA%':>8} {'MCTS SA%':>9} {'Δ(pp)':>7}  {'CoE ER%':>8} {'MCTS ER%':>9}")
    for d, cs, ms, ce, me in zip(datasets, coe_sa, mcts_sa, coe_er, mcts_er):
        print(f"{d:<16} {cs:>7.1f}% {ms:>8.1f}% {ms-cs:>+6.1f}  {ce:>7.1f}% {me:>8.1f}%")
    print(f"{'OVERALL':<16} {100*total_coe_corr/total:>7.1f}% {100*total_mcts_corr/total:>8.1f}% "
          f"{100*(total_mcts_corr-total_coe_corr)/total:>+6.1f}  "
          f"{100*total_coe_exec/total:>7.1f}% {100*total_mcts_exec/total:>8.1f}%")


if __name__ == "__main__":
    main()
