#!/usr/bin/env python3
"""
Cross-run comparison for repeated scenario executions.

Takes N result directories (repeated runs of the same scenario) and, for each
phase that appears in all directories, generates comparison plots showing that
phase's performance across all N runs.

Output structure:
    {output_dir}/{phase}/program_duration.png
    {output_dir}/{phase}/latency_cdf.png

Usage:
    python3 compare.py results/hol-run1 results/hol-run2 results/hol-run3
    python3 compare.py results/hol-run1 results/hol-run2 --output my_comparison/
"""

import argparse
import os
import sys
from typing import Dict, List

# Ensure sibling imports work regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from analyze import (
    discover_phases,
    load_all_results,
    load_results,
    group_latencies_by_program,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUN_COLORS = plt.cm.tab10.colors
RUN_LINESTYLES = ["-", "--", "-.", ":"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_run_labels(dirs: List[str]) -> List[str]:
    """Extract short labels from directory paths; disambiguate if basenames collide."""
    labels = [os.path.basename(d.rstrip("/")) for d in dirs]
    if len(set(labels)) < len(labels):
        labels = [
            os.path.join(
                os.path.basename(os.path.dirname(d.rstrip("/"))),
                os.path.basename(d.rstrip("/")),
            )
            for d in dirs
        ]
    return labels

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_program_duration_comparison(
    phase: str,
    results_dirs: List[str],
    run_labels: List[str],
    out_path: str,
) -> None:
    """Grouped bar chart: program durations for one phase across N runs."""
    run_data: List[Dict[str, float]] = []
    all_programs: List[str] = []

    for results_dir in results_dirs:
        records = load_all_results(os.path.join(results_dir, phase))
        by_program: Dict[str, List[dict]] = {}
        for r in records:
            pid = r.get("program_id", "unknown")
            by_program.setdefault(pid, []).append(r)

        durations: Dict[str, float] = {}
        for pid, recs in by_program.items():
            sent = [r["sent_at"] for r in recs if "sent_at" in r]
            done = [r["completed_at"] for r in recs if "completed_at" in r]
            if sent and done:
                durations[pid] = max(done) - min(sent)
            if pid not in all_programs:
                all_programs.append(pid)
        run_data.append(durations)

    if not all_programs:
        print(f"[compare] No data for phase {phase}, skipping program_duration.png")
        return

    all_programs = sorted(all_programs)
    n_programs = len(all_programs)
    n_runs = len(results_dirs)

    fig, ax = plt.subplots(figsize=(max(8, n_programs * 0.8 + 2), 5))
    x = range(n_programs)
    bar_w = 0.8 / max(n_runs, 1)

    for i, label in enumerate(run_labels):
        vals = [run_data[i].get(pid, 0) for pid in all_programs]
        offsets = [xi - 0.4 + (i + 0.5) * bar_w for xi in x]
        ax.bar(offsets, vals, width=bar_w * 0.9,
               label=label, color=RUN_COLORS[i % len(RUN_COLORS)])

    ax.set_title(f"Program Duration Across Runs \u2014 {phase}", fontsize=11)
    ax.set_xticks(list(x))
    ax.set_xticklabels(all_programs, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Duration (s)")
    ax.legend(fontsize=7)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[compare] Wrote {out_path}")


def plot_latency_cdf_comparison(
    phase: str,
    results_dirs: List[str],
    run_labels: List[str],
    out_path: str,
) -> None:
    """Per-program CDF subplots with one curve per run."""
    # Collect latencies per run per program.
    run_groups: List[Dict[str, List[float]]] = []
    all_pids: set = set()
    for results_dir in results_dirs:
        records = load_results(os.path.join(results_dir, phase))
        groups = group_latencies_by_program(records)
        run_groups.append(groups)
        all_pids.update(groups.keys())

    if not all_pids:
        print(f"[compare] No latency data for phase {phase}, skipping latency_cdf.png")
        return

    all_pids_sorted = sorted(all_pids)
    n_programs = len(all_pids_sorted)

    fig, axes = plt.subplots(
        n_programs, 1,
        figsize=(12, max(4, 4 * n_programs)),
        squeeze=False, sharex=True, sharey=True,
    )

    for p_idx, pid in enumerate(all_pids_sorted):
        ax = axes[p_idx][0]
        for r_idx, label in enumerate(run_labels):
            lats = run_groups[r_idx].get(pid)
            if not lats:
                continue
            s = sorted(lats)
            ys = [(j + 1) / len(s) for j in range(len(s))]
            ax.plot(
                s, ys,
                label=label,
                color=RUN_COLORS[r_idx % len(RUN_COLORS)],
                linestyle=RUN_LINESTYLES[r_idx % len(RUN_LINESTYLES)],
                linewidth=1.2,
            )
        ax.set_title(pid, fontsize=9)
        if p_idx == n_programs - 1:
            ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("CDF")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Latency CDF by Program Across Runs \u2014 {phase}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.82, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] Wrote {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare the same phase across multiple scenario runs",
    )
    parser.add_argument(
        "results_dirs", nargs="+",
        help="Two or more result directories to compare",
    )
    parser.add_argument(
        "--output", default="comparison_plots",
        help="Output directory for comparison plots (default: comparison_plots/)",
    )
    args = parser.parse_args()

    results_dirs = [d.rstrip("/") for d in args.results_dirs]

    if len(results_dirs) < 2:
        print("[compare] Error: need at least 2 result directories to compare.")
        sys.exit(1)

    for d in results_dirs:
        if not os.path.isdir(d):
            print(f"[compare] Error: {d} is not a directory.")
            sys.exit(1)

    run_labels = make_run_labels(results_dirs)

    # Discover phases in each directory and find the common set.
    all_phases = [discover_phases(d) for d in results_dirs]
    phase_sets = [set(p) for p in all_phases]
    common = sorted(set.intersection(*phase_sets))

    if not common:
        print("[compare] No common phases found across all directories:")
        for label, phases in zip(run_labels, all_phases):
            print(f"  {label}: {phases}")
        sys.exit(1)

    skipped = sorted(set.union(*phase_sets) - set(common))
    if skipped:
        print(f"[compare] Skipping phases not present in all runs: {skipped}")

    print(f"[compare] Comparing {len(results_dirs)} runs across {len(common)} phases: {common}")

    output_dir = args.output.rstrip("/")

    for phase in common:
        phase_out = os.path.join(output_dir, phase)
        os.makedirs(phase_out, exist_ok=True)

        plot_program_duration_comparison(
            phase, results_dirs, run_labels,
            os.path.join(phase_out, "program_duration.png"),
        )
        plot_latency_cdf_comparison(
            phase, results_dirs, run_labels,
            os.path.join(phase_out, "latency_cdf.png"),
        )

    print(f"[compare] Done. Plots written to {output_dir}/")


if __name__ == "__main__":
    main()
