"""
Plot journal_summary.csv: Journal name, Impact factor, Paper count.
Visualization styles:
  dual       - two horizontal bar charts (paper count + IF) [default]
  colormap   - single bar chart (paper count) with bars colored by IF + colorbar
  scatter    - scatter plot: IF vs paper count (bubble size = paper count), journal labels
"""
import argparse
import csv
import os

try:
    import matplotlib.pyplot as plt
    import matplotlib
    import matplotlib.colors as mcolors
    matplotlib.use("Agg")
except ImportError:
    raise SystemExit("matplotlib is required. Install with: pip install matplotlib")

INPUT_CSV = "journal_summary.csv"
OUTPUT_IMAGE = "journal_summary.png"
DEFAULT_TOP = 126  # plot top N journals by impact factor (0 = all)
STYLES = ("dual", "colormap", "scatter")


def load_summary(csv_path: str):
    """Load CSV; return (journals, impact_factors, paper_counts)."""
    journals, impact_factors, paper_counts = [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("Journal name") or row.get("journal name") or "").strip()
            if not name:
                continue
            try:
                if_val = float(str(row.get("Impact factor", 0) or row.get("impact factor", 0)).strip().replace(",", "."))
            except (ValueError, TypeError):
                if_val = 0.0
            try:
                count = int(float(str(row.get("Paper count", 0) or row.get("paper count", 0)).strip()))
            except (ValueError, TypeError):
                count = 0
            journals.append(name)
            impact_factors.append(if_val)
            paper_counts.append(count)
    return journals, impact_factors, paper_counts


def main():
    parser = argparse.ArgumentParser(
        description="Plot journal summary CSV: Journal name, Impact factor, Paper count."
    )
    parser.add_argument(
        "-i", "--input",
        default=INPUT_CSV,
        help=f"Input CSV path (default: {INPUT_CSV})",
    )
    parser.add_argument(
        "-o", "--output",
        default=OUTPUT_IMAGE,
        help=f"Output image path (default: {OUTPUT_IMAGE})",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP,
        help=f"Plot only top N journals by impact factor (0 = all, default: {DEFAULT_TOP})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Image DPI (default: 150)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive plot window (in addition to saving).",
    )
    parser.add_argument(
        "--style",
        choices=STYLES,
        default="dual",
        help="Visualization style: dual (two bar charts), colormap (bars colored by IF), scatter (IF vs count)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"Input file not found: {args.input}")

    journals, impact_factors, paper_counts = load_summary(args.input)
    if not journals:
        raise SystemExit("No rows found in CSV.")

    n = len(journals)
    if args.top and args.top > 0 and n > args.top:
        journals = journals[: args.top]
        impact_factors = impact_factors[: args.top]
        paper_counts = paper_counts[: args.top]
        n = len(journals)

    y_pos = list(range(n))
    y_labels = [j if len(j) <= 40 else j[:37] + "..." for j in journals]

    if args.style == "dual":
        _plot_dual(y_pos, y_labels, paper_counts, impact_factors, args.output, args.dpi)
    elif args.style == "colormap":
        _plot_colormap(y_pos, y_labels, paper_counts, impact_factors, args.output, args.dpi)
    else:
        _plot_scatter(journals, paper_counts, impact_factors, args.output, args.dpi)

    print(f"Saved: {args.output}")
    if args.show:
        plt.show()
    plt.close()


def _plot_dual(y_pos, y_labels, paper_counts, impact_factors, output_path, dpi):
    """Two horizontal bar charts side by side."""
    n = len(y_pos)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(6, n * 0.28)), sharey=True)
    ax1.barh(y_pos, paper_counts, color="steelblue", alpha=0.85, edgecolor="navy", linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(y_labels, fontsize=9)
    ax1.set_xlabel("Paper count", fontsize=11)
    ax1.set_title("Paper count by journal", fontsize=12)
    ax1.invert_yaxis()
    ax1.grid(axis="x", alpha=0.3)
    ax2.barh(y_pos, impact_factors, color="darkorange", alpha=0.85, edgecolor="darkred", linewidth=0.5)
    ax2.set_xlabel("Impact factor", fontsize=11)
    ax2.set_title("Impact factor by journal", fontsize=12)
    ax2.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")


def _plot_colormap(y_pos, y_labels, paper_counts, impact_factors, output_path, dpi):
    """Single bar chart: paper count with bars colored by impact factor + colorbar."""
    n = len(y_pos)
    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.28)))
    if_max = max(impact_factors) or 1.0
    norm = mcolors.Normalize(vmin=0, vmax=if_max)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(norm(if_val)) for if_val in impact_factors]
    ax.barh(y_pos, paper_counts, color=colors, edgecolor="gray", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Paper count", fontsize=11)
    ax.set_ylabel("Journal", fontsize=11)
    ax.set_title("Paper count by journal (color = impact factor)", fontsize=12)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label("Impact factor", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")


def _plot_scatter(journals, paper_counts, impact_factors, output_path, dpi):
    """Scatter: impact factor vs paper count; point size = paper count; selected journals labeled."""
    fig, ax = plt.subplots(figsize=(10, 7))
    # Bubble size proportional to paper count (scale for visibility)
    sizes = [max(30, c * 3) for c in paper_counts]
    scatter = ax.scatter(paper_counts, impact_factors, s=sizes, alpha=0.6, c=impact_factors, cmap="plasma", edgecolors="gray", linewidths=0.5)
    ax.set_xlabel("Paper count", fontsize=11)
    ax.set_ylabel("Impact factor", fontsize=11)
    ax.set_title("Journals: impact factor vs paper count (size ∝ papers)", fontsize=12)
    ax.grid(True, alpha=0.3)
    # Label only journals with IF > 25 to avoid clutter
    label_indices = [i for i, if_val in enumerate(impact_factors) if if_val > 25]
    # Keep deterministic ordering: highest IF first, then by paper count
    label_indices.sort(key=lambda i: (-impact_factors[i], -paper_counts[i]))
    for idx in label_indices:
        name = journals[idx] if len(journals[idx]) <= 35 else journals[idx][:32] + "..."
        # Place label to the right of the point with a fixed gap from the marker edge
        # s is area in points^2; marker radius in points ~ sqrt(s)/2
        radius_pts = (sizes[idx] ** 0.5) / 2.0
        gap_pts = 4.0  # fixed distance in points between marker edge and label
        ax.annotate(
            name,
            (paper_counts[idx], impact_factors[idx]),
            fontsize=6,  # smaller label size
            alpha=0.9,
            xytext=(radius_pts + gap_pts, 0),
            textcoords="offset points",
            ha="left",
            va="center",
        )
    plt.colorbar(scatter, ax=ax, label="Impact factor")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")


if __name__ == "__main__":
    main()
