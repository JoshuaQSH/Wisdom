# wisdom/utils/visulization.py
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from captum.attr import visualization as viz
import numpy as np
import torch
import pandas as pd
from pathlib import Path

# cmap = ['PuBuGn', 'Greens', 'Purples', 'Reds', 'Blues', 'YlGn', 'summer', 'cool', 'bwr']
def viz_attr_diff(img_orig, img_pert, cmap="bwr", alpha=0.8, tag='random'):    
    
    # tensors -> HWC numpy
    o = img_orig.cpu().detach().numpy().transpose(1,2,0)
    p = img_pert.cpu().detach().numpy().transpose(1,2,0)

    d = np.abs(p - o)

    fig, ax = plt.subplots(1, 3, figsize=(9,3))
    ax[0].imshow(o.squeeze() if o.shape[2]==1 else o)
    ax[0].set_title("Original")
    ax[1].imshow(p.squeeze() if p.shape[2]==1 else p)
    ax[1].set_title(f"Perturbed ({tag})")
    ax[2].imshow(o.squeeze() if o.shape[2]==1 else o, alpha=1-alpha)
    ax[2].imshow(d.squeeze() if d.shape[2]==1 else d,
                 cmap=cmap, alpha=alpha)
    ax[2].set_title(f"Overlay ({tag})")
    for a in ax: a.axis("off")

    fig.tight_layout()
    out = f"feature_importance_{tag}_diff.pdf"
    fig.savefig(out, dpi=1200, format='pdf', bbox_inches="tight")
    plt.close(fig)

def viz_attr(img, attr, dataset_name, model_name, with_original=False):
    attr_np = attr.cpu().detach().numpy()
    img_np  = img.cpu().detach().numpy()
    
    if with_original:
        fig, ax = viz.visualize_image_attr_multiple(np.transpose(attr_np, (1, 2, 0)),
                                        np.transpose(img_np, (1, 2, 0)),
                                        ["original_image", "heat_map"],
                                        ["all", "positive"],
                                        show_colorbar=True,
                                        outlier_perc=2,
                                        use_pyplot=False)
    else:
        fig, ax = viz.visualize_image_attr(np.transpose(attr_np, (1, 2, 0)),
                                        np.transpose(img_np, (1, 2, 0)),
                                        "heat_map",
                                        "positive",
                                        show_colorbar=True,
                                        outlier_perc=2,
                                        use_pyplot=False)

    fig.tight_layout()
    fig.savefig(f"{dataset_name}_{model_name}.png",
                dpi=300,
                bbox_inches="tight")
    plt.close(fig)

def viz_topk_neurons_score(csv_file, top_k=10):
    df = pd.read_csv(csv_file)
    df_sorted = df.sort_values(by='Score', ascending=False).head(top_k)

    plt.figure(figsize=(10, 6))
    for layer_name, group in df_sorted.groupby('LayerName'):
        plt.scatter(group['NeuronIndex'], group['Score'], label=layer_name, alpha=0.7, s=70)

    plt.xlabel('Neuron Index')
    plt.ylabel('Importance Score')
    plt.title(f'Top-{top_k} Neuron Scores Across All Layers')
    plt.legend()
    plt.savefig(f'top_{top_k}_neuron_scores.pdf', format='pdf', dpi=1200)
    # plt.show()


# color_choice = ['PuBuGn', 'Greens', 'Purples', 'Reds', 'Blues', 'YlGn', 'summer', 'cool', 'bwr']
RQ1_METHOD_PALETTE = ['#eadcff', '#d0beed', '#b6a1db', '#8269b8', '#7f66b5', '#5a4491']
RQ1_METHOD_PRIORITY = {
    "Random": 0,
    "GradShap": 1,
    "IntegGrad": 2,
    "GradXAct": 3,
}


def _rq1_order_methods(methods):
    unique_methods = list(dict.fromkeys(methods))
    wisdom = [m for m in unique_methods if m == "Wisdom"]
    others = [m for m in unique_methods if m != "Wisdom"]
    others.sort(key=lambda m: (RQ1_METHOD_PRIORITY.get(m, 50), m))
    return others + wisdom


def _rq1_method_colors(methods):
    ordered = _rq1_order_methods(methods)
    colors = {}
    lighter = RQ1_METHOD_PALETTE[:-1]
    non_wisdom = [m for m in ordered if m != "Wisdom"]
    for idx, method in enumerate(non_wisdom):
        colors[method] = lighter[min(idx, len(lighter) - 1)]
    if "Wisdom" in ordered:
        colors["Wisdom"] = RQ1_METHOD_PALETTE[-1]
    return ordered, colors


def _rq1_available_metrics(df, requested_metrics=None):
    all_metrics = {
        "Confidence Drop": "Confidence Drop",
        "mAP50 Drop": "mAP50 Drop",
        "mAP50-95 Drop": "mAP50-95 Drop",
        "IoU Drop": "IoU Drop",
        "Cls Acc Drop": "Cls Accuracy Drop",
        "Recall Drop": "Detection Recall Drop",
    }
    ordered_names = list(all_metrics.keys()) if requested_metrics is None else list(requested_metrics)
    metrics = [(name, all_metrics[name]) for name in ordered_names if name in all_metrics and name in df.columns]
    return metrics


def _draw_grouped_boxes(ax, df, group_col, group_values, group_labels, methods, colors, metric_col, metric_title):
    offsets = np.linspace(-0.35, 0.35, max(len(methods), 1))
    width = 0.65 / max(len(methods), 1)
    centers = np.arange(len(group_values))

    for method_idx, method in enumerate(methods):
        data = []
        positions = []
        for group_idx, group_value in enumerate(group_values):
            subset = df[(df[group_col] == group_value) & (df["Attribution Method"] == method)][metric_col]
            values = subset.dropna().astype(float).tolist()
            if not values:
                continue
            data.append(values)
            positions.append(centers[group_idx] + offsets[method_idx])

        if not data:
            continue

        box = ax.boxplot(
            data,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
        for patch in box["boxes"]:
            patch.set(facecolor=colors.get(method, "#5a4491"), edgecolor="#2b1f4a", linewidth=0.8)
        for median in box["medians"]:
            median.set(color="#1f1f1f", linewidth=1.0)
        for whisker in box["whiskers"]:
            whisker.set(color="#2b1f4a", linewidth=0.8)
        for cap in box["caps"]:
            cap.set(color="#2b1f4a", linewidth=0.8)

        for pos, values in zip(positions, data):
            ax.scatter(
                np.full(len(values), pos),
                values,
                s=10,
                c="#1f1f1f",
                alpha=0.35,
                linewidths=0,
                zorder=3,
            )

    ax.set_xlabel(group_col.replace("-", " "))
    ax.set_ylabel(metric_title)
    ax.set_title(metric_title)
    ax.set_xticks(centers)
    ax.set_xticklabels(group_labels)
    ax.axhline(y=0, color="black", linewidth=0.6, linestyle="--", alpha=0.7)


def viz_rq1_acc_drop(csv_file, out_path=None, figsize=None, metrics=None):
    """Visualize RQ1 drops as grouped box plots over repeated runs."""
    df = pd.read_csv(csv_file)
    if out_path is None:
        out_path = csv_file.replace("_acc_drop.csv", "_acc_drop_plot.pdf")

    metrics = _rq1_available_metrics(df, requested_metrics=metrics)
    if not metrics:
        print("Warning: no drop metrics found in CSV, skipping plot.")
        return
    if figsize is None:
        figsize = (4 * len(metrics), 5)

    methods, colors = _rq1_method_colors(df["Attribution Method"].tolist())
    n_list = sorted(df["Top-N"].dropna().unique())

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, sharey=False)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric_col, metric_title) in zip(axes, metrics):
        _draw_grouped_boxes(
            ax=ax,
            df=df,
            group_col="Top-N",
            group_values=n_list,
            group_labels=[str(n) for n in n_list],
            methods=methods,
            colors=colors,
            metric_col=metric_col,
            metric_title=metric_title,
        )

    handles = [Patch(facecolor=colors.get(method, "#5a4491"), edgecolor="#2b1f4a", label=method) for method in methods]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(methods), 5),
        bbox_to_anchor=(0.5, -0.05),
        fontsize=8,
        frameon=True,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, format="pdf", dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"RQ1 plot saved: {out_path}")


def viz_rq1_cross_model_acc_drop(csv_specs, out_path, top_n=20, figsize=None, metrics=None):
    """Plot grouped box plots across YOLO model sizes for one Top-N setting.

    Parameters
    ----------
    csv_specs : list[tuple[str, str]]
        Pairs of (model_label, acc_drop_csv_path).
    out_path : str
        Output PDF path.
    top_n : int
        The Top-N pruning setting to visualize.
    """
    frames = []
    for model_label, csv_path in csv_specs:
        df = pd.read_csv(csv_path)
        df = df[df["Top-N"] == top_n].copy()
        df["Model Label"] = model_label
        frames.append(df)

    if not frames:
        print("Warning: no CSVs supplied to viz_rq1_cross_model_acc_drop, skipping plot.")
        return

    df = pd.concat(frames, ignore_index=True)
    metrics = _rq1_available_metrics(df, requested_metrics=metrics)
    if not metrics:
        print("Warning: no drop metrics found in cross-model CSVs, skipping plot.")
        return
    if figsize is None:
        figsize = (4 * len(metrics), 5)

    methods, colors = _rq1_method_colors(df["Attribution Method"].tolist())
    model_labels = [label for label, _path in csv_specs]

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, sharey=False)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric_col, metric_title) in zip(axes, metrics):
        _draw_grouped_boxes(
            ax=ax,
            df=df,
            group_col="Model Label",
            group_values=model_labels,
            group_labels=model_labels,
            methods=methods,
            colors=colors,
            metric_col=metric_col,
            metric_title=f"{metric_title} (Top-{top_n})",
        )
        ax.set_xlabel("YOLO Variant")

    handles = [Patch(facecolor=colors.get(method, "#5a4491"), edgecolor="#2b1f4a", label=method) for method in methods]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(methods), 5),
        bbox_to_anchor=(0.5, -0.05),
        fontsize=8,
        frameon=True,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, format="pdf", dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"RQ1 cross-model plot saved: {out_path}")


RQ2_MODEL_LABELS = {
    "n": "YOLOv11n",
    "s": "YOLOv11s",
    "m": "YOLOv11m",
}

RQ2_MODE_LABELS = {
    "implayer": "Imp-layer",
    "pergroup": "Per-group",
    "perlayer": "Per-layer",
}

RQ2_MODE_COLORS = {
    "implayer": "#1f77b4",
    "pergroup": "#ff7f0e",
    "perlayer": "#2ca02c",
}

RQ2_METRIC_SPECS = [
    ("precision", "Precision", "orig_precision", "I_precision", "R_precision"),
    ("recall", "Recall", "orig_recall", "I_recall", "R_recall"),
    ("map", "mAP50-95", "orig_map", "I_map", "R_map"),
]

RQ2_EXPECTED_FRACS = [2, 4, 6, 8, 10]


def viz_rq1_topk_focus(csv_file, out_dir, out_prefix=None, metrics=None, split_metrics=True):
    """Write focused Top-K RQ1 PDFs, one per metric by default."""
    csv_path = Path(csv_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_prefix or csv_path.stem.replace("_acc_drop", "")

    metric_specs = {
        "confidence": "Confidence Drop",
        "map50": "mAP50 Drop",
        "map5095": "mAP50-95 Drop",
    }
    keys = list(metric_specs) if metrics is None else list(metrics)

    if split_metrics:
        for key in keys:
            label = metric_specs[key]
            out_path = out_dir / f"{prefix}_{key}_topk.pdf"
            viz_rq1_acc_drop(
                str(csv_file),
                out_path=str(out_path),
                metrics=[label],
                figsize=(6, 5),
            )
        return

    confidence_out = out_dir / f"{prefix}_confidence_topk.pdf"
    map_out = out_dir / f"{prefix}_map_topk.pdf"
    viz_rq1_acc_drop(str(csv_file), out_path=str(confidence_out), metrics=["Confidence Drop"], figsize=(6, 5))
    viz_rq1_acc_drop(
        str(csv_file),
        out_path=str(map_out),
        metrics=["mAP50 Drop", "mAP50-95 Drop"],
        figsize=(8, 5),
    )


def _rq2_load_map_table(results_dir: Path):
    data = {}
    for csv_path in sorted(results_dir.glob("[nsm]_*_wis_map_results.csv")):
        model, mode, importance, *_ = csv_path.stem.split("_")
        if importance != "wis":
            continue

        df = pd.read_csv(csv_path)
        if "orig_map" not in df.columns:
            continue
        df = df.dropna(subset=["orig_map"]).copy()
        df = df.sort_values("frac_pct").drop_duplicates(subset=["frac_pct"], keep="last")
        data.setdefault(model, {})[mode] = df
    return data


def _rq2_load_coverage_table(results_dir: Path):
    rows = []
    for csv_path in sorted(results_dir.glob("[nsm]_*_wis_*pct_union_coverage.csv")):
        parts = csv_path.stem.split("_")
        if len(parts) < 4:
            continue
        model, mode, importance, frac_tag = parts[:4]
        if importance != "wis":
            continue

        frac_pct = int(frac_tag.replace("pct", ""))
        df = pd.read_csv(csv_path)
        i_row = df[(df["scope"] == "dataset") & (df["variant"] == "I")].iloc[0]
        r_row = df[(df["scope"] == "dataset") & (df["variant"] == "R")].iloc[0]
        rows.append(
            {
                "model": model,
                "mode": mode,
                "frac_pct": frac_pct,
                "C_O": float(i_row["C_O_overall"]),
                "C_I": float(i_row["C_union_overall"]),
                "C_R": float(r_row["C_union_overall"]),
                "delta_I": float(i_row["delta_overall"]),
                "delta_R": float(r_row["delta_overall"]),
            }
        )

    if not rows:
        return {}

    cov_df = pd.DataFrame(rows).sort_values(["model", "mode", "frac_pct"])
    data = {}
    for (model, mode), group in cov_df.groupby(["model", "mode"], sort=True):
        data.setdefault(model, {})[mode] = group.reset_index(drop=True)
    return data


def _rq2_frac_ticks(*dfs):
    fracs = sorted({int(v) for df in dfs for v in df["frac_pct"].tolist()})
    return [frac for frac in RQ2_EXPECTED_FRACS if frac in fracs] or fracs


def _rq2_dedup_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    by_label = {}
    for handle, label in zip(handles, labels):
        by_label[label] = handle
    ax.legend(by_label.values(), by_label.keys(), fontsize=9, frameon=True)


def _rq2_plot_metric(ax, metric_name, y_label, map_tables):
    fracs = _rq2_frac_ticks(*map_tables.values())
    baseline_drawn = False

    for mode in ["implayer", "pergroup", "perlayer"]:
        df = map_tables.get(mode)
        if df is None:
            continue

        df = df[df["frac_pct"].isin(fracs)].sort_values("frac_pct")
        color = RQ2_MODE_COLORS[mode]

        if not baseline_drawn:
            ax.plot(
                df["frac_pct"],
                df[f"orig_{metric_name}"],
                color="black",
                linestyle=":",
                linewidth=2.0,
                marker="o",
                label="Orig",
            )
            baseline_drawn = True

        ax.plot(
            df["frac_pct"],
            df[f"I_{metric_name}"],
            color=color,
            linestyle="-",
            linewidth=2.0,
            marker="o",
            label=f"{RQ2_MODE_LABELS[mode]} I",
        )
        ax.plot(
            df["frac_pct"],
            df[f"R_{metric_name}"],
            color=color,
            linestyle="--",
            linewidth=2.0,
            marker="s",
            label=f"{RQ2_MODE_LABELS[mode]} R",
        )

    ax.set_xlabel("Perturbation frac (%)")
    ax.set_ylabel(y_label)
    ax.set_xticks(fracs)
    ax.grid(True, alpha=0.3)


def _rq2_plot_coverage_lines(ax, coverage_tables):
    mode_order = [mode for mode in ["implayer", "pergroup", "perlayer"] if mode in coverage_tables]
    if not mode_order:
        return

    fracs = _rq2_frac_ticks(*[coverage_tables[mode] for mode in mode_order])
    for mode in mode_order:
        df = coverage_tables[mode]
        df = df[df["frac_pct"].isin(fracs)].sort_values("frac_pct")
        color = RQ2_MODE_COLORS[mode]
        ax.plot(
            df["frac_pct"],
            df["delta_I"],
            color=color,
            linestyle="-",
            linewidth=2.0,
            marker="o",
            label=f"{RQ2_MODE_LABELS[mode]} ΔI",
        )
        ax.plot(
            df["frac_pct"],
            df["delta_R"],
            color=color,
            linestyle="--",
            linewidth=2.0,
            marker="s",
            label=f"{RQ2_MODE_LABELS[mode]} ΔR",
        )

    ax.set_xticks(fracs)
    ax.set_xlabel("Perturbation frac (%)")
    ax.set_ylabel("Coverage change Δ")
    ax.grid(True, alpha=0.3)
    ax.axhline(0.0, color="black", linewidth=1.0)


def viz_rq2_wisdom_line_plots(results_dir, output_dir, models=("n", "s", "m")):
    """Generate YOLOv11 WISDOM-only frac-sweep line plots from result CSVs."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    map_tables = _rq2_load_map_table(results_dir)
    coverage_tables = _rq2_load_coverage_table(results_dir)
    saved_paths = []

    for model in models:
        model_maps = map_tables.get(model, {})
        model_cov = coverage_tables.get(model, {})
        if not model_maps or not model_cov:
            continue

        model_label = RQ2_MODEL_LABELS[model]
        for metric_name, y_label, *_cols in RQ2_METRIC_SPECS:
            fig, ax = plt.subplots(figsize=(8, 5))
            _rq2_plot_metric(ax, metric_name, y_label, model_maps)
            ax.set_title(f"{model_label} — {y_label} Trend (wisdom only)")
            _rq2_dedup_legend(ax)
            fig.tight_layout()

            metric_path = output_dir / f"{model_label}_{metric_name}.pdf"
            fig.savefig(metric_path, format="pdf", dpi=1200, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(metric_path)

        fig, ax = plt.subplots(figsize=(9, 5))
        _rq2_plot_coverage_lines(ax, model_cov)
        ax.set_title(f"{model_label} — Coverage Change Trend (wisdom only)")
        _rq2_dedup_legend(ax)
        fig.tight_layout()

        coverage_path = output_dir / f"{model_label}_coverage_change.pdf"
        fig.savefig(coverage_path, format="pdf", dpi=1200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(coverage_path)

    return saved_paths
