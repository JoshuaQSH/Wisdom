# wisdom/utils/visulization.py
import matplotlib.pyplot as plt
from captum.attr import visualization as viz
import numpy as np
import torch
import pandas as pd
import random

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
METHOD_COLORS = {
    "Wisdom": "#2ca02c",       # Greens palette
    "GradXAct": "#9467bd",     # Purples palette
    "IntegGrad": "#d62728",    # Reds palette
    "GradShap": "#1f77b4",     # Blues palette
}
RANDOM_COLOR = "#7f7f7f"


def viz_rq1_acc_drop(csv_file, out_path=None, figsize=(14, 5)):
    """Visualize RQ1 accuracy drop results as a grouped bar chart.

    Plots three sub-figures side by side:
      1. Confidence Drop vs Top-N neurons pruned
      2. IoU Drop vs Top-N neurons pruned
      3. Classification Accuracy Drop vs Top-N neurons pruned

    Each method gets its own colour following the project's color_choice
    palette; random baselines are shown in grey.

    Parameters
    ----------
    csv_file : str
        Path to the RQ1 acc_drop CSV (output of run_rq1).
    out_path : str, optional
        Output PDF path.  Defaults to ``rq1_acc_drop.pdf`` next to csv_file.
    figsize : tuple
        Figure size (width, height).
    """
    df = pd.read_csv(csv_file)
    if out_path is None:
        out_path = csv_file.replace("_acc_drop.csv", "_acc_drop_plot.pdf")

    metrics = [
        ("Confidence Drop", "Confidence Drop"),
        ("IoU Drop", "IoU Drop"),
        ("Cls Acc Drop", "Classification Accuracy Drop"),
    ]
    # Keep only metrics that exist in the CSV
    metrics = [(col, title) for col, title in metrics if col in df.columns]
    if not metrics:
        print("Warning: no drop metrics found in CSV, skipping plot.")
        return

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, sharey=False)
    if n_metrics == 1:
        axes = [axes]

    methods = df["Attribution Method"].unique()
    n_list = sorted(df["Top-N"].unique())
    bar_width = 0.8 / len(methods)
    x = np.arange(len(n_list))

    for ax, (col, title) in zip(axes, metrics):
        for i, method in enumerate(methods):
            subset = df[df["Attribution Method"] == method]
            vals = []
            for n in n_list:
                row = subset[subset["Top-N"] == n]
                vals.append(float(row[col].values[0]) if len(row) > 0 else 0.0)

            if "Random" in method:
                color = RANDOM_COLOR
                hatch = "//"
            else:
                color = METHOD_COLORS.get(method, "#333333")
                hatch = None

            bars = ax.bar(
                x + i * bar_width, vals, bar_width,
                label=method, color=color, alpha=0.85,
                hatch=hatch, edgecolor="white", linewidth=0.5,
            )

        ax.set_xlabel("Top-N Neurons Pruned")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(x + bar_width * (len(methods) - 1) / 2)
        ax.set_xticklabels([str(n) for n in n_list])
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")

    # Single legend outside the subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(methods), 4),
               bbox_to_anchor=(0.5, -0.05), fontsize=8, frameon=True)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, format="pdf", dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"RQ1 plot saved: {out_path}")