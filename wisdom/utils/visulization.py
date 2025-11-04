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