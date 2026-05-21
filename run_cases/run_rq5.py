#!/usr/bin/env python
"""
run_rq5.py – RQ5: Runtime and memory efficiency
=================================================
Measures the overhead introduced by the WISDOM approach compared to
single attribution methods and random neuron selection.

Records:
  - Attribution computation time per method
  - Consensus voting time
  - Pruning + evaluation time
  - Peak GPU memory usage

Output: rq5_efficiency.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wisdom_yolo_train import COCOImageDataset, _collate, train_wisdom_yolo
from wisdom.utils.detection_loader import load_detection_model
from wisdom.utils.yolo_wrapper import YOLOWrapper
from wisdom.core.wisdom_train import (
    _is_trainable_module,
    _compute_yolo_importance,
    _eval_loss_yolo,
)
from wisdom.pruning.mask_pruning import mask_model_neurons


# ── Timing helpers ─────────────────────────────────────────────────
def get_gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def reset_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


# ── RQ5 experiment ─────────────────────────────────────────────────
def run_rq5(
    weights: str,
    img_dir: str,
    csv_file: str,
    out_csv: str = "rq5_efficiency.csv",
    device: str = "cuda:0",
    num_images: int = 4,
    batch_size: int = 2,
    imgsz: int = 320,
    wisdom_only: bool = False,
    wisdom_methods: str = "lgxa,lig",
) -> str:
    bundle = load_detection_model(weights, device=device)
    torch_model = bundle.model.eval()
    wrapper = YOLOWrapper(torch_model, num_classes=bundle.num_classes)
    wrapper.eval().to(device)

    ds = COCOImageDataset(img_dir, max_images=num_images, imgsz=imgsz)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    first_batch = next(iter(loader))
    images = first_batch[0]
    consensus_methods = [m.strip() for m in wisdom_methods.split(",") if m.strip()]
    if not consensus_methods:
        raise ValueError("wisdom_methods must contain at least one attribution method.")

    records = []

    # 1. Single attribution method timing
    if not wisdom_only:
        methods_to_time = ["lgxa", "lig", "lgs"]
        for method in methods_to_time:
            reset_gpu_memory()
            t0 = time.perf_counter()
            _compute_yolo_importance(wrapper, images, method, device, num_classes=bundle.num_classes)
            t1 = time.perf_counter()
            mem = get_gpu_memory_mb()
            records.append({
                "Operation": f"Attribution ({method})",
                "Time (s)": round(t1 - t0, 4),
                "Peak GPU Memory (MB)": round(mem, 1),
            })
            print(f"  Attribution ({method}): {t1-t0:.4f}s, {mem:.1f} MB")
            reset_gpu_memory()

    # 2. Consensus voting (full WISDOM pipeline) timing
    import tempfile
    reset_gpu_memory()
    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp_csv = f.name
    try:
        train_wisdom_yolo(
            weights=weights,
            img_dir=img_dir,
            out_csv=tmp_csv,
            batch_size=batch_size,
            num_images=num_images,
            top_m=10,
            methods=consensus_methods,
            voting_mode="fine-grained",
            device=device,
            imgsz=imgsz,
        )
    finally:
        if os.path.exists(tmp_csv):
            os.unlink(tmp_csv)
    t1 = time.perf_counter()
    mem = get_gpu_memory_mb()
    records.append({
        "Operation": f"WISDOM Consensus ({'+'.join(consensus_methods)})",
        "Time (s)": round(t1 - t0, 4),
        "Peak GPU Memory (MB)": round(mem, 1),
    })
    print(f"  WISDOM Consensus: {t1-t0:.4f}s, {mem:.1f} MB")

    # 3. Pruning + evaluation timing
    reset_gpu_memory()
    scores_df = pd.read_csv(csv_file)
    top_neurons = scores_df.nlargest(10, "Score")
    selection: Dict[str, List[int]] = {}
    for _, row in top_neurons.iterrows():
        lname = row["LayerName"]
        mapped = lname.replace("yolo_model.", "") if lname.startswith("yolo_model.") else lname
        selection.setdefault(mapped, []).append(int(row["NeuronIndex"]))

    t0 = time.perf_counter()
    handle = mask_model_neurons(torch_model, selection)
    _eval_loss_yolo(torch_model, images, device)
    handle.remove()
    t1 = time.perf_counter()
    mem = get_gpu_memory_mb()
    records.append({
        "Operation": "Pruning + Evaluation",
        "Time (s)": round(t1 - t0, 4),
        "Peak GPU Memory (MB)": round(mem, 1),
    })
    print(f"  Pruning + Eval: {t1-t0:.4f}s, {mem:.1f} MB")

    if not wisdom_only:
        # 4. Baseline: plain forward pass
        reset_gpu_memory()
        t0 = time.perf_counter()
        _eval_loss_yolo(torch_model, images, device)
        t1 = time.perf_counter()
        mem = get_gpu_memory_mb()
        records.append({
            "Operation": "Baseline Forward Pass",
            "Time (s)": round(t1 - t0, 4),
            "Peak GPU Memory (MB)": round(mem, 1),
        })
        print(f"  Baseline Forward: {t1-t0:.4f}s, {mem:.1f} MB")

        # 5. Random neuron selection baseline
        import random
        reset_gpu_memory()
        trainable = [(n, m) for n, m in torch_model.named_modules() if _is_trainable_module(m)]
        all_neurons = [(n, i) for n, m in trainable
                       for i in range(m.out_channels if isinstance(m, nn.Conv2d) else m.out_features)]
        rand_selection: Dict[str, List[int]] = {}
        for n, idx in random.sample(all_neurons, 10):
            rand_selection.setdefault(n, []).append(idx)

        t0 = time.perf_counter()
        handle = mask_model_neurons(torch_model, rand_selection)
        _eval_loss_yolo(torch_model, images, device)
        handle.remove()
        t1 = time.perf_counter()
        mem = get_gpu_memory_mb()
        records.append({
            "Operation": "Random Pruning + Evaluation",
            "Time (s)": round(t1 - t0, 4),
            "Peak GPU Memory (MB)": round(mem, 1),
        })
        print(f"  Random Prune + Eval: {t1-t0:.4f}s, {mem:.1f} MB")

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)

    log_dir = os.path.join(os.path.dirname(out_csv) or ".", "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "rq5_results.log")
    with open(log_path, "w") as f:
        f.write("RQ5: Efficiency summary\n")
        f.write("=" * 60 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n")
    print(f"Log saved: {log_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_base = out_csv.replace(".csv", "")

        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.bar(df["Operation"], df["Time (s)"], color="#5a4491")
        ax.set_ylabel("Time (s)")
        ax.set_title("RQ5: Operation Runtime")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        time_plot_path = f"{plot_base}_time.pdf"
        fig.savefig(time_plot_path, format="pdf", dpi=1200, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.bar(df["Operation"], df["Peak GPU Memory (MB)"], color="#d0beed")
        ax.set_ylabel("Peak GPU Memory (MB)")
        ax.set_title("RQ5: Peak GPU Memory")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        mem_plot_path = f"{plot_base}_memory.pdf"
        fig.savefig(mem_plot_path, format="pdf", dpi=1200, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved: {time_plot_path}")
        print(f"Plot saved: {mem_plot_path}")
    except Exception as e:
        print(f"Warning: could not generate plot: {e}")

    print(f"\nSaved: {out_csv}")
    print(df.to_string(index=False))
    return out_csv


# ── CLI ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RQ5: Efficiency evaluation for YOLOv11")
    p.add_argument("--weights", default="weights/yolo11n.pt")
    p.add_argument("--img-dir", default="standalone/data/coco/images/val2017")
    p.add_argument("--csv-file", default="neuron_eval_out/wisdom_yolo11n_scores.csv")
    p.add_argument("--out-csv", default="results/rq5_yolo11n_efficiency.csv")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-images", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--wisdom-only", action="store_true", help="Only measure the WISDOM path and skip single-method/random baselines.")
    p.add_argument("--wisdom-methods", default="lgxa,lig", help="Comma-separated attribution methods for the WISDOM consensus timing run.")
    args = p.parse_args()
    run_rq5(**vars(args))
