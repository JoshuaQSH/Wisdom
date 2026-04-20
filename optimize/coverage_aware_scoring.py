#!/usr/bin/env python3
"""Coverage-Aware Neuron Scoring.

Combines attribution importance scores with mean boundary proximity
to produce a coverage-aware score that favours neurons which are both
important AND frequently near cluster boundaries.

Usage:
    python -m optimize.coverage_aware_scoring \
        --importance-csv neuron_eval_out/yolo11n_lgxa_500.csv \
        --weights weights/yolo11n.pt \
        --img-dir standalone/data/coco/images/val2017 \
        --output neuron_eval_out/yolo11n_lgxa_cas_500.csv \
        --n-calib 200 --n-clusters 3
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# Allow running as module from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimize.coverage_utils import ActivationCollector


def _load_csv(path: str):
    """Load importance CSV → {layer: {neuron_idx: score}}."""
    PREFIX = "yolo_model."
    data: dict = defaultdict(dict)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            lname = row["LayerName"]
            if lname.startswith(PREFIX):
                lname = lname[len(PREFIX):]
            idx = int(row["NeuronIndex"])
            score = float(row["Score"])
            data[lname][idx] = score
    return dict(data)


def compute_boundary_proximity(
    model, target_neurons, images, device, n_clusters=3, batch_size=4,
):
    """Compute mean boundary proximity for each neuron over calibration images.

    Returns {layer: {idx: mean_inverse_boundary_distance}}.
    Neurons with activations frequently near cluster boundaries get high values.
    """
    from wisdom.clustering.assign import fit_per_neuron

    collector = ActivationCollector(model, target_neurons, device)
    collector.attach()

    # Collect activation series
    series = {l: {i: [] for i in idxs} for l, idxs in target_neurons.items()}
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        acts = collector.collect(batch)
        for lname, act_t in acts.items():
            cpu = act_t.detach().cpu()
            for col, idx in enumerate(target_neurons[lname]):
                series[lname][idx].extend(cpu[:, col].tolist())
    collector.detach()

    # Fit clusters
    np_series = {
        l: {idx: np.asarray(vals, dtype=np.float64) for idx, vals in d.items()}
        for l, d in series.items()
    }
    groups = fit_per_neuron(
        np_series, method="KMeans",
        params={"n_clusters": n_clusters, "random_state": 42},
        use_silhouette=False, k_max=n_clusters + 2,
    )

    # Compute mean boundary proximity per neuron
    proximity = {}
    for lname in groups:
        proximity[lname] = {}
        for idx in groups[lname]:
            centers = groups[lname][idx]["centers"].squeeze()  # (C,)
            centers_sorted = np.sort(centers)
            if len(centers_sorted) < 2:
                proximity[lname][idx] = 0.0
                continue
            boundaries = np.array([
                (centers_sorted[i] + centers_sorted[i + 1]) / 2
                for i in range(len(centers_sorted) - 1)
            ])
            vals = np_series[lname][idx]
            # For each activation value, distance to nearest boundary
            dists = np.abs(vals[:, None] - boundaries[None, :])
            min_dists = dists.min(axis=1)  # (N,)
            # Mean inverse distance (higher = closer to boundaries on average)
            mean_inv = float(np.mean(1.0 / (min_dists + 0.01)))
            proximity[lname][idx] = mean_inv

    return proximity


def main():
    p = argparse.ArgumentParser(description="Coverage-Aware Neuron Scoring")
    p.add_argument("--importance-csv", required=True,
                   help="Input importance scores CSV (WISDOM or lgxa format)")
    p.add_argument("--weights", default="weights/yolo11n.pt")
    p.add_argument("--img-dir", default="standalone/data/coco/images/val2017")
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument("--n-calib", type=int, default=200,
                   help="Number of calibration images for boundary proximity")
    p.add_argument("--n-clusters", type=int, default=3)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()

    print(f"Loading importance scores from {a.importance_csv}")
    scores = _load_csv(a.importance_csv)
    total_neurons = sum(len(v) for v in scores.values())
    print(f"  {total_neurons} neurons across {len(scores)} layers")

    # Build target_neurons for all scored neurons
    target_neurons = {l: sorted(d.keys()) for l, d in scores.items()}

    # Load model
    from ultralytics import YOLO
    yolo = YOLO(a.weights)
    model = yolo.model.eval().to(a.device)

    # Load calibration images
    from optimize.run_rq2_opt import COCOImageDataset
    ds = COCOImageDataset(a.img_dir, max_images=a.n_calib, imgsz=a.imgsz)
    calib_imgs = torch.stack([ds[i][0] for i in range(len(ds))])
    print(f"Loaded {len(calib_imgs)} calibration images")

    # Compute boundary proximity
    print(f"Computing boundary proximity (n_clusters={a.n_clusters})...")
    proximity = compute_boundary_proximity(
        model, target_neurons, calib_imgs, a.device,
        n_clusters=a.n_clusters, batch_size=4,
    )

    # Combine scores: CAS = importance × boundary_proximity
    rows = []
    for lname in scores:
        for idx, imp_score in scores[lname].items():
            prox = proximity.get(lname, {}).get(idx, 0.0)
            cas_score = imp_score * prox
            rows.append({
                "LayerName": f"yolo_model.{lname}",
                "NeuronIndex": idx,
                "Score": cas_score,
                "ImportanceScore": imp_score,
                "BoundaryProximity": prox,
            })

    # Sort by Score descending for readability
    rows.sort(key=lambda r: r["Score"], reverse=True)

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["LayerName", "NeuronIndex", "Score",
                                                "ImportanceScore", "BoundaryProximity"])
        writer.writeheader()
        writer.writerows(rows)

    n_positive = sum(1 for r in rows if r["Score"] > 0)
    print(f"\nCoverage-aware scores saved to {a.output}")
    print(f"  {len(rows)} neurons, {n_positive} with CAS > 0")

    # Show top-10
    print("\nTop-10 Coverage-Aware Neurons:")
    print(f"  {'Layer':<40s} {'Idx':>4s} {'CAS':>10s} {'Imp':>10s} {'Prox':>8s}")
    for r in rows[:10]:
        ln = r["LayerName"].replace("yolo_model.", "")
        print(f"  {ln:<40s} {r['NeuronIndex']:>4d} {r['Score']:>10.4f} "
              f"{r['ImportanceScore']:>10.4f} {r['BoundaryProximity']:>8.4f}")


if __name__ == "__main__":
    main()
