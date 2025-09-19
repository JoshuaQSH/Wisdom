# src/wisdom/utils/io_cache.py
from __future__ import annotations
from typing import Dict, Optional
import os, json
import numpy as np

def _cache_dir(path: Optional[str]) -> Optional[str]:
    if path is None: return None
    os.makedirs(path, exist_ok=True)
    return path

def save_cluster_groups(cache_path: Optional[str],
                        tag: str,
                        groups: Dict[str, Dict[int, dict]]) -> None:
    if cache_path is None: return
    d = _cache_dir(cache_path)
    meta_file = os.path.join(d, f"{tag}.json")
    npz_file = os.path.join(d, f"{tag}.npz")
    meta = {}
    arrays = {}
    for layer, dct in groups.items():
        meta[layer] = {}
        for i, info in dct.items():
            kc = f"{layer}::{i}::centers"
            kl = f"{layer}::{i}::labels"
            meta[layer][str(i)] = {"method": info["method"], "params": info["params"],
                                   "cent_key": kc, "lab_key": kl}
            arrays[kc] = info["centers"]
            arrays[kl] = info["labels"]
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    np.savez_compressed(npz_file, **arrays)

def load_cluster_groups(cache_path: Optional[str], tag: str) -> Optional[Dict[str, Dict[int, dict]]]:
    if cache_path is None: return None
    d = _cache_dir(cache_path)
    meta_file = os.path.join(d, f"{tag}.json")
    npz_file = os.path.join(d, f"{tag}.npz")
    if not (os.path.exists(meta_file) and os.path.exists(npz_file)):
        return None
    with open(meta_file, "r") as f:
        meta = json.load(f)
    npz = np.load(npz_file, allow_pickle=True)
    groups: Dict[str, Dict[int, dict]] = {}
    for layer, dct in meta.items():
        groups[layer] = {}
        for i_str, info in dct.items():
            centers = npz[info["cent_key"]]
            labels  = npz[info["lab_key"]]
            groups[layer][int(i_str)] = {"method": info["method"], "params": info["params"],
                                         "centers": centers, "labels": labels}
    return groups