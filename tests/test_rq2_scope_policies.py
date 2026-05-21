import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from optimize.run_rq2_opt import (
    _select_important_pixel_indices,
    _select_random_pixel_indices,
)


def test_object_vs_full_random_excludes_selected_object_pixels():
    importance = torch.arange(16, dtype=torch.float32).view(4, 4)
    obj_mask = torch.zeros(1, 4, 4)
    obj_mask[:, 0, :] = 1.0

    imp_idx = _select_important_pixel_indices(importance, frac=0.5, allowed_mask=obj_mask[0])
    torch.manual_seed(0)
    rnd_idx = _select_random_pixel_indices(
        importance.numel(),
        frac=0.5,
        allowed_mask=None,
        exclude_idx=imp_idx,
    )

    assert sorted(imp_idx.tolist()) == [2, 3]
    assert set(imp_idx.tolist()).isdisjoint(set(rnd_idx.tolist()))


def test_masked_only_selection_stays_in_background():
    importance = torch.arange(16, dtype=torch.float32).view(4, 4)
    obj_mask = torch.zeros(1, 4, 4)
    obj_mask[:, 0, :] = 1.0
    bg_mask = 1.0 - obj_mask

    imp_idx = _select_important_pixel_indices(importance, frac=0.25, allowed_mask=bg_mask[0])
    torch.manual_seed(1)
    rnd_idx = _select_random_pixel_indices(
        importance.numel(),
        frac=0.25,
        allowed_mask=bg_mask[0],
        exclude_idx=imp_idx,
    )

    bg_idx = set(torch.nonzero(bg_mask[0].view(-1) > 0.5, as_tuple=True)[0].tolist())
    assert set(imp_idx.tolist()).issubset(bg_idx)
    assert set(rnd_idx.tolist()).issubset(bg_idx)
    assert set(imp_idx.tolist()).isdisjoint(set(rnd_idx.tolist()))


def test_full_image_random_excludes_full_image_important_pixels():
    importance = torch.arange(9, dtype=torch.float32).view(3, 3)
    imp_idx = _select_important_pixel_indices(importance, frac=1 / 3)
    torch.manual_seed(2)
    rnd_idx = _select_random_pixel_indices(importance.numel(), frac=1 / 3, exclude_idx=imp_idx)

    assert sorted(imp_idx.tolist()) == [6, 7, 8]
    assert len(rnd_idx) == 3
    assert set(imp_idx.tolist()).isdisjoint(set(rnd_idx.tolist()))
