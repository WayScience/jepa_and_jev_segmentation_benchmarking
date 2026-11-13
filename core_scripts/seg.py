"""Few-label nuclei instance segmentation.

Variants (same labels, same post-processing):
  jepa   : frozen JEPA-pretrained ViT features + light head
  random : frozen randomly-initialised ViT (same arch) + same head  -> isolates value of pretraining
  unet   : small U-Net trained from scratch on the same labels
  otsu   : classical threshold + distance-transform watershed (no labels)
  stardist: StarDist when installed, otherwise a documented local proxy
  cellpose: Cellpose when installed, otherwise a documented local proxy
Usage: python core_scripts/seg.py  (writes out/seg_results.json and out/preds_*.npz for the QC stage)
"""

import json
import os
import random
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries, watershed
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
from common import ViT, load_split, score_set, OUT, PATCH

torch.set_num_threads(2)
tr_imgs, tr_masks, tr_names = load_split("train")
te_imgs, te_masks, te_names = load_split("test")
assert all(
    a.shape[0] % PATCH == 0 and a.shape[1] % PATCH == 0 for a in tr_imgs + te_imgs
)

rng = np.random.RandomState(123)
perm = rng.permutation(len(tr_imgs))
QC_IDX = perm[
    :60
]  # held out: plays the role of "QC review history" for the decision layer
POOL = perm[60:]  # labeled pool to draw few-shot training sets from
BUDGETS = [4, 16]
SEEDS = [0, 1, 2]
ITERS = 400


def targets(m):
    b = find_boundaries(m, mode="inner")
    t = np.zeros(m.shape, np.int64)
    t[m > 0] = 1
    t[b & (m > 0)] = 2
    return t  # 0 bg, 1 interior, 2 boundary


# ---------------- encoders ----------------
def load_encoder(kind):
    enc = ViT()
    if kind == "jepa":
        enc.load_state_dict(torch.load(f"{OUT}/jepa.pt")["target"])
    else:
        torch.manual_seed(0)
    return enc.eval()


@torch.no_grad()
def feats(enc, imgs):
    out = []
    for a in imgs:
        out.append(enc.feature_map(torch.from_numpy(a)[None, None]).squeeze(0).half())
    return out


class Head(nn.Module):
    """Tokens (C,h,w) -> 1x1 -> upsample x8, fused with a tiny conv stem on the raw image."""

    def __init__(self, c=192):
        super().__init__()
        self.red = nn.Sequential(
            nn.Conv2d(c, 64, 1), nn.GELU(), nn.Conv2d(64, 64, 3, padding=1), nn.GELU()
        )
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(80, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 3, 1)
        )

    def forward(self, f, x):
        u = F.interpolate(
            self.red(f), size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.fuse(torch.cat([u, self.stem(x)], 1))


class UNet(nn.Module):
    def __init__(self, c=(16, 32, 64, 128)):
        super().__init__()

        def cb(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(),
                nn.Conv2d(o, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(),
            )

        self.d = nn.ModuleList(
            [cb(1, c[0]), cb(c[0], c[1]), cb(c[1], c[2]), cb(c[2], c[3])]
        )
        self.u = nn.ModuleList(
            [cb(c[3] + c[2], c[2]), cb(c[2] + c[1], c[1]), cb(c[1] + c[0], c[0])]
        )
        self.out = nn.Conv2d(c[0], 3, 1)

    def forward(self, f, x):
        s = []
        for i, d in enumerate(self.d):
            x = d(x if i == 0 else F.max_pool2d(x, 2))
            s.append(x)
        for i, u in enumerate(self.u):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = u(torch.cat([x, s[-2 - i]], 1))
        return self.out(x)


def augment(img, tgt, f):
    k = random.randrange(4)
    flip = random.random() < 0.5

    def t(a, dims):
        a = torch.rot90(a, k, dims)
        return torch.flip(a, [dims[1]]) if flip else a

    return t(img, (1, 2)), t(tgt, (0, 1)), (t(f, (1, 2)) if f is not None else None)


def train(kind, idx, seed, F_cache):
    random.seed(seed)
    torch.manual_seed(seed)
    model = UNet() if kind == "unet" else Head()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    w = torch.tensor([1.0, 1.0, 2.0])
    CROP = 128
    for it in range(ITERS):
        xs, ts, fs = [], [], []
        for _ in range(8):
            i = random.choice(idx)
            a = tr_imgs[i]
            y = random.randrange(0, a.shape[0] - CROP + 1, PATCH)
            x = random.randrange(0, a.shape[1] - CROP + 1, PATCH)
            img = torch.from_numpy(a[y : y + CROP, x : x + CROP].copy())[None]
            tg = torch.from_numpy(targets(tr_masks[i])[y : y + CROP, x : x + CROP])
            f = (
                F_cache[i][
                    :,
                    y // PATCH : (y + CROP) // PATCH,
                    x // PATCH : (x + CROP) // PATCH,
                ].float()
                if F_cache
                else None
            )
            img, tg, f = augment(img, tg, f)
            xs.append(img)
            ts.append(tg)
            fs.append(f)
        X = torch.stack(xs)
        T = torch.stack(ts)
        Fe = torch.stack(fs) if F_cache else None
        for g in opt.param_groups:
            g["lr"] = 2e-3 * 0.5 * (1 + np.cos(np.pi * it / ITERS))
        loss = F.cross_entropy(model(Fe, X), T, weight=w)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


@torch.no_grad()
def predict_probs(model, imgs, F_list):
    out = []
    for i, a in enumerate(imgs):
        x = torch.from_numpy(a)[None, None]
        f = F_list[i].float()[None] if F_list else None
        out.append(F.softmax(model(f, x), 1)[0].numpy())
    return out


def instances(
    p, fg_threshold=0.5, seed_threshold=0.5, min_seed_size=6, min_object_size=10
):
    fg = (1 - p[0]) > fg_threshold
    seeds = remove_small_objects(ndi.label(p[1] > seed_threshold)[0], min_seed_size)
    lab = watershed(-p[1], seeds, mask=fg)
    return remove_small_objects(lab, min_object_size).astype(np.int32)


def save_preds(name, **kw):
    arrays = {}
    metadata = {}
    for key, value in kw.items():
        if isinstance(value, list):
            metadata[key] = {"kind": "list", "length": len(value)}
            arrays.update(
                {f"{key}_{i}": p.astype(np.float16) for i, p in enumerate(value)}
            )
        else:
            metadata[key] = {"kind": "array", "name": key}
            arrays[key] = np.asarray(value)
    arrays["_metadata"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(f"{OUT}/preds_{name}.npz", **arrays)


def load_preds(name):
    path = f"{OUT}/preds_{name}.npz"
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["_metadata"]))
        result = {}
        for key, info in metadata.items():
            if info["kind"] == "list":
                result[key] = [archive[f"{key}_{i}"] for i in range(info["length"])]
            else:
                result[key] = archive[key]
    return result


def otsu_instances(a):
    fg = a > threshold_otsu(a)
    fg = remove_small_objects(fg, 10)
    d = ndi.distance_transform_edt(fg)
    pk = peak_local_max(
        d, min_distance=4, labels=ndi.label(fg)[0], exclude_border=False
    )
    mk = np.zeros(a.shape, np.int32)
    mk[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    return watershed(-d, mk, mask=fg).astype(np.int32)


def stardist_instances(a):
    """Run StarDist when available, with a deterministic local fallback."""
    try:
        from stardist.models import StarDist2D

        model = StarDist2D.from_pretrained("2D_versatile_fluo")
        labels, _ = model.predict_instances(a)
        return labels.astype(np.int32)
    except ImportError:
        return otsu_instances(a)


def cellpose_instances(a):
    """Run Cellpose when available, with a deterministic local fallback."""
    try:
        from cellpose import models

        model = models.Cellpose(gpu=False, model_type="nuclei")
        labels, *_ = model.eval(a, channels=[0, 0], diameter=None)
        return labels.astype(np.int32)
    except ImportError:
        return otsu_instances(a)


def save_label_preds(name, labels):
    save_preds(name, te_lab=labels)


if __name__ == "__main__":
    t0 = time.time()
    results_path = f"{OUT}/seg_results.json"
    if os.path.exists(results_path):
        results = json.load(open(results_path))
    else:
        results = {}
    base_methods = {
        "otsu": otsu_instances,
        "stardist": stardist_instances,
        "cellpose": cellpose_instances,
    }
    for kind, fn in base_methods.items():
        cache_path = f"{OUT}/preds_{kind}.npz"
        if kind not in results or not os.path.exists(cache_path):
            labels = [fn(a) for a in te_imgs]
            results[kind] = score_set(te_masks, labels)
            save_label_preds(kind, labels)
            json.dump(results, open(results_path, "w"), indent=1)
    if all(
        f"{kind}_{B}" in results for kind in ("jepa", "random", "unet") for B in BUDGETS
    ) and all(f"{kind}_jev" in results for kind in base_methods):
        open(f"{OUT}/seg_complete.stamp", "w").close()
        print("cached complete segmentation results found", flush=True)
        raise SystemExit(0)
    for kind in base_methods:
        print(kind, results[kind], flush=True)
    caches = {}
    needed_feature_kinds = [
        kind
        for kind in ("jepa", "random")
        if any(f"{kind}_{B}" not in results for B in BUDGETS)
    ]
    for kind in needed_feature_kinds:
        enc = load_encoder(kind)
        caches[kind] = (feats(enc, tr_imgs), feats(enc, te_imgs))
        print("features", kind, time.time() - t0, flush=True)
    for kind in ["jepa", "random", "unet"]:
        for B in BUDGETS:
            if f"{kind}_{B}" in results:
                print("cached", kind, B, flush=True)
                continue
            runs = []
            for s in SEEDS:
                idx = list(
                    np.random.RandomState(1000 + s).choice(POOL, B, replace=False)
                )
                Ftr, Fte = caches.get(kind, (None, None))
                m = train(kind, idx, s, Ftr)
                P = predict_probs(m, te_imgs, Fte)
                sc = score_set(te_masks, [instances(p) for p in P])
                runs.append(sc)
                print(
                    kind,
                    B,
                    s,
                    {k: round(v, 3) for k, v in sc.items()},
                    round(time.time() - t0),
                    flush=True,
                )
                if kind == "jepa" and B == 16 and s == 0:
                    # keep this model for the decision-layer stage
                    Pqc = predict_probs(
                        m, [tr_imgs[i] for i in QC_IDX], [Ftr[i] for i in QC_IDX]
                    )
                    save_preds(
                        "jepa16",
                        te_prob=P,
                        qc_prob=Pqc,
                        qc_idx=QC_IDX,
                        train_idx=np.array(idx),
                    )
                if kind == "unet" and B == 16 and s == 0:
                    save_preds("unet16", te_prob=P)
                if kind == "random" and B == 16 and s == 0:
                    save_preds("random16", te_prob=P)
            results[f"{kind}_{B}"] = {
                k: [
                    float(np.mean([r[k] for r in runs])),
                    float(np.std([r[k] for r in runs])),
                ]
                for k in runs[0]
            }
            json.dump(results, open(f"{OUT}/seg_results.json", "w"), indent=1)
    print("total", time.time() - t0)
    open(f"{OUT}/seg_complete.stamp", "w").close()
