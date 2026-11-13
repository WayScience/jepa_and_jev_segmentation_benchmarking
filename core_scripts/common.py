"""Shared pieces: data loading, ViT encoder, metrics."""

import glob
import os
import numpy as np
import tifffile
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.environ.get("JEPA_DATA", os.path.join(ROOT, "data", "dsb2018"))
OUT = os.environ.get("JEPA_OUT", os.path.join(ROOT, "out"))
os.makedirs(OUT, exist_ok=True)
PATCH = 8


def norm_img(a):
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99.8)
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


def load_split(split):
    fs = sorted(glob.glob(f"{DATA}/{split}/images/*.tif"))
    if not fs:
        raise FileNotFoundError(
            f"No {split} images found in {DATA}/{split}/images. "
            "Download the DSB2018 dataset and set JEPA_DATA to its dsb2018 directory."
        )
    imgs, masks, names = [], [], []
    for f in fs:
        a = tifffile.imread(f)
        if a.ndim == 3:  # safety: collapse channels
            a = a[..., :3].mean(-1) if a.shape[-1] in (3, 4) else a[0]
        m = tifffile.imread(f.replace("/images/", "/masks/")).astype(np.int32)
        H, W = (
            (a.shape[0] // PATCH) * PATCH,
            (a.shape[1] // PATCH) * PATCH,
        )  # trim to patch multiple
        imgs.append(norm_img(a)[:H, :W].copy())
        masks.append(m[:H, :W].copy())
        names.append(os.path.basename(f))
    return imgs, masks, names


def sincos_2d(h, w, dim):
    """Fixed 2D sin-cos position embedding (works for any grid size)."""
    gy, gx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    d = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(d, dtype=torch.float32) / d))
    oy = gy.flatten()[:, None] * omega[None]
    ox = gx.flatten()[:, None] * omega[None]
    return torch.cat([oy.sin(), oy.cos(), ox.sin(), ox.cos()], 1)  # (h*w, dim)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp)), nn.GELU(), nn.Linear(int(dim * mlp), dim)
        )

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class ViT(nn.Module):
    """Patch-8 ViT encoder for 1-channel images. keep_idx selects visible tokens."""

    def __init__(self, dim=192, depth=6, heads=3):
        super().__init__()
        self.dim = dim
        self.proj = nn.Conv2d(1, dim, PATCH, PATCH)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, keep_idx=None):
        B, _, H, W = x.shape
        h, w = H // PATCH, W // PATCH
        t = self.proj(x).flatten(2).transpose(1, 2) + sincos_2d(h, w, self.dim).to(
            x.device
        )
        if keep_idx is not None:
            t = t[:, keep_idx]
        for b in self.blocks:
            t = b(t)
        return self.norm(t)

    @torch.no_grad()
    def feature_map(self, x):
        B, _, H, W = x.shape
        t = self.forward(x)
        return t.transpose(1, 2).reshape(B, self.dim, H // PATCH, W // PATCH)


# ---------------- metrics ----------------
def match_instances(gt, pred, thr=0.5):
    """Greedy-optimal (Hungarian) IoU matching. Returns tp, fp, fn, matched IoUs, per-pred best IoU dict."""
    from scipy.optimize import linear_sum_assignment

    gi = [i for i in np.unique(gt) if i]
    pi = [i for i in np.unique(pred) if i]
    if not gi or not pi:
        return 0, len(pi), len(gi), [], {p: 0.0 for p in pi}
    # contingency via bincount
    g_idx = {v: k for k, v in enumerate(gi)}
    p_idx = {v: k for k, v in enumerate(pi)}
    gmap = np.zeros(gt.max() + 1, int) - 1
    for v, k in g_idx.items():
        gmap[v] = k
    pmap = np.zeros(pred.max() + 1, int) - 1
    for v, k in p_idx.items():
        pmap[v] = k
    g, p = gmap[gt.ravel()], pmap[pred.ravel()]
    both = (g >= 0) & (p >= 0)
    inter = np.zeros((len(gi), len(pi)))
    np.add.at(inter, (g[both], p[both]), 1)
    ga = np.bincount(g[g >= 0], minlength=len(gi))
    pa = np.bincount(p[p >= 0], minlength=len(pi))
    iou = inter / (ga[:, None] + pa[None, :] - inter + 1e-9)
    r, c = linear_sum_assignment(-iou)
    ok = iou[r, c] >= thr
    tp = int(ok.sum())
    best = {pi[j]: float(iou[:, j].max()) for j in range(len(pi))}
    return tp, len(pi) - tp, len(gi) - tp, list(iou[r, c][ok]), best


def score_set(gts, preds, thr=0.5):
    TP = FP = FN = 0
    ious = []
    for g, p in zip(gts, preds):
        tp, fp, fn, mi, _ = match_instances(g, p, thr)
        TP += tp
        FP += fp
        FN += fn
        ious += mi
    return dict(
        precision=TP / max(TP + FP, 1),
        recall=TP / max(TP + FN, 1),
        f1=2 * TP / max(2 * TP + FP + FN, 1),
        accuracy=TP / max(TP + FP + FN, 1),
        mean_matched_iou=float(np.mean(ious)) if ious else 0.0,
        tp=TP,
        fp=FP,
        fn=FN,
    )
