"""Tiny I-JEPA pretraining on unlabeled nuclei images (CPU-sized).

Context encoder sees a large block minus target blocks; a narrow predictor
predicts the EMA target encoder's representations of 4 target blocks.
Loss is in embedding space (no pixel reconstruction).
"""

import copy
import math
import random
import sys
import time
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import ViT, Block, sincos_2d, load_split, OUT, PATCH

torch.set_num_threads(2)
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

CROP = 128
G = CROP // PATCH  # 16x16 token grid
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
BS = 32
CHECKPOINT = f"{OUT}/jepa_checkpoint.pt"
FINAL = f"{OUT}/jepa.pt"


def atomic_torch_save(value, path):
    tmp = f"{path}.tmp"
    torch.save(value, tmp)
    os.replace(tmp, path)


imgs, _, _ = load_split("train")  # masks NOT used


def sample_batch():
    out = np.empty((BS, 1, CROP, CROP), np.float32)
    for i in range(BS):
        a = imgs[random.randrange(len(imgs))]
        y = random.randrange(a.shape[0] - CROP + 1)
        x = random.randrange(a.shape[1] - CROP + 1)
        c = a[y : y + CROP, x : x + CROP]
        c = np.rot90(c, random.randrange(4))
        if random.random() < 0.5:
            c = c[:, ::-1]
        c = c * random.uniform(0.7, 1.3) + random.uniform(
            -0.1, 0.1
        )  # mild intensity jitter
        out[i, 0] = c
    return torch.from_numpy(out)


def block(scale, aspect):
    area = G * G * random.uniform(*scale)
    ar = math.exp(random.uniform(math.log(aspect[0]), math.log(aspect[1])))
    h = max(1, min(G, int(round(math.sqrt(area * ar)))))
    w = max(1, min(G, int(round(math.sqrt(area / ar)))))
    y = random.randrange(G - h + 1)
    x = random.randrange(G - w + 1)
    m = np.zeros((G, G), bool)
    m[y : y + h, x : x + w] = True
    return m


def make_masks():
    targets = [np.flatnonzero(block((0.15, 0.2), (0.75, 1.5))) for _ in range(4)]
    ctx = block((0.85, 1.0), (1, 1))
    for t in targets:
        ctx.flat[t] = False
    ctx_idx = np.flatnonzero(ctx)
    if len(ctx_idx) < 20:
        return make_masks()
    return torch.from_numpy(ctx_idx), [torch.from_numpy(t) for t in targets]


class Predictor(nn.Module):
    def __init__(self, enc_dim=192, dim=96, depth=3, heads=3):
        super().__init__()
        self.dim = dim
        self.inp = nn.Linear(enc_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, enc_dim)

    def forward(self, ctx_tokens, ctx_idx, tgt_idx):
        pos = sincos_2d(G, G, self.dim)
        B = ctx_tokens.shape[0]
        x = self.inp(ctx_tokens) + pos[ctx_idx][None]
        m = self.mask_token.expand(B, len(tgt_idx), -1) + pos[tgt_idx][None]
        x = torch.cat([x, m], 1)
        for b in self.blocks:
            x = b(x)
        return self.out(self.norm(x[:, -len(tgt_idx) :]))


enc = ViT()
tgt = copy.deepcopy(enc).requires_grad_(False)
pred = Predictor()
params = list(enc.parameters()) + list(pred.parameters())
opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.05)
WARM = 200

log = []
t0 = time.time()
start_step = 0
if os.path.exists(FINAL):
    saved = torch.load(FINAL, map_location="cpu")
    if saved.get("step", -1) >= STEPS:
        print(
            f"cached final model found at step {saved['step']}; nothing to do",
            flush=True,
        )
        raise SystemExit(0)
if os.path.exists(CHECKPOINT):
    saved = torch.load(CHECKPOINT, map_location="cpu")
    enc.load_state_dict(saved["context"])
    tgt.load_state_dict(saved["target"])
    pred.load_state_dict(saved["predictor"])
    opt.load_state_dict(saved["optimizer"])
    start_step = saved["step"] + 1
    log = saved.get("log", [])
    print(f"resuming from step {saved['step']}", flush=True)

for step in range(start_step, STEPS + 1):
    lr = (
        5e-4
        * min(1, (step + 1) / WARM)
        * (0.5 * (1 + math.cos(math.pi * step / STEPS)) if step > WARM else 1)
    )
    for g in opt.param_groups:
        g["lr"] = lr
    mom = 0.996 + (1 - 0.996) * step / STEPS

    x = sample_batch()
    ctx_idx, tgts = make_masks()
    with torch.no_grad():
        h = F.layer_norm(tgt(x), (enc.dim,))
    z = enc(x, keep_idx=ctx_idx)
    loss = 0
    for t in tgts:
        p = pred(z, ctx_idx, t)
        loss = loss + F.smooth_l1_loss(p, h[:, t])
    loss = loss / len(tgts)
    # Collapse guard (VICReg-style variance + covariance on context features, in the
    # spirit of LeJEPA's SIGReg): keep every embedding dim spread out across tokens.
    zf = z.reshape(-1, z.shape[-1])
    zc = zf - zf.mean(0)
    std = torch.sqrt(zc.var(0) + 1e-4)
    var_loss = F.relu(1 - std).mean()
    cov = (zc.T @ zc) / (zc.shape[0] - 1)
    cov_loss = (cov - torch.diag(torch.diag(cov))).pow(2).sum() / zf.shape[1]
    pred_loss = loss
    loss = pred_loss + 1.0 * var_loss + 0.04 * cov_loss
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    with torch.no_grad():
        for pt, pe in zip(tgt.parameters(), enc.parameters()):
            pt.mul_(mom).add_(pe.detach(), alpha=1 - mom)
        # collapse monitor: per-dim std of target features
        fstd = h.std(dim=(0, 1)).mean().item()
    if step % 25 == 0:
        log.append(
            dict(
                step=step,
                loss=loss.item(),
                pred_loss=pred_loss.item(),
                var_loss=var_loss.item(),
                cov_loss=cov_loss.item(),
                feat_std=fstd,
                lr=lr,
                sec=time.time() - t0,
            )
        )
        print(json.dumps(log[-1]), flush=True)
    if step % 250 == 0 or step == STEPS:
        atomic_torch_save(
            dict(
                target=tgt.state_dict(),
                context=enc.state_dict(),
                predictor=pred.state_dict(),
                optimizer=opt.state_dict(),
                step=step,
                log=log,
            ),
            CHECKPOINT,
        )
        if step == STEPS:
            atomic_torch_save(
                dict(target=tgt.state_dict(), context=enc.state_dict(), step=step),
                FINAL,
            )
        json.dump(log, open(f"{OUT}/jepa_log.json", "w"))
print("done", time.time() - t0)
