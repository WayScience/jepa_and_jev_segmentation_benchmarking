"""Features-only probe: how much does the frozen JEPA representation know, with no pixel stem?

Each 8x8 patch gets a label: background / inside a nucleus / on a nucleus boundary
(boundary = patch contains boundary pixels or touches >1 instance). A linear classifier
is trained on frozen tokens from N labeled images and scored on all test patches.
"""

import json
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
from skimage.segmentation import find_boundaries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
import seg
from common import OUT, PATCH

torch.set_num_threads(2)


def patch_labels(m):
    H, W = m.shape
    b = find_boundaries(m, mode="inner")
    h, w = H // PATCH, W // PATCH
    mb = m.reshape(h, PATCH, w, PATCH).transpose(0, 2, 1, 3).reshape(h, w, -1)
    bb = b.reshape(h, PATCH, w, PATCH).transpose(0, 2, 1, 3).reshape(h, w, -1)
    fgf = (mb > 0).mean(-1)
    nin = np.array([[len(set(x[x > 0])) for x in row] for row in mb])
    lab = np.zeros((h, w), int)
    lab[fgf > 0.5] = 1
    lab[(bb.mean(-1) > 0.08) | (nin > 1)] = 2
    return lab.ravel()


def tokens(enc, imgs):
    out = []
    with torch.no_grad():
        for a in imgs:
            f = enc.feature_map(torch.from_numpy(a)[None, None])[0]
            out.append(f.reshape(f.shape[0], -1).T.numpy())
    return out


Yte = np.concatenate([patch_labels(m) for m in seg.te_masks])
res = {}
for kind in ["jepa", "random"]:
    enc = seg.load_encoder(kind)
    Ttr = tokens(enc, [seg.tr_imgs[i] for i in seg.POOL])
    Xte = np.concatenate(tokens(enc, seg.te_imgs))
    Ytr_all = [patch_labels(seg.tr_masks[i]) for i in seg.POOL]
    for B in seg.BUDGETS:
        scores = []
        for s in seg.SEEDS:
            pick = np.random.RandomState(1000 + s).choice(
                len(seg.POOL), B, replace=False
            )
            # note: same seeds as seg.py, but indices into POOL directly
            X = np.concatenate([Ttr[i] for i in pick])
            Y = np.concatenate([Ytr_all[i] for i in pick])
            mu, sd = X.mean(0), X.std(0) + 1e-6
            clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced").fit(
                (X - mu) / sd, Y
            )
            scores.append(balanced_accuracy_score(Yte, clf.predict((Xte - mu) / sd)))
        res[f"{kind}_{B}"] = [float(np.mean(scores)), float(np.std(scores))]
        print(kind, B, res[f"{kind}_{B}"], flush=True)
res["test_patch_mix"] = {
    k: int((Yte == i).sum()) for i, k in enumerate(["background", "inside", "boundary"])
}
json.dump(res, open(f"{OUT}/probe_results.json", "w"), indent=1)
