"""Jev-style decision layer for mask QC.

Each predicted nucleus becomes a structured *state*; we ask typed *questions*
(Choice / Noul, mirroring the shape described in docs.typesafe.ai) and get back
calibrated probabilities + confidence. Jev itself is early-access and text/state
based, so `LocalStandIn` answers the same request shape with a calibrated
gradient-boosted model trained on a held-out "QC review history" (60 images).
Swap `LocalStandIn.answer` for a real Jev call to use the real model.
"""

import json
import sys
import warnings
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from skimage.feature import peak_local_max
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore")
import seg
from common import match_instances, score_set, OUT

ERR = ["ok", "merged", "fragment", "false_positive", "poor_boundary"]
VERDICTS = ["accept", "review", "reject"]
FEATS = [
    "area",
    "rel_area",
    "solidity",
    "eccentricity",
    "extent",
    "perimeter_ratio",
    "n_peaks",
    "mean_p_interior",
    "min_p_interior_core",
    "mean_p_boundary_edge",
    "mean_p_fg",
    "edge_contrast",
    "intensity_mean",
    "intensity_cv",
    "touches_border",
    "n_neighbors_touching",
]


def object_states(img, prob, lab):
    """Per-object structured state (what you'd send to Jev)."""
    props = regionprops(lab, intensity_image=img)
    if not props:
        return []
    med_area = np.median([p.area for p in props])
    bnd = find_boundaries(lab, mode="inner")
    dil = ndi.grey_dilation(lab, size=3)
    states = []
    for p in props:
        sl = p.slice
        m = lab[sl] == p.label
        pi, pb, pf = prob[1][sl][m], prob[2][sl], 1 - prob[0][sl]
        edge = bnd[sl] & m
        ring = ndi.binary_dilation(m, iterations=2) & ~m
        core = ndi.binary_erosion(m, iterations=2)
        d = ndi.distance_transform_edt(m)
        n_peaks = len(
            peak_local_max(d, min_distance=3, threshold_abs=2, exclude_border=False)
        )
        neigh = np.unique(dil[sl][ndi.binary_dilation(m) & ~m])
        I = img[sl][m]
        states.append(
            dict(
                label=int(p.label),
                area=float(p.area),
                rel_area=float(p.area / med_area),
                solidity=float(p.solidity),
                eccentricity=float(p.eccentricity),
                extent=float(p.extent),
                perimeter_ratio=float(
                    p.perimeter / (2 * np.sqrt(np.pi * p.area) + 1e-6)
                ),
                n_peaks=int(n_peaks),
                mean_p_interior=float(pi.mean()),
                min_p_interior_core=float(prob[1][sl][core].min())
                if core.any()
                else float(pi.min()),
                mean_p_boundary_edge=float(pb[edge].mean()) if edge.any() else 0.0,
                mean_p_fg=float(pf[m].mean()),
                edge_contrast=float(I.mean() - img[sl][ring].mean())
                if ring.any()
                else 0.0,
                intensity_mean=float(I.mean()),
                intensity_cv=float(I.std() / (I.mean() + 1e-6)),
                touches_border=int(
                    sl[0].start == 0
                    or sl[1].start == 0
                    or sl[0].stop == lab.shape[0]
                    or sl[1].stop == lab.shape[1]
                ),
                n_neighbors_touching=int(
                    len([v for v in neigh if v not in (0, p.label)])
                ),
            )
        )
    return states


def error_labels(gt, lab):
    """Ground-truth error type per predicted object (only used for training/eval)."""
    _, _, _, _, best = match_instances(gt, lab, 0.5)
    out = {}
    for p in [v for v in np.unique(lab) if v]:
        m = lab == p
        if best.get(p, 0) >= 0.5:
            out[p] = "ok"
            continue
        g, c = np.unique(gt[m], return_counts=True)
        cover = {gi: ci for gi, ci in zip(g, c) if gi}
        frac_in_gt = sum(cover.values()) / m.sum()
        big = [gi for gi, ci in cover.items() if ci / (gt == gi).sum() > 0.5]
        if frac_in_gt < 0.3:
            out[p] = "false_positive"
        elif len(big) >= 2:
            out[p] = "merged"
        elif (
            cover
            and max(cover.values()) / (gt == max(cover, key=cover.get)).sum() < 0.6
        ):
            out[p] = "fragment"
        else:
            out[p] = "poor_boundary"
    return out


def build_request(state, context):
    """A Jev-shaped request: unstructured/structured state in, typed questions out."""
    return {
        "state": {
            "context": context,
            "object": {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in state.items()
            },
        },
        "questions": [
            {
                "type": "choice",
                "key": "error_type",
                "prompt": "What best describes this predicted nucleus mask?",
                "options": ERR,
            },
            {
                "type": "choice",
                "key": "verdict",
                "prompt": "Should this mask be accepted, sent to human review, or rejected?",
                "options": VERDICTS,
            },
            {
                "type": "noul",
                "key": "single_complete_nucleus",
                "statement": "This mask outlines exactly one complete nucleus with IoU >= 0.5.",
            },
        ],
    }


class LocalStandIn:
    """Answers Jev-shaped requests with a calibrated classifier. Replace `answer` with a Jev API call."""

    def __init__(self, accept_at=0.85, reject_at=0.3):
        self.accept_at, self.reject_at = accept_at, reject_at

    def fit(self, states, labels):
        X = np.array([[s[f] for f in FEATS] for s in states])
        y = np.array([ERR.index(l) for l in labels])
        self.present = sorted(set(y))
        base = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=0,
        )
        self.clf = CalibratedClassifierCV(base, method="isotonic", cv=5).fit(X, y)
        return self

    def answer(self, request):
        s = request["state"]["object"]
        pr = self.clf.predict_proba(np.array([[s[f] for f in FEATS]]))[0]
        probs = {ERR[c]: float(p) for c, p in zip(self.clf.classes_, pr)}
        for e in ERR:
            probs.setdefault(e, 0.0)
        p_ok = probs["ok"]
        v = (
            "accept"
            if p_ok >= self.accept_at
            else ("reject" if p_ok < self.reject_at else "review")
        )
        # verdict probabilities derived from the calibrated P(ok) bands
        vp = {
            "accept": p_ok if v == "accept" else 0.0,
            "review": 0.0,
            "reject": 1 - p_ok if v == "reject" else 0.0,
        }
        vp["review"] = 1 - vp["accept"] - vp["reject"]
        top = max(probs, key=probs.get)
        return {
            "error_type": {
                "choice": top,
                "probabilities": {k: round(x, 4) for k, x in probs.items()},
                "confidence": round(probs[top], 4),
            },
            "verdict": {
                "choice": v,
                "probabilities": {k: round(x, 4) for k, x in vp.items()},
                "confidence": round(max(vp.values()), 4),
            },
            "single_complete_nucleus": {"noul": round(p_ok, 4)},
        }

    def p_ok(self, states):
        if not states:
            return np.empty(0, dtype=np.float32)
        X = np.array([[s[f] for f in FEATS] for s in states])
        probs = self.clf.predict_proba(X)
        ok_col = list(self.clf.classes_).index(ERR.index("ok"))
        return probs[:, ok_col].astype(np.float32)


def refine_with_jev(img, prob, model, ctx, rounds=2):
    """Try alternate watershed thresholds and keep the mask with the highest mean P(ok)."""
    configs = [
        (0.3, 0.3, 3, 6),
        (0.35, 0.4, 4, 8),
        (0.4, 0.4, 4, 8),
        (0.4, 0.5, 6, 10),
        (0.5, 0.4, 6, 10),
        (0.5, 0.5, 6, 10),
        (0.5, 0.6, 8, 10),
        (0.6, 0.5, 6, 12),
        (0.6, 0.6, 8, 12),
        (0.7, 0.65, 10, 14),
    ]
    history = []
    best_lab = None
    best_conf = -1.0
    for config in configs:
        lab = seg.instances(prob, *config)
        states = object_states(img, prob, lab)
        if not states:
            confidence = 0.0
        else:
            confidence = float(np.mean(model.p_ok(states))) if states else 0.0
        history.append(dict(config=config, confidence=confidence, objects=len(states)))
        if confidence > best_conf:
            best_conf, best_lab = confidence, lab
    states = object_states(img, prob, best_lab)
    if states:
        keep = {
            s["label"]
            for s, p in zip(states, model.p_ok(states))
            if p >= model.reject_at
        }
        filtered = np.where(np.isin(best_lab, list(keep)), best_lab, 0).astype(np.int32)
        filtered_states = object_states(img, prob, filtered)
        filtered_conf = (
            float(np.mean(model.p_ok(filtered_states))) if filtered_states else 0.0
        )
        history.append(
            dict(
                config="jev_filter",
                confidence=filtered_conf,
                objects=len(filtered_states),
            )
        )
        if filtered_conf > best_conf:
            best_lab, best_conf = filtered, filtered_conf
    return best_lab, history


def label_to_prob(img, lab):
    """Make a Jev-compatible probability map from a label-only segmenter."""
    fg = lab > 0
    boundary = find_boundaries(lab, mode="inner")
    interior = ndi.distance_transform_edt(fg).astype(np.float32)
    if interior.max() > 0:
        interior /= interior.max()
    interior = np.clip(interior * (~boundary) + 0.15 * boundary, 0, 1)
    bg = 1.0 - fg.astype(np.float32)
    boundary_p = boundary.astype(np.float32)
    total = bg + interior + boundary_p + 1e-6
    return np.stack([bg / total, interior / total, boundary_p / total])


def refine_labels_with_jev(img, lab, model, ctx):
    """Score label-only candidates with Jev and keep the highest-confidence mask."""
    base = lab.astype(np.int32)
    candidates = [("base", base)]
    for min_size in (5, 10, 20):
        cleaned = base.copy()
        for label in np.unique(cleaned):
            if label and (cleaned == label).sum() < min_size:
                cleaned[cleaned == label] = 0
        candidates.append((f"remove_small_{min_size}", cleaned))
    candidates.append(("otsu_candidate", seg.otsu_instances(img)))
    history = []
    best_lab, best_conf = base, -1.0
    for name, candidate in candidates:
        prob = label_to_prob(img, candidate)
        states = object_states(img, prob, candidate)
        confidence = float(np.mean(model.p_ok(states))) if states else 0.0
        history.append(dict(config=name, confidence=confidence, objects=len(states)))
        if confidence > best_conf:
            best_conf, best_lab = confidence, candidate
    return best_lab, history


def collect(imgs, masks, probs, ctx_fn):
    rows = []
    for k, (img, gt, p) in enumerate(zip(imgs, masks, probs)):
        p = p.astype(np.float32)
        lab = seg.instances(p)
        el = error_labels(gt, lab)
        for s in object_states(img, p, lab):
            rows.append(dict(img=k, state=s, label=el[s["label"]], ctx=ctx_fn(k)))
    return rows


def ece(p, y, bins=10):
    e, out = 0.0, []
    edges = np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b if b < 1 else p <= b)
        if m.sum():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
            out.append(
                dict(
                    lo=float(a),
                    hi=float(b),
                    n=int(m.sum()),
                    conf=float(p[m].mean()),
                    acc=float(y[m].mean()),
                )
            )
    return float(e), out


def corrected_accuracy(rows, masks, review_set):
    """Simulate a reviewer fixing every reviewed object that is wrong.
    Fix = drop the bad mask and paste in the GT nuclei it touched (oracle correction)."""
    preds = {}
    for k in sorted(set(r["img"] for r in rows)):
        preds[k] = None
    out = []
    for k, gt in enumerate(masks):
        lab = LABS[k].copy()
        nxt = lab.max() + 1
        for r in rows:
            if r["img"] != k or id(r) not in review_set or r["label"] == "ok":
                continue
            m = lab == r["state"]["label"]
            touched = [g for g in np.unique(gt[m]) if g]
            lab[m] = 0
            for g in touched:
                gm = (gt == g) & (lab == 0)
                lab[gm] = nxt
                nxt += 1
        out.append(lab)
    return score_set(masks, out)["accuracy"]


if __name__ == "__main__":
    d = seg.load_preds(sys.argv[1] if len(sys.argv) > 1 else "jepa16")
    qc_idx = d["qc_idx"]
    qc_imgs = [seg.tr_imgs[i] for i in qc_idx]
    qc_masks = [seg.tr_masks[i] for i in qc_idx]
    ctx = "Fluorescence nuclei (DSB2018). Mask from JEPA features + 16-label head + watershed."
    qc_rows = collect(qc_imgs, qc_masks, d["qc_prob"], lambda k: ctx)
    te_rows = collect(seg.te_imgs, seg.te_masks, d["te_prob"], lambda k: ctx)
    LABS = [seg.instances(p.astype(np.float32)) for p in d["te_prob"]]
    print("qc objects", len(qc_rows), "test objects", len(te_rows))
    print("qc label mix", {e: sum(r["label"] == e for r in qc_rows) for e in ERR})
    print("test label mix", {e: sum(r["label"] == e for r in te_rows) for e in ERR})

    jev = LocalStandIn().fit(
        [r["state"] for r in qc_rows], [r["label"] for r in qc_rows]
    )
    answers = [jev.answer(build_request(r["state"], r["ctx"])) for r in te_rows]
    p_ok = np.array([a["single_complete_nucleus"]["noul"] for a in answers])
    y_ok = np.array([r["label"] == "ok" for r in te_rows], float)

    # naive confidence: the segmentation network's own mean interior probability
    naive = np.array([r["state"]["mean_p_interior"] for r in te_rows])
    ece_jev, rel_jev = ece(p_ok, y_ok)
    ece_naive, rel_naive = ece(naive, y_ok)

    # error-type accuracy on wrong objects
    wrong = [i for i, r in enumerate(te_rows) if r["label"] != "ok"]
    et_acc = (
        np.mean(
            [answers[i]["error_type"]["choice"] == te_rows[i]["label"] for i in wrong]
        )
        if wrong
        else 0
    )

    # verdict bands
    bands = {}
    for v in VERDICTS:
        idx = [i for i, a in enumerate(answers) if a["verdict"]["choice"] == v]
        bands[v] = dict(
            n=len(idx),
            frac=len(idx) / len(answers),
            precision_ok=float(y_ok[idx].mean()) if idx else None,
        )

    # triage curve: review budget (fraction of objects) -> corrected accuracy
    base_acc = score_set(seg.te_masks, LABS)["accuracy"]
    order_jev = np.argsort(p_ok)
    order_naive = np.argsort(naive)
    rs = np.random.RandomState(0)
    order_rand = rs.permutation(len(te_rows))
    budgets = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    curve = {
        "budget": budgets,
        "jev": [],
        "naive": [],
        "random": [],
        "errors_caught_jev": [],
        "errors_caught_naive": [],
        "errors_caught_random": [],
    }
    n_err = max(1, int((1 - y_ok).sum()))
    for b in budgets:
        k = int(round(b * len(te_rows)))
        for name, order in [
            ("jev", order_jev),
            ("naive", order_naive),
            ("random", order_rand),
        ]:
            sel = set(id(te_rows[i]) for i in order[:k])
            curve[name].append(corrected_accuracy(te_rows, seg.te_masks, sel))
            curve[f"errors_caught_{name}"].append(
                float((1 - y_ok[order[:k]]).sum() / n_err)
            )
        print(
            "budget",
            b,
            {n: round(curve[n][-1], 3) for n in ["jev", "naive", "random"]},
            flush=True,
        )

    ex = []
    for pick in [
        int(np.argmax(p_ok)),
        int(np.argmin(p_ok)),
        int(np.argsort(np.abs(p_ok - 0.6))[0]),
    ]:
        r = te_rows[pick]
        ex.append(
            dict(
                request=build_request(r["state"], r["ctx"]),
                response=answers[pick],
                truth=r["label"],
            )
        )

    res = dict(
        n_qc_objects=len(qc_rows),
        n_test_objects=len(te_rows),
        test_error_rate=float(1 - y_ok.mean()),
        auroc_jev=float(roc_auc_score(1 - y_ok, 1 - p_ok)),
        auroc_naive=float(roc_auc_score(1 - y_ok, 1 - naive)),
        brier_jev=float(brier_score_loss(y_ok, p_ok)),
        brier_naive=float(brier_score_loss(y_ok, naive)),
        ece_jev=ece_jev,
        ece_naive=ece_naive,
        reliability_jev=rel_jev,
        reliability_naive=rel_naive,
        error_type_acc_on_wrong=float(et_acc),
        n_wrong=len(wrong),
        test_label_mix={e: sum(r["label"] == e for r in te_rows) for e in ERR},
        bands=bands,
        base_accuracy=base_acc,
        curve=curve,
        examples=ex,
    )
    json.dump(res, open(f"{OUT}/jev_results.json", "w"), indent=1, default=float)
    seg_results = json.load(open(f"{OUT}/seg_results.json"))
    for method in ("otsu", "stardist", "cellpose"):
        pred = seg.load_preds(method)["te_lab"]
        refined = []
        histories = []
        for img, lab in zip(seg.te_imgs, pred):
            out, history = refine_labels_with_jev(
                img, lab, jev, f"{method} label-only segmenter"
            )
            refined.append(out)
            histories.append(history)
        seg_results[f"{method}_jev"] = seg.score_set(seg.te_masks, refined)
        seg.save_label_preds(f"{method}_jev", refined)
        seg_results[f"{method}_jev"]["history"] = histories
    json.dump(
        seg_results, open(f"{OUT}/seg_results.json", "w"), indent=1, default=float
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in res.items()
                if k
                not in ("curve", "examples", "reliability_jev", "reliability_naive")
            },
            indent=1,
            default=float,
        )
    )
