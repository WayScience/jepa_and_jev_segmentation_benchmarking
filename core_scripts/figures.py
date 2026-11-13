"""Render image panels as base64 PNGs for the report."""

import base64
import io
import json
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from sklearn.decomposition import PCA
from umap import UMAP
import torch
import seg
import jev_layer
from common import OUT


def png(arr):
    buf = io.BytesIO()
    plt.imsave(buf, np.clip(arr, 0, 1), format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def gray3(a):
    return np.repeat(a[..., None], 3, -1)


def overlay(img, lab, colors=None):
    out = gray3(img**0.8) * 0.85
    b = find_boundaries(lab, mode="inner")
    if colors is None:
        out[b] = [0.25, 0.85, 0.95]
    else:
        for l, c in colors.items():
            out[b & (lab == l)] = c
    return out


def pca_rgb(fmap, ref_pca=None):
    C, h, w = fmap.shape
    X = fmap.reshape(C, -1).T.float().numpy()
    p = ref_pca or PCA(3).fit(X)
    Y = p.transform(X)
    Y = (Y - np.percentile(Y, 2, 0)) / (
        np.percentile(Y, 98, 0) - np.percentile(Y, 2, 0) + 1e-6
    )
    return np.clip(Y.reshape(h, w, 3), 0, 1), p


def patch_classes(mask):
    h, w = mask.shape[0] // 8, mask.shape[1] // 8
    blocks = mask[: h * 8, : w * 8].reshape(h, 8, w, 8).transpose(0, 2, 1, 3)
    labels = (blocks > 0).mean((2, 3)).ravel()
    return np.where(labels > 0.5, 1, 0)


def object_confidence(img, prob, lab, model, ctx="demo"):
    values = {}
    for state in jev_layer.object_states(img, prob, lab):
        response = model.answer(jev_layer.build_request(state, ctx))
        values[state["label"]] = float(response["single_complete_nucleus"]["noul"])
    return values


def confidence_overlay(img, lab, values):
    out = gray3(img**0.8) * 0.85
    cmap = plt.get_cmap("viridis")
    for label, confidence in values.items():
        boundary = find_boundaries(lab == label, mode="inner")
        out[boundary] = cmap(np.clip(confidence, 0, 1))[:3]
    return out


def error_overlay(gt, pred):
    out = gray3(gt > 0) * 0.18
    gt_fg, pred_fg = gt > 0, pred > 0
    out[gt_fg & ~pred_fg] = [0.2, 0.35, 1.0]
    out[~gt_fg & pred_fg] = [1.0, 0.2, 0.2]
    out[gt_fg & pred_fg] = [0.2, 0.8, 0.35]
    return out


def change_overlay(before, after):
    out = gray3(before > 0) * 0.18
    added = (after > 0) & (before == 0)
    removed = (before > 0) & (after == 0)
    out[added] = [0.2, 0.8, 0.35]
    out[removed] = [1.0, 0.2, 0.2]
    return out


def iou_plot(gts, predictions):
    scores = {name: [] for name in predictions}
    for gt, per_model in zip(gts, zip(*predictions.values())):
        for name, pred in zip(predictions, per_model):
            scores[name].append(seg.score_set([gt], [pred])["mean_matched_iou"])
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    positions = np.arange(len(scores))
    ax.boxplot(
        list(scores.values()),
        positions=positions,
        widths=0.55,
        patch_artist=True,
        boxprops={"facecolor": "#dbeafe"},
        medianprops={"color": "#1f5fbf", "linewidth": 2},
    )
    for j, (name, values) in enumerate(scores.items()):
        x = np.random.RandomState(7 + j).normal(j, 0.06, len(values))
        ax.scatter(x, values, s=18, alpha=0.45, label=name)
    ax.set_xticks(
        positions, list(scores), rotation=45, ha="right", rotation_mode="anchor"
    )
    ax.tick_params(axis="x", labelsize=8, pad=2)
    ax.set_ylabel("Matched instance IoU")
    ax.set_ylim(0, 1)
    ax.set_title("Per-image IoU against ground truth")
    ax.grid(axis="y", alpha=0.25)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def embedding_plot(enc_j, enc_r, images, masks, probs, model):
    Xj, Xr, C = [], [], []
    rng = np.random.RandomState(7)
    for img, mask, prob in zip(images, masks, probs):
        x = torch.from_numpy(img)[None, None]
        with torch.no_grad():
            Xj.append(
                enc_j.feature_map(x)[0].permute(1, 2, 0).reshape(-1, enc_j.dim).numpy()
            )
            Xr.append(
                enc_r.feature_map(x)[0].permute(1, 2, 0).reshape(-1, enc_r.dim).numpy()
            )
        lab = seg.instances(prob)
        values = object_confidence(img, prob, lab, model)
        token_labels = lab[
            seg.PATCH // 2 :: seg.PATCH, seg.PATCH // 2 :: seg.PATCH
        ].ravel()
        C.append(np.array([values.get(int(label), np.nan) for label in token_labels]))
    Xj, Xr, C = np.concatenate(Xj), np.concatenate(Xr), np.concatenate(C)
    keep = np.flatnonzero(np.isfinite(C))
    keep = rng.choice(keep, min(3000, len(keep)), replace=False)
    C, Xj, Xr = C[keep], Xj[keep], Xr[keep]
    pcaj = PCA(2, random_state=7).fit_transform(Xj)
    pcar = PCA(2, random_state=7).fit_transform(Xr)
    umapj = UMAP(
        n_components=2, n_neighbors=20, min_dist=0.15, random_state=7
    ).fit_transform(Xj)
    umapr = UMAP(
        n_components=2, n_neighbors=20, min_dist=0.15, random_state=7
    ).fit_transform(Xr)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    for ax, z, title in zip(
        axes.flat,
        [pcaj, pcar, umapj, umapr],
        ["JEPA · PCA", "Random init · PCA", "JEPA · UMAP", "Random init · UMAP"],
    ):
        ax.scatter(
            z[:, 0], z[:, 1], s=7, alpha=0.55, c=C, cmap="viridis", vmin=0, vmax=1
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="viridis"),
        ax=axes,
        label="Jev confidence, P(ok)",
        shrink=0.75,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


if __name__ == "__main__":
    d = seg.load_preds("jepa16")
    dr = seg.load_preds("random16")
    du = seg.load_preds("unet16")
    base_labels = {
        name: [x.astype(np.int32) for x in seg.load_preds(name)["te_lab"]]
        for name in ("otsu", "stardist", "cellpose")
    }
    refined_labels = {
        name: [x.astype(np.int32) for x in seg.load_preds(f"{name}_jev")["te_lab"]]
        for name in ("otsu", "stardist", "cellpose")
    }
    picks = (
        [int(x) for x in json.load(open(f"{OUT}/fig_picks.json"))] if False else None
    )
    # Fit the QC model on the held-out review history.
    sizes = [(i, len(np.unique(m)) - 1) for i, m in enumerate(seg.te_masks)]
    qc_idx = d["qc_idx"]
    ctx = "demo"
    qc_rows = jev_layer.collect(
        [seg.tr_imgs[i] for i in qc_idx],
        [seg.tr_masks[i] for i in qc_idx],
        d["qc_prob"],
        lambda k: ctx,
    )
    jev = jev_layer.LocalStandIn().fit(
        [r["state"] for r in qc_rows], [r["label"] for r in qc_rows]
    )
    VCOL = {
        "accept": [0.35, 0.85, 0.45],
        "review": [1.0, 0.75, 0.15],
        "reject": [1.0, 0.3, 0.3],
    }

    enc_j, enc_r = seg.load_encoder("jepa"), seg.load_encoder("random")
    confidence_stats = []
    confidence_values = {}
    for i, (img, prob) in enumerate(zip(seg.te_imgs, d["te_prob"])):
        prob = prob.astype(np.float32)
        lab = seg.instances(prob)
        values = object_confidence(img, prob, lab, jev, ctx)
        confidence_values[i] = values
        confidence_stats.append(
            (i, float(np.mean(list(values.values()))) if values else 0.0)
        )
    confidence_order = np.argsort([value for _, value in confidence_stats])
    low_confidence_idx = int(confidence_stats[confidence_order[0]][0])
    high_confidence_idx = int(confidence_stats[confidence_order[-1]][0])
    sc = [
        seg.score_set([gt], [seg.instances(p.astype(np.float32))])["accuracy"]
        for gt, p in zip(seg.te_masks, d["te_prob"])
    ]
    order = np.argsort(sc)
    base_picks = [
        int(order[int(q * (len(order) - 1))]) for q in (0.05, 0.2, 0.35, 0.5, 0.7, 0.9)
    ]
    picks = list(dict.fromkeys(base_picks + [low_confidence_idx, high_confidence_idx]))
    panels = []
    for i in picks:
        img, gt = seg.te_imgs[i], seg.te_masks[i]
        pj = d["te_prob"][i].astype(np.float32)
        labj = seg.instances(pj)
        pr = dr["te_prob"][i].astype(np.float32)
        labr = seg.instances(pr)
        labu = seg.instances(du["te_prob"][i].astype(np.float32))
        method_labels = {
            "otsu": base_labels["otsu"][i],
            "stardist": base_labels["stardist"][i],
            "cellpose": base_labels["cellpose"][i],
            "otsu_jev": refined_labels["otsu"][i],
            "stardist_jev": refined_labels["stardist"][i],
            "cellpose_jev": refined_labels["cellpose"][i],
        }
        values = confidence_values[i]
        refined_labj, refinement_history = jev_layer.refine_with_jev(img, pj, jev, ctx)
        refined_labr, _ = jev_layer.refine_with_jev(img, pr, jev, "random-init ViT")
        refined_labu, _ = jev_layer.refine_with_jev(
            img, du["te_prob"][i].astype(np.float32), jev, "U-Net"
        )
        states = jev_layer.object_states(img, pj, labj)
        cols, counts = {}, {"accept": 0, "review": 0, "reject": 0}
        for s in states:
            v = jev.answer(jev_layer.build_request(s, ctx))["verdict"]["choice"]
            cols[s["label"]] = VCOL[v]
            counts[v] += 1
        x = torch.from_numpy(img)[None, None]
        x0 = torch.zeros_like(x) + float(
            np.median(img)
        )  # blank image: positional response
        fj, pca = pca_rgb(enc_j.feature_map(x)[0] - enc_j.feature_map(x0)[0])
        fr, _ = pca_rgb(enc_r.feature_map(x)[0] - enc_r.feature_map(x0)[0])
        up = lambda a: np.kron(a, np.ones((8, 8, 1)))
        sj = seg.score_set([gt], [labj])
        sf = seg.score_set([gt], [refined_labj])
        sr = seg.score_set([gt], [labr])
        su = seg.score_set([gt], [labu])
        srr = seg.score_set([gt], [refined_labr])
        sur = seg.score_set([gt], [refined_labu])
        method_scores = {
            name: seg.score_set([gt], [lab])["mean_matched_iou"]
            for name, lab in method_labels.items()
        }
        pixel_iou_j = float(
            np.logical_and(gt > 0, labj > 0).sum()
            / max(np.logical_or(gt > 0, labj > 0).sum(), 1)
        )
        pixel_iou_u = float(
            np.logical_and(gt > 0, labu > 0).sum()
            / max(np.logical_or(gt > 0, labu > 0).sum(), 1)
        )
        panels.append(
            dict(
                idx=i,
                name=seg.te_names[i],
                shape=list(img.shape),
                n_gt=int(len(np.unique(gt)) - 1),
                acc_jepa=sj["accuracy"],
                acc_unet=su["accuracy"],
                acc_jepa_refined=sf["accuracy"],
                acc_random=sr["accuracy"],
                matched_iou_jepa=sj["mean_matched_iou"],
                matched_iou_random=sr["mean_matched_iou"],
                matched_iou_unet=su["mean_matched_iou"],
                matched_iou_jepa_refined=sf["mean_matched_iou"],
                matched_iou_random_refined=srr["mean_matched_iou"],
                matched_iou_unet_refined=sur["mean_matched_iou"],
                pixel_iou_jepa=pixel_iou_j,
                pixel_iou_unet=pixel_iou_u,
                tp_jepa=sj["tp"],
                fp_jepa=sj["fp"],
                fn_jepa=sj["fn"],
                counts=counts,
                confidence_mean=float(np.mean(list(values.values())))
                if values
                else 0.0,
                refined_confidence=max(x["confidence"] for x in refinement_history),
                refinement_history=refinement_history,
                confidence_example=i in (low_confidence_idx, high_confidence_idx),
                confidence_label="Low Jev confidence"
                if i == low_confidence_idx
                else ("High Jev confidence" if i == high_confidence_idx else ""),
                pixel_iou_random=float(
                    np.logical_and(gt > 0, labr > 0).sum()
                    / max(np.logical_or(gt > 0, labr > 0).sum(), 1)
                ),
                raw=png(gray3(img)),
                gt=png(overlay(img, gt)),
                jepa=png(overlay(img, labj)),
                random=png(overlay(img, labr)),
                unet=png(overlay(img, labu)),
                error_jepa=png(error_overlay(gt, labj)),
                error_random=png(error_overlay(gt, labr)),
                error_unet=png(error_overlay(gt, labu)),
                confidence_overlay=png(confidence_overlay(img, labj, values)),
                refined=png(overlay(img, refined_labj)),
                random_refined=png(overlay(img, refined_labr)),
                unet_refined=png(overlay(img, refined_labu)),
                changed_jepa=png(change_overlay(labj, refined_labj)),
                changed_random=png(change_overlay(labr, refined_labr)),
                changed_unet=png(change_overlay(labu, refined_labu)),
                otsu=png(overlay(img, method_labels["otsu"])),
                otsu_jev=png(overlay(img, method_labels["otsu_jev"])),
                stardist=png(overlay(img, method_labels["stardist"])),
                stardist_jev=png(overlay(img, method_labels["stardist_jev"])),
                cellpose=png(overlay(img, method_labels["cellpose"])),
                cellpose_jev=png(overlay(img, method_labels["cellpose_jev"])),
                method_scores=method_scores,
                pca_jepa=png(up(fj)),
                pca_random=png(up(fr)),
                verdict=png(overlay(img, labj, cols)),
            )
        )
        print("panel", i, counts, flush=True)
    panels.append(
        {
            "embedding": embedding_plot(
                enc_j, enc_r, seg.te_imgs[:8], seg.te_masks[:8], d["te_prob"][:8], jev
            )
        }
    )
    iou_predictions = {
        "JEPA": [seg.instances(p.astype(np.float32)) for p in d["te_prob"]],
        "Jev-refined JEPA": [
            jev_layer.refine_with_jev(img, p.astype(np.float32), jev, ctx)[0]
            for img, p in zip(seg.te_imgs, d["te_prob"])
        ],
        "Random init": [seg.instances(p.astype(np.float32)) for p in dr["te_prob"]],
        "Jev-refined Random init": [
            jev_layer.refine_with_jev(
                img, p.astype(np.float32), jev, "random-init ViT"
            )[0]
            for img, p in zip(seg.te_imgs, dr["te_prob"])
        ],
        "U-Net": [seg.instances(p.astype(np.float32)) for p in du["te_prob"]],
        "Jev-refined U-Net": [
            jev_layer.refine_with_jev(img, p.astype(np.float32), jev, "U-Net")[0]
            for img, p in zip(seg.te_imgs, du["te_prob"])
        ],
    }
    for name in ("otsu", "stardist", "cellpose"):
        iou_predictions[name.title()] = base_labels[name]
        iou_predictions[f"Jev-refined {name.title()}"] = refined_labels[name]
    panels.append(
        {
            "iou_plot": iou_plot(
                seg.te_masks,
                iou_predictions,
            )
        }
    )
    json.dump(panels, open(f"{OUT}/panels.json", "w"))
