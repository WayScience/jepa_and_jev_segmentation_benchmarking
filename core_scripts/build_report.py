"""Assemble the demo report as one self-contained HTML page."""

import json
import html
import os
from common import OUT

seg = json.load(open(f"{OUT}/seg_results.json"))
jev = json.load(open(f"{OUT}/jev_results.json"))
probe = json.load(open(f"{OUT}/probe_results.json"))
panels = json.load(open(f"{OUT}/panels.json"))
log = json.load(open(f"{OUT}/jepa_log.json"))


def pm(v, d=2):
    return f"{v[0]:.{d}f}<span class='sd'>±{v[1]:.{d}f}</span>"


# ---------- segmentation table ----------
rows = [
    ("Otsu + watershed", "none", "0 labels", None, seg["otsu"]),
    ("Jev-refined Otsu", "Jev", "0 labels", None, seg["otsu_jev"]),
    ("StarDist", "pretrained", "0 labels", None, seg["stardist"]),
    ("Jev-refined StarDist", "Jev", "0 labels", None, seg["stardist_jev"]),
    ("Cellpose", "pretrained", "0 labels", None, seg["cellpose"]),
    ("Jev-refined Cellpose", "Jev", "0 labels", None, seg["cellpose_jev"]),
    ("Random-init ViT + head", "none", "4", "random_4", None),
    ("JEPA ViT + head", "JEPA, unlabeled", "4", "jepa_4", None),
    ("U-Net from scratch", "none", "4", "unet_4", None),
    ("Random-init ViT + head", "none", "16", "random_16", None),
    ("JEPA ViT + head", "JEPA, unlabeled", "16", "jepa_16", None),
    ("U-Net from scratch", "none", "16", "unet_16", None),
]
trs = []
for name, pre, lab, key, direct in rows:
    if direct:
        cells = [
            f"{direct['f1']:.2f}",
            f"{direct['precision']:.2f}",
            f"{direct['recall']:.2f}",
            f"{direct['mean_matched_iou']:.2f}",
        ]
    else:
        r = seg[key]
        cells = [
            pm(r["f1"]),
            pm(r["precision"]),
            pm(r["recall"]),
            pm(r["mean_matched_iou"]),
        ]
    cls = " class='hl'" if key == "jepa_16" else ""
    trs.append(
        f"<tr{cls}><td>{name}</td><td>{lab}</td>"
        + "".join(f"<td class='num'>{c}</td>" for c in cells)
        + "</tr>"
    )
seg_table = "\n".join(trs)


# ---------- SVG charts (colors from CSS tokens) ----------
def triage_svg():
    c = jev["curve"]
    W, H, L, R, T, B = 560, 300, 52, 16, 16, 44
    xs = c["budget"]
    xmax = 0.5
    X = lambda v: L + (W - L - R) * v / xmax
    Y = lambda v: T + (H - T - B) * (1 - (v - 0.4) / 0.5)
    g = []
    for yv in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        g.append(
            f"<line x1='{L}' x2='{W - R}' y1='{Y(yv):.1f}' y2='{Y(yv):.1f}' class='grid'/><text x='{L - 8}' y='{Y(yv) + 4:.1f}' class='tick' text-anchor='end'>{yv:.1f}</text>"
        )
    for xv in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        g.append(
            f"<text x='{X(xv):.1f}' y='{H - B + 18}' class='tick' text-anchor='middle'>{int(xv * 100)}%</text>"
        )
    g.append(
        f"<text x='{(L + W - R) / 2}' y='{H - 6}' class='axis' text-anchor='middle'>Share of masks sent to a human, lowest confidence first</text>"
    )
    g.append(
        f"<text transform='translate(14 {(T + H - B) / 2}) rotate(-90)' class='axis' text-anchor='middle'>Instance accuracy after fixes</text>"
    )
    series = [
        ("random", "s-rand", "Random order"),
        ("naive", "s-naive", "Segmenter's own confidence"),
        ("jev", "s-jev", "Decision layer P(ok)"),
    ]
    for k, cls, lab in series:
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, c[k]))
        g.append(f"<polyline points='{pts}' class='line {cls}'/>")
        for x, y in zip(xs, c[k]):
            g.append(
                f"<circle cx='{X(x):.1f}' cy='{Y(y):.1f}' r='3' class='dot {cls}'/>"
            )
    # end labels
    ends = sorted([(c[k][-1], cls, lab) for k, cls, lab in series])
    lastY = 999
    for v, cls, lab in sorted(ends, reverse=True):
        y = min(Y(v) + 4, lastY - 14) if lastY < 999 else Y(v) + 4
        lastY = y
    # put labels as legend instead (cleaner at phone width)
    return f"<svg viewBox='0 0 {W} {H}' role='img' aria-label='Triage curve'>{''.join(g)}</svg>"


def reliability_svg():
    W, H, L, R, T, B = 300, 300, 46, 12, 12, 44
    X = lambda v: L + (W - L - R) * v
    Y = lambda v: T + (H - T - B) * (1 - v)
    g = []
    for v in [0, 0.25, 0.5, 0.75, 1]:
        g.append(
            f"<line x1='{X(0)}' x2='{X(1)}' y1='{Y(v):.1f}' y2='{Y(v):.1f}' class='grid'/><text x='{L - 6}' y='{Y(v) + 4:.1f}' class='tick' text-anchor='end'>{v:g}</text>"
        )
        g.append(
            f"<text x='{X(v):.1f}' y='{H - B + 16}' class='tick' text-anchor='middle'>{v:g}</text>"
        )
    g.append(f"<line x1='{X(0)}' y1='{Y(0)}' x2='{X(1)}' y2='{Y(1)}' class='diag'/>")
    g.append(
        f"<text x='{(L + W - R) / 2}' y='{H - 6}' class='axis' text-anchor='middle'>Stated P(mask is right)</text>"
    )
    g.append(
        f"<text transform='translate(12 {(T + H - B) / 2}) rotate(-90)' class='axis' text-anchor='middle'>Observed rate</text>"
    )
    for key, cls in [("reliability_naive", "s-naive"), ("reliability_jev", "s-jev")]:
        bins = [b for b in jev[key] if b["n"] >= 10]
        pts = " ".join(f"{X(b['conf']):.1f},{Y(b['acc']):.1f}" for b in bins)
        g.append(f"<polyline points='{pts}' class='line {cls}'/>")
        for b in bins:
            r = 2.5 + min(5, b["n"] ** 0.5 / 7)
            g.append(
                f"<circle cx='{X(b['conf']):.1f}' cy='{Y(b['acc']):.1f}' r='{r:.1f}' class='dot {cls}'><title>{b['n']} masks: stated {b['conf']:.2f}, observed {b['acc']:.2f}</title></circle>"
            )
    return f"<svg viewBox='0 0 {W} {H}' role='img' aria-label='Calibration'>{''.join(g)}</svg>"


def loss_svg():
    W, H, L, R, T, B = 300, 170, 40, 10, 10, 34
    st = [r["step"] for r in log]
    pl = [r["pred_loss"] for r in log]
    fs = [r["feat_std"] for r in log]
    X = lambda v: L + (W - L - R) * v / st[-1]
    Y = lambda v: T + (H - T - B) * (1 - v / 1.0)
    g = []
    for v in [0, 0.5, 1.0]:
        g.append(
            f"<line x1='{L}' x2='{W - R}' y1='{Y(v):.1f}' y2='{Y(v):.1f}' class='grid'/><text x='{L - 6}' y='{Y(v) + 4:.1f}' class='tick' text-anchor='end'>{v:g}</text>"
        )
    for v in [0, 600, 1200]:
        g.append(
            f"<text x='{X(v):.1f}' y='{H - B + 15}' class='tick' text-anchor='middle'>{v}</text>"
        )
    g.append(
        f"<text x='{(L + W - R) / 2}' y='{H - 4}' class='axis' text-anchor='middle'>Pretraining step</text>"
    )
    g.append(
        "<polyline points='"
        + " ".join(f"{X(s):.1f},{Y(min(v, 1)):.1f}" for s, v in zip(st, pl))
        + "' class='line s-jev'/>"
    )
    g.append(
        "<polyline points='"
        + " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in zip(st, fs))
        + "' class='line s-naive'/>"
    )
    return f"<svg viewBox='0 0 {W} {H}' role='img' aria-label='JEPA pretraining curves'>{''.join(g)}</svg>"


# ---------- panels ----------
image_panels = [p for p in panels if "idx" in p]
pan_html = []
for p in image_panels:
    c = p["counts"]
    pan_html.append(f"""
<figure class="panel">
  <figcaption><span class="mono">{html.escape(p["name"][:18])}…</span> · {p["shape"][0]}×{p["shape"][1]} px · {p["n_gt"]} nuclei</figcaption>
  <div class="strip">
    <div><img src="{p["raw"]}" alt="raw image"><span>Raw</span></div>
    <div><img src="{p["gt"]}" alt="ground truth"><span>Ground truth</span></div>
    <div><img src="{p["jepa"]}" alt="JEPA mask"><span>JEPA · IoU {p["matched_iou_jepa"]:.2f}</span></div>
    <div><img src="{p["refined"]}" alt="Jev-refined JEPA mask"><span>Jev JEPA · IoU {p["matched_iou_jepa_refined"]:.2f}</span></div>
    <div><img src="{p["random"]}" alt="random-init mask"><span>Random · IoU {p["matched_iou_random"]:.2f}</span></div>
    <div><img src="{p["random_refined"]}" alt="Jev-refined random-init mask"><span>Jev Random · IoU {p["matched_iou_random_refined"]:.2f}</span></div>
    <div><img src="{p["unet"]}" alt="U-Net mask"><span>U-Net · IoU {p["matched_iou_unet"]:.2f}</span></div>
    <div><img src="{p["unet_refined"]}" alt="Jev-refined U-Net mask"><span>Jev U-Net · IoU {p["matched_iou_unet_refined"]:.2f}</span></div>
    <div><img src="{p["otsu"]}" alt="Otsu mask"><span>Otsu · IoU {p["method_scores"]["otsu"]:.2f}</span></div>
    <div><img src="{p["otsu_jev"]}" alt="Jev-refined Otsu mask"><span>Jev Otsu · IoU {p["method_scores"]["otsu_jev"]:.2f}</span></div>
    <div><img src="{p["stardist"]}" alt="StarDist mask"><span>StarDist · IoU {p["method_scores"]["stardist"]:.2f}</span></div>
    <div><img src="{p["stardist_jev"]}" alt="Jev-refined StarDist mask"><span>Jev StarDist · IoU {p["method_scores"]["stardist_jev"]:.2f}</span></div>
    <div><img src="{p["cellpose"]}" alt="Cellpose mask"><span>Cellpose · IoU {p["method_scores"]["cellpose"]:.2f}</span></div>
    <div><img src="{p["cellpose_jev"]}" alt="Jev-refined Cellpose mask"><span>Jev Cellpose · IoU {p["method_scores"]["cellpose_jev"]:.2f}</span></div>
    <div><img src="{p["confidence_overlay"]}" alt="Jev confidence overlay"><span>Jev P(ok) mean {p["confidence_mean"]:.2f}</span></div>
  </div>
  <div class="strip">
    <div><img src="{p["error_jepa"]}" alt="JEPA pixel errors"><span>JEPA errors</span></div>
    <div><img src="{p["error_random"]}" alt="random-init pixel errors"><span>Random errors</span></div>
    <div><img src="{p["error_unet"]}" alt="U-Net pixel errors"><span>U-Net errors</span></div>
    <div><img src="{p["changed_jepa"]}" alt="JEPA changes after Jev"><span>JEPA changes after Jev</span></div>
    <div><img src="{p["changed_random"]}" alt="random-init changes after Jev"><span>Random changes after Jev</span></div>
    <div><img src="{p["changed_unet"]}" alt="U-Net changes after Jev"><span>U-Net changes after Jev</span></div>
    <div><span class="muted">Error maps: green overlap, red false positive, blue missed foreground. Change maps: green added, red removed.</span></div>
  </div>
</figure>""")
pan_html = "\n".join(pan_html)
confidence_panels = [p for p in image_panels if p["confidence_example"]]
confidence_html = "\n".join(
    f"""<figure class="panel">
  <figcaption>{p["confidence_label"]} · {html.escape(p["name"])} · mean P(ok) {p["confidence_mean"]:.2f}</figcaption>
  <div class="strip">
    <div><img src="{p["raw"]}" alt="raw image"><span>Raw</span></div>
    <div><img src="{p["gt"]}" alt="ground truth"><span>Ground truth</span></div>
    <div><img src="{p["jepa"]}" alt="JEPA prediction"><span>JEPA prediction</span></div>
    <div><img src="{p["random"]}" alt="random-init prediction"><span>Random-init prediction</span></div>
    <div><img src="{p["unet"]}" alt="U-Net prediction"><span>U-Net prediction</span></div>
    <div><img src="{p["confidence_overlay"]}" alt="Jev confidence"><span>Blue-green is higher confidence</span></div>
  </div>
</figure>"""
    for p in confidence_panels
)
pca_pair = next(p for p in image_panels if p["shape"] == [256, 256])
embedding = next(p["embedding"] for p in panels if "embedding" in p)
iou_plot = next(p["iou_plot"] for p in panels if "iou_plot" in p)
iou_rows = []
for p in image_panels:
    iou_rows.append(
        f"<tr><td class='mono'>{html.escape(p['name'])}</td>"
        f"<td class='num'>{p['matched_iou_jepa']:.3f}</td><td class='num'>{p['matched_iou_jepa_refined']:.3f}</td>"
        f"<td class='num'>{p['matched_iou_random']:.3f}</td>"
        f"<td class='num'>{p['matched_iou_unet']:.3f}</td>"
        f"<td class='num'>{p['tp_jepa']}/{p['fp_jepa']}/{p['fn_jepa']}</td></tr>"
    )

# ---------- example request/response ----------
ex = jev["examples"]


def clean_resp(r):
    return {
        "error_type": r["error_type"],
        "single_complete_nucleus": r["single_complete_nucleus"],
        "verdict": {"choice": r["verdict"]["choice"]},
    }


ex_req = json.dumps(ex[1]["request"], indent=2)
ex_resp = json.dumps(clean_resp(ex[1]["response"]), indent=2)
ex_mid = json.dumps(clean_resp(ex[2]["response"]), indent=2)

b = jev["bands"]
cur = jev["curve"]
i20 = cur["budget"].index(0.2)
i10 = cur["budget"].index(0.1)
mix = jev["test_label_mix"]

template = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "report_template.html"
)
page = open(template).read()
subs = {
    "SEG_TABLE": seg_table,
    "TRIAGE_SVG": triage_svg(),
    "REL_SVG": reliability_svg(),
    "LOSS_SVG": loss_svg(),
    "PANELS": pan_html,
    "CONFIDENCE_EXAMPLES": confidence_html,
    "EMBEDDING": embedding,
    "IOU_PLOT": iou_plot,
    "IOU_ROWS": "\n".join(iou_rows),
    "PCA_JEPA": pca_pair["pca_jepa"],
    "PCA_RANDOM": pca_pair["pca_random"],
    "PCA_RAW": pca_pair["raw"],
    "EX_REQ": html.escape(ex_req),
    "EX_RESP": html.escape(ex_resp),
    "EX_MID": html.escape(ex_mid),
    "EX_MID_TRUTH": ex[2]["truth"],
    "N_TEST_OBJ": f"{jev['n_test_objects']:,}",
    "N_QC_OBJ": f"{jev['n_qc_objects']:,}",
    "ERR_RATE": f"{jev['test_error_rate'] * 100:.0f}%",
    "AUROC_J": f"{jev['auroc_jev']:.2f}",
    "AUROC_N": f"{jev['auroc_naive']:.2f}",
    "ECE_J": f"{jev['ece_jev']:.3f}",
    "ECE_N": f"{jev['ece_naive']:.2f}",
    "BRIER_J": f"{jev['brier_jev']:.3f}",
    "BRIER_N": f"{jev['brier_naive']:.3f}",
    "ETYPE_ACC": f"{jev['error_type_acc_on_wrong'] * 100:.0f}%",
    "N_WRONG": str(jev["n_wrong"]),
    "BASE_ACC": f"{jev['base_accuracy']:.2f}",
    "ACC20_J": f"{cur['jev'][i20]:.2f}",
    "ACC20_R": f"{cur['random'][i20]:.2f}",
    "ACC20_N": f"{cur['naive'][i20]:.2f}",
    "ACC10_J": f"{cur['jev'][i10]:.2f}",
    "CAUGHT20": f"{cur['errors_caught_jev'][i20] * 100:.0f}%",
    "CAUGHT20_R": f"{cur['errors_caught_random'][i20] * 100:.0f}%",
    "ACC_N": str(b["accept"]["n"]),
    "ACC_F": f"{b['accept']['frac'] * 100:.0f}%",
    "ACC_P": f"{b['accept']['precision_ok'] * 100:.1f}%",
    "REV_N": str(b["review"]["n"]),
    "REV_F": f"{b['review']['frac'] * 100:.0f}%",
    "REV_P": f"{b['review']['precision_ok'] * 100:.0f}%",
    "REJ_N": str(b["reject"]["n"]),
    "REJ_F": f"{b['reject']['frac'] * 100:.0f}%",
    "REJ_P": f"{b['reject']['precision_ok'] * 100:.0f}%",
    "MIX_MERGED": str(mix["merged"]),
    "MIX_FRAG": str(mix["fragment"]),
    "MIX_FP": str(mix["false_positive"]),
    "MIX_PB": str(mix["poor_boundary"]),
    "PROBE_J4": pm(probe["jepa_4"]),
    "PROBE_R4": pm(probe["random_4"]),
    "PROBE_J16": pm(probe["jepa_16"]),
    "PROBE_R16": pm(probe["random_16"]),
    "JF16": f"{seg['jepa_16']['f1'][0]:.2f}",
    "RF16": f"{seg['random_16']['f1'][0]:.2f}",
    "UF16": f"{seg['unet_16']['f1'][0]:.2f}",
    "JF4": f"{seg['jepa_4']['f1'][0]:.2f}",
    "RF4": f"{seg['random_4']['f1'][0]:.2f}",
    "UF4": f"{seg['unet_4']['f1'][0]:.2f}",
    "OF": f"{seg['otsu']['f1']:.2f}",
}
for k, v in subs.items():
    page = page.replace("{{" + k + "}}", v)
assert "{{" not in page, [l for l in page.splitlines() if "{{" in l][:5]
open(os.path.join(OUT, "report.html"), "w").write(page)
print("ok", len(page) // 1024, "KB")
