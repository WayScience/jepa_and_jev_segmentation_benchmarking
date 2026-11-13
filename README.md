# JEPA + Jev Nuclei Demo

This repository contains a CPU-only nuclei segmentation demo.
It uses the StarDist DSB2018 subset.

The pipeline has three stages:

1. Train a small I-JEPA model on unlabeled images.
2. Train segmentation heads with 4 or 16 labeled images.
3. Rank predicted masks with a local Jev-shaped quality-control model.

The report includes segmentation examples, matched IoU values, PCA, UMAP, and
quality-control curves. Jev refinement is available for JEPA, Otsu, StarDist,
and Cellpose outputs.

## Run the pipeline

Install `uv`, then run:

```bash
just all
```

The default dataset path is `data/dsb2018`. The dataset must contain:

```text
data/dsb2018/
├── train/images/
├── train/masks/
├── test/images/
└── test/masks/
```

Use another dataset path with:

```bash
JEPA_DATA=/path/to/dsb2018 just all
```

The pipeline writes results to `out/`. Set `JEPA_OUT` to use another output
directory.

## Results

The current run used 1,200 JEPA steps and 50 held-out test images.
The segmentation table reports mean ± standard deviation over three label
selections. A matched prediction has instance IoU of at least 0.5.

| Method | Labels | F1 | Matched IoU |
| --- | ---: | ---: | ---: |
| Otsu + watershed | 0 | 0.69 | 0.79 |
| Random-init ViT + head | 4 | 0.63 ± 0.04 | 0.79 ± 0.02 |
| JEPA ViT + head | 4 | 0.64 ± 0.04 | 0.79 ± 0.02 |
| U-Net | 4 | 0.71 ± 0.04 | 0.82 ± 0.02 |
| Random-init ViT + head | 16 | 0.68 ± 0.03 | 0.80 ± 0.00 |
| JEPA ViT + head | 16 | 0.69 ± 0.02 | 0.80 ± 0.01 |
| U-Net | 16 | 0.73 ± 0.02 | 0.84 ± 0.00 |

The report also includes Otsu, StarDist, and Cellpose results before and after
Jev candidate selection. StarDist and Cellpose use installed pretrained
backends. If a backend is not installed, the pipeline uses a deterministic
watershed proxy and labels the result as a proxy.

To enable the real optional backends, install them into `.venv`:

```bash
uv pip install --python .venv/bin/python stardist cellpose
```

The current generated report used the proxy path because neither optional
package is installed.

The U-Net is the strongest segmentation baseline in this small CPU
experiment. JEPA improves on the random-init ViT at both label budgets, but
the difference is small.

The frozen-token probe gives balanced accuracy of `0.67 ± 0.09` with 4 labels
and `0.78 ± 0.03` with 16 labels for JEPA features. The random-init scores are
`0.69 ± 0.08` and `0.77 ± 0.04`.

The quality-control model scored `0.93` AUROC. The segmenter's confidence
scored `0.56` AUROC. The base instance accuracy was `0.52`. Reviewing the 20%
least-confident masks increased accuracy to `0.79`, compared with `0.58` for
random review.

The report evaluates 1,847 predicted test objects. It identifies 380 wrong
objects. Its error-type classifier is correct for 65% of wrong objects.

The report also colors PCA and UMAP token embeddings by Jev P(ok). It shows
the highest-confidence image at mean P(ok) 0.94 and the lowest-confidence image
at mean P(ok) 0.37. The high-confidence example has JEPA accuracy 0.95. The
low-confidence example has JEPA accuracy 0.29.

These results support Jev as a quality-control layer. Jev ranks bad masks well,
but it does not replace the segmenter or a human reviewer. The measured AUROC
is 0.93, and the 20% review simulation raises instance accuracy from 0.52 to
0.79.

The report also runs a refinement loop for each method. Jev scores the base
mask, cleaned masks, and an Otsu candidate. It keeps the candidate with the
highest mean P(ok). This is a candidate selector, not proof that the selected
mask is correct.

Open the generated report at [out/report.html](out/report.html).

## Cache and resume behavior

JEPA checkpoints are saved to `out/jepa_checkpoint.pt` every 250 steps.
Rerun `just all` after an interruption to resume training.

A completed `out/jepa.pt` is reused when it contains the requested number of
steps. Segmentation, probe, quality-control, figure, and report outputs also
use completion stamps.

Use these commands to control the cache:

```bash
just all             # Run missing stages.
just clean-results   # Remove generated models, metrics, figures, and reports.
just force-all       # Remove results and run every stage again.
FORCE=1 just all     # Run every stage without removing old files first.
```

The clean command does not remove `data/`, `.venv/`, or source files.
The environment install also uses `setup_stamp` and is skipped when it is
already complete.
The clean command removes the pretraining completion stamp with the model.
This prevents a missing `out/jepa.pt` file from being treated as cached.

Prediction caches use compressed `.npz` files. They do not use pickle.
PyTorch model checkpoints use `.pt` files.

## Repository layout

- `core_scripts/`: pipeline code.
- `data/`: local dataset files.
- `out/`: cached models, metrics, figures, and the report.
- `report_template.html`: HTML report template.
- `justfile`: complete pipeline commands.

`jev_layer.LocalStandIn.answer()` is a local calibrated stand-in for Jev.
Replace it with a TypeSafe API call when a Jev integration is available.
