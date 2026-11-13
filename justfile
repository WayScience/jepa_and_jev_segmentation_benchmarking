set shell := ["bash", "-cu"]

steps := "1200"
uv_run := "uv run --no-project --python .venv/bin/python"
data := env_var_or_default("JEPA_DATA", justfile_directory() / "data" / "dsb2018")
dependencies := "numpy torch scipy scikit-image scikit-learn tifffile matplotlib umap-learn"
out := justfile_directory() / "out"
force := env_var_or_default("FORCE", "0")

all: pretrain segment probe jev figures report

setup_stamp:
    if [ ! -x .venv/bin/python ]; then uv venv .venv; fi
    if [ "{{force}}" = "1" ] || [ ! -f setup_stamp ]; then uv pip install --python .venv/bin/python {{dependencies}}; touch setup_stamp; else echo "cached environment found"; fi

pretrain: setup_stamp
    if [ "{{force}}" = "1" ] || [ ! -f {{out}}/pretrain_complete.stamp ] || [ ! -s {{out}}/jepa.pt ]; then JEPA_DATA={{data}} {{uv_run}} core_scripts/jepa_pretrain.py {{steps}}; touch {{out}}/pretrain_complete.stamp; else echo "cached pretraining found"; fi

segment: pretrain
    if [ "{{force}}" = "1" ] || [ ! -f {{out}}/seg_complete.stamp ] || [ ! -f {{out}}/seg_results.json ] || [ ! -f {{out}}/preds_jepa16.npz ] || [ ! -f {{out}}/preds_random16.npz ] || [ ! -f {{out}}/preds_unet16.npz ] || [ ! -f {{out}}/preds_cellpose.npz ]; then JEPA_DATA={{data}} {{uv_run}} core_scripts/seg.py; else echo "cached segmentation found"; fi

probe: segment
    if [ "{{force}}" = "1" ] || [ ! -f {{out}}/probe_complete.stamp ] || [ ! -f {{out}}/probe_results.json ]; then JEPA_DATA={{data}} {{uv_run}} core_scripts/probe.py; touch {{out}}/probe_complete.stamp; else echo "cached probe found"; fi

jev: segment
    if [ "{{force}}" = "1" ] || [ ! -f {{out}}/jev_complete.stamp ] || [ ! -f {{out}}/jev_results.json ] || [ ! -f {{out}}/preds_otsu_jev.npz ] || [ ! -f {{out}}/preds_stardist_jev.npz ] || [ ! -f {{out}}/preds_cellpose_jev.npz ]; then JEPA_DATA={{data}} {{uv_run}} core_scripts/jev_layer.py jepa16; touch {{out}}/jev_complete.stamp; else echo "cached Jev analysis found"; fi

figures: jev
    if [ "{{force}}" = "1" ] || [ ! -f {{out}}/figures_complete.stamp ] || [ ! -f {{out}}/panels.json ]; then JEPA_DATA={{data}} {{uv_run}} core_scripts/figures.py; touch {{out}}/figures_complete.stamp; else echo "cached figures found"; fi

report: figures probe
    if [ "{{force}}" = "1" ] || [ ! -f {{out}}/report_complete.stamp ] || [ ! -s {{out}}/report.html ]; then JEPA_DATA={{data}} {{uv_run}} core_scripts/build_report.py; touch {{out}}/report_complete.stamp; else echo "cached report found"; fi

clean-results:
    rm -f {{out}}/jepa.pt {{out}}/jepa_checkpoint.pt {{out}}/jepa_log.json {{out}}/pretrain_complete.stamp
    rm -f {{out}}/seg_results.json {{out}}/seg_complete.stamp
    rm -f {{out}}/probe_results.json {{out}}/probe_complete.stamp
    rm -f {{out}}/jev_results.json {{out}}/jev_complete.stamp
    rm -f {{out}}/panels.json {{out}}/figures_complete.stamp
    rm -f {{out}}/report.html {{out}}/report_complete.stamp
    rm -f {{out}}/preds_*.npz
    rm -f seg_results.json probe_results.json jev_results.json

force-all: clean-results
    just FORCE=1 all

prepare-monuseg RAW OUT='data/monuseg': setup_stamp
    {{uv_run}} core_scripts/prepare_monuseg.py "{{RAW}}" --out {{OUT}}
