import glob
import os

SAMPLES = [
    os.path.basename(os.path.dirname(p))
    for p in glob.glob(f"{INPUT_FRAGS_DIR}*/{FRAG_FILENAME}")
]
DHS_FILES = [
    f.split('/')[-1].replace('.bed', '')
    for f in glob.glob(f"{TRAIN_DHS_DIR}*.bed")
    if '_wl' not in f.split('/')[-1]
]

rule train_preprocess_dhs:
    input:
        dhs=f"{TRAIN_DHS_DIR}{{dhs_file}}.bed"
    output:
        dhs_preprocessed=temp(f"{TRAIN_DHS_DIR}{{dhs_file}}_wl{MATRIX_COLUMNS}.bed")
    params:
        matrix_columns=MATRIX_COLUMNS
    resources:
        runtime=10
    group: "prep_dhs"
    script:
        "../scripts/preprocess_dhs.py"

rule train_downsample_dhs:
    input:
        dhs=expand(f"{TRAIN_DHS_DIR}{{dhs_file}}_wl{MATRIX_COLUMNS}.bed", dhs_file=DHS_FILES)
    output:
        downsampled_dhs=temp(expand(f"{TRAIN_DHS_DIR}{{dhs_file}}_wl{MATRIX_COLUMNS}_downsampled.bed", dhs_file=DHS_FILES))
    resources:
        runtime=5
    group: "downsample_dhs"
    script:
        "../scripts/downsample_dhs.py"

rule train_preprocess_fragments:
    input:
        fragment=f"{INPUT_FRAGS_DIR}{{sample}}/{FRAG_FILENAME}",
        dhs=f"{TRAIN_DHS_DIR}{{dhs_file}}_wl{MATRIX_COLUMNS}_downsampled.bed"
    output:
        raw=temp(f"{TRAIN_OUTPUT_DIR}{{sample}}__{{dhs_file}}.npy"),
        cov=temp(f"{TRAIN_OUTPUT_DIR}{{sample}}__{{dhs_file}}.cov.txt")
    params:
        matrix_rows=MATRIX_ROWS,
        matrix_columns=MATRIX_COLUMNS,
        matrix_shift=MATRIX_SHIFT
    resources:
        runtime=20
    group: "prep_frag"
    script:
        "../scripts/preprocess_fragments.py"

rule calculate_min_coverage:
    input:
        covs=expand(f"{TRAIN_OUTPUT_DIR}{{sample}}__{{dhs_file}}.cov.txt", sample=SAMPLES, dhs_file=DHS_FILES)
    output:
        MIN_COV_FILE
    resources:
        runtime=10
    script:
        "../scripts/calculate_min_coverage.py"

rule train_downsample_matrices:
    input:
        raw=f"{TRAIN_OUTPUT_DIR}{{sample}}__{{dhs_file}}.npy",
        mincov=MIN_COV_FILE
    output:
        f"{TRAIN_OUTPUT_DIR}{{sample}}__{{dhs_file}}_downsampled.npy"
    resources:
        runtime=5
    group: "downsample_matrices"
    script:
        "../scripts/downsample_matrices.py"

rule compute_bin_edges:
    input:
        matrices=expand(f"{TRAIN_OUTPUT_DIR}{{sample}}__{{dhs_file}}_downsampled.npy", sample=SAMPLES, dhs_file=DHS_FILES)
    output:
        bin_edges=BIN_EDGES_FILE
    params:
        matrix_rows=MATRIX_ROWS,
    resources:
        runtime=30
    script:
        "../scripts/compute_bin_edges.py"

rule rebin_matrices:
    input:
        matrix = f"{TRAIN_OUTPUT_DIR}{{sample}}__{{dhs_file}}_downsampled.npy",
        bin_edges = BIN_EDGES_FILE
    output:
        f"{TRAIN_OUTPUT_DIR}{{sample}}__{{dhs_file}}_rebinned.npy"
    resources:
        runtime=2
    group: "rebin_matrices"
    script:
        "../scripts/rebin_matrices.py"
