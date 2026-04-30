import glob as glob
import os

INFERENCE_SAMPLES = [
    os.path.basename(os.path.dirname(p))
    for p in glob.glob(f"{INFERENCE_FRAGS_DIR}*/{FRAG_FILENAME}")
]
INFERENCE_DHS_FILES = [
    f.split('/')[-1].replace('.bed', '')
    for f in glob.glob(f"{INFERENCE_DHS_DIR}*.bed")
    if '_wl' not in f.split('/')[-1]
]


def inference_input(wildcards):
    return f"{INFERENCE_OUTPUT_DIR}{wildcards.sample}__{wildcards.dhs_file}_{INPUT_TYPE}.npy"


rule inference_preprocess_dhs:
    input:
        dhs=f"{INFERENCE_DHS_DIR}{{dhs_file}}.bed"
    output:
        dhs_preprocessed=f"{INFERENCE_DHS_DIR}{{dhs_file}}_wl{MATRIX_COLUMNS}.bed"
    params:
        matrix_columns=MATRIX_COLUMNS
    resources:
        runtime=20
    group: "prep_dhs"
    script:
        "../scripts/preprocess_dhs.py"

if GENERATE_BASE_MATRICES:
    rule inference_preprocess_fragments:
        input:
            fragment=f"{INFERENCE_FRAGS_DIR}{{sample}}/{FRAG_FILENAME}",
            dhs=expand(
                f"{INFERENCE_DHS_DIR}{{dhs_file}}_wl{MATRIX_COLUMNS}.bed",
                dhs_file=INFERENCE_DHS_FILES,
            )
        output:
            raw=[
                f"{INFERENCE_BASE_MATRICES_DIR}{{sample}}__{dhs_file}.npy"
                for dhs_file in INFERENCE_DHS_FILES
            ],
            cov=[
                f"{INFERENCE_BASE_MATRICES_DIR}{{sample}}__{dhs_file}.cov.txt"
                for dhs_file in INFERENCE_DHS_FILES
            ]
        params:
            matrix_rows=MATRIX_ROWS,
            matrix_columns=MATRIX_COLUMNS,
            matrix_shift=MATRIX_SHIFT
        resources:
            runtime=20
        group: "prep_frag"
        script:
            "../scripts/preprocess_fragments.py"

if COVERAGE_HANDLING == "downsample":
    rule inference_downsample_matrices:
        input:
            raw=f"{INFERENCE_BASE_MATRICES_DIR}{{sample}}__{{dhs_file}}.npy",
            mincov=MIN_COV_FILE
        output:
            temp(f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}_downsampled.npy")
        resources:
            runtime=5
        group: "downsample_matrices"
        script:
            "../scripts/downsample_matrices.py"

if COVERAGE_HANDLING == "normalize":
    rule inference_calculate_sample_coverage:
        input:
            covs=expand(f"{INFERENCE_BASE_MATRICES_DIR}{{{{sample}}}}__{{dhs_file}}.cov.txt", dhs_file=INFERENCE_DHS_FILES)
        output:
            temp(f"{INFERENCE_OUTPUT_DIR}{{sample}}_sample_coverage.txt")
        resources:
            runtime=5
        script:
            "../scripts/calculate_sample_coverage.py"

    rule inference_normalize_matrices:
        input:
            raw=f"{INFERENCE_BASE_MATRICES_DIR}{{sample}}__{{dhs_file}}.npy",
            sample_cov=f"{INFERENCE_OUTPUT_DIR}{{sample}}_sample_coverage.txt"
        output:
            temp(f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}_normalized.npy")
        resources:
            runtime=5
        group: "normalize_matrices"
        script:
            "../scripts/normalize_matrices.py"

rule calculate_coverage_after_coverage_handling:
    input:
        f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}_{COVERAGE_SUFFIX}.npy"
    output:
        f"{ACCESSIBILITY_DIR}{{sample}}__{{dhs_file}}.cov.txt"
    resources:
        runtime=5
    group: "coverage_handling"
    script:
        "../scripts/calculate_coverage.py"

rule inference_rebin_matrices:
    input:
        matrix=f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}_{COVERAGE_SUFFIX}.npy",
        bin_edges=BIN_EDGES_FILE
    output:
        temp(f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}_rebinned.npy")
    resources:
        runtime=10
    group: "rebin_matrices"
    script:
        "../scripts/rebin_matrices.py"

rule run_inference:
    input:
        matrix=inference_input,
        checkpoint=MODEL["checkpoint"],
    output:
        f"{ACCESSIBILITY_DIR}{{sample}}__{{dhs_file}}_{INFERENCE_OUTPUT_SUFFIX}"
    params:
        checkpoint=MODEL["checkpoint"],
        model_type=MODEL["name"],
    resources:
        runtime=10
    group: "inference"
    script:
        "../scripts/run_inference.py"
