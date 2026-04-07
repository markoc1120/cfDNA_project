import glob as glob

INFERENCE_SAMPLES = [
    f.split('/')[-1].replace('.hg38.frag.gz', '')
    for f in glob.glob(f"{INFERENCE_FRAGS_DIR}*.hg38.frag.gz")
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
        dhs_preprocessed=temp(f"{INFERENCE_DHS_DIR}{{dhs_file}}_wl{MATRIX_COLUMNS}.bed")
    params:
        matrix_columns=MATRIX_COLUMNS
    resources:
        runtime=10,
        mem_mb=300
    group: "prep_dhs"
    script:
        "../scripts/preprocess_dhs.py"

rule inference_preprocess_fragments:
    input:
        fragment=f"{INFERENCE_FRAGS_DIR}{{sample}}.hg38.frag.gz",
        dhs=f"{INFERENCE_DHS_DIR}{{dhs_file}}_wl{MATRIX_COLUMNS}.bed"
    output:
        raw=temp(f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}.npy"),
        cov=temp(f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}.cov.txt")
    params:
        matrix_rows=MATRIX_ROWS,
        matrix_columns=MATRIX_COLUMNS,
        matrix_shift=MATRIX_SHIFT
    resources:
        runtime=10,
        mem_mb=200
    group: "prep_frag"
    script:
        "../scripts/preprocess_fragments.py"

rule inference_downsample_matrices:
    input:
        raw=f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}.npy",
        mincov=MIN_COV_FILE
    output:
        temp(f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}_downsampled.npy")
    resources:
        runtime=5,
        mem_mb=300
    group: "downsample_matrices"
    script:
        "../scripts/downsample_matrices.py"

rule calculate_coverage_after_downsample_matrices:
    input:
        f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}_downsampled.npy"
    output:
        f"{ACCESSIBILITY_DIR}{{sample}}__{{dhs_file}}.cov.txt"
    resources:
        runtime=5,
        mem_mb=50
    group: "downsample_matrices"
    script:
        "../scripts/calculate_coverage.py"

rule inference_rebin_matrices:
    input:
        matrix=f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}_downsampled.npy",
        bin_edges=BIN_EDGES_FILE
    output:
        f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}_rebinned.npy"
    resources:
        runtime=10,
        mem_mb=150
    group: "rebin_matrices"
    script:
        "../scripts/rebin_matrices.py"

rule run_inference:
    input:
        matrix=inference_input,
        checkpoint=MODEL["checkpoint"],
    output:
        score=temp(f"{ACCESSIBILITY_DIR}{{sample}}__{{dhs_file}}_score.txt")
    params:
        checkpoint=MODEL["checkpoint"],
        model_type=MODEL["name"],
    resources:
        runtime=10,
        mem_mb=300
    group: "inference"
    script:
        "../scripts/run_inference.py"
