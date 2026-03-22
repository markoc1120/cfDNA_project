import glob as glob

INFERENCE_DHS_DIR = DATA["inference_dhs_dir"]
INFERENCE_SORTED_DHS_DIR = DATA["inference_sorted_dhs_dir"]
INFERENCE_FRAGS_DIR = DATA["inference_frags_dir"]
INFERENCE_SORTED_FRAGS_DIR = DATA["inference_sorted_frags_dir"]
INFERENCE_OUTPUT_DIR = DATA["inference_output_dir"]
INFERENCE_USE_REBINNED = MODEL.get("use_rebinned", True)

INFERENCE_SAMPLES = [
    f.split('/')[-1].replace('.hg38.frag.gz', '')
    for f in glob.glob(f"{INFERENCE_FRAGS_DIR}*.hg38.frag.gz")
]
INFERENCE_DHS_FILES = [
    f.split('/')[-1].replace('.bed', '')
    for f in glob.glob(f"{INFERENCE_DHS_DIR}*.bed")
]


def inference_input(wildcards):
    if INFERENCE_USE_REBINNED:
        return f"{INFERENCE_OUTPUT_DIR}{wildcards.sample}__{wildcards.dhs_file}_rebinned.npy"
    return f"{INFERENCE_OUTPUT_DIR}{wildcards.sample}__{wildcards.dhs_file}_downsampled.npy"


rule inference_sort_dhs:
    input:
        dhs=f"{INFERENCE_DHS_DIR}{{dhs_file}}.bed"
    output:
        dhs_sorted=f"{INFERENCE_SORTED_DHS_DIR}{{dhs_file}}_sorted.bed"
    resources:
        runtime=5,
        mem_mb=200
    group: "sort_dhs"
    shell:
        """
        sort -k1,1V -k2,2n {input.dhs} > {output.dhs_sorted}
        """

rule inference_preprocess_dhs:
    input:
        dhs_sorted=f"{INFERENCE_SORTED_DHS_DIR}{{dhs_file}}_sorted.bed"
    output:
        dhs_sorted_preprocessed=f"{INFERENCE_SORTED_DHS_DIR}{{dhs_file}}_sorted_wl{MATRIX_COLUMNS}.bed"
    params:
        matrix_columns=MATRIX_COLUMNS
    resources:
        runtime=30,
        mem_mb=300
    group: "prep_dhs"
    script:
        "../scripts/preprocess_dhs.py"

rule inference_sort_fragments:
    input:
        fragment=f"{INFERENCE_FRAGS_DIR}{{sample}}.hg38.frag.gz"
    output:
        fragment_sorted=f"{INFERENCE_SORTED_FRAGS_DIR}{{sample}}_sorted.hg38.frag.gz"
    resources:
        runtime=600,
        mem_mb=200
    group: "sort_frag"
    shell:
        """
        zcat {input.fragment} | sort -k1,1V -k2,2n | gzip -c > {output.fragment_sorted}
        """

rule inference_preprocess_fragments:
    input:
        fragment=f"{INFERENCE_SORTED_FRAGS_DIR}{{sample}}_sorted.hg38.frag.gz",
        dhs=f"{INFERENCE_SORTED_DHS_DIR}{{dhs_file}}_sorted_wl{MATRIX_COLUMNS}.bed"
    output:
        raw=f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}.npy",
        cov=f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}.cov.txt"
    params:
        matrix_rows=MATRIX_ROWS,
        matrix_columns=MATRIX_COLUMNS,
        matrix_shift=MATRIX_SHIFT
    resources:
        runtime=30,
        mem_mb=200
    group: "prep_frag"
    script:
        "../scripts/preprocess_fragments.py"

rule inference_downsample_matrices:
    input:
        raw=f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}.npy",
        mincov=MIN_COV_FILE
    output:
        f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}_downsampled.npy"
    resources:
        runtime=10,
        mem_mb=150
    group: "downsample_matrices"
    script:
        "../scripts/downsample_matrices.py"

rule inference_rebin_matrices:
    input:
        matrix=f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}_downsampled.npy",
        bin_edges=BIN_EDGES_FILE
    output:
        f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}_rebinned.npy"
    resources:
        runtime=5,
        mem_mb=150
    group: "rebin_matrices"
    script:
        "../scripts/rebin_matrices.py"

rule run_inference:
    input:
        matrix=inference_input,
        cov=f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}.cov.txt",
        min_cov=MIN_COV_FILE,
        checkpoint=MODEL["checkpoint"],
    output:
        score=f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}_score.txt"
    params:
        checkpoint=MODEL["checkpoint"],
        model_type=MODEL["name"],
    resources:
        runtime=10,
        mem_mb=50
    group: "inference"
    script:
        "../scripts/run_inference.py"
