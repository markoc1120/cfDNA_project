import glob

INPUT_FRAGS_DIR = DATA["input_frags_dir"]
TRAIN_DHS_DIR = DATA["training_dhs_dir"]
SORTED_FRAGS_DIR = DATA["sorted_frags_dir"]
TRAIN_SORTED_DHS_DIR = DATA["training_sorted_dhs_dir"]
TRAIN_OUTPUT_DIR = DATA["training_output_dir"]
MIN_COV_FILE = DATA["training_min_coverage_file"]
BIN_EDGES_FILE = DATA["training_bin_edges_file"]
MATRIX_COLUMNS = MATRIX["columns"]
MATRIX_ROWS = MATRIX["rows"]
MATRIX_SHIFT = MATRIX["shift"]

SAMPLES = [
    f.split('/')[-1].replace('.hg38.frag.gz', '')
    for f in glob.glob(f"{INPUT_FRAGS_DIR}*.hg38.frag.gz")
]
DHS_FILES = [
    f.split('/')[-1].replace('.bed', '')
    for f in glob.glob(f"{TRAIN_DHS_DIR}*.bed")
]

rule train_sort_dhs:
    input:
        dhs=f"{TRAIN_DHS_DIR}{{dhs_file}}.bed"
    output:
        dhs_sorted=f"{TRAIN_SORTED_DHS_DIR}{{dhs_file}}_sorted.bed"
    resources:
        runtime=5,
        mem_mb=200
    group: "sort_dhs"
    shell:
        '''
        sort -k1,1V -k2,2n {input.dhs} > {output.dhs_sorted}
        '''

rule train_preprocess_dhs:
    input:
        dhs_sorted=f"{TRAIN_SORTED_DHS_DIR}{{dhs_file}}_sorted.bed"
    output:
        dhs_sorted_preprocessed=f"{TRAIN_SORTED_DHS_DIR}{{dhs_file}}_sorted_wl{MATRIX_COLUMNS}.bed"
    params:
        matrix_columns=MATRIX_COLUMNS
    resources:
        runtime=30,
        mem_mb=300
    group: "prep_dhs"
    script:
        "../scripts/preprocess_dhs.py"

rule train_downsample_dhs:
    input:
        dhs=expand(f"{TRAIN_SORTED_DHS_DIR}{{dhs_file}}_sorted_wl{MATRIX_COLUMNS}.bed", dhs_file=DHS_FILES)
    output:
        downsampled_dhs=expand(f"{TRAIN_SORTED_DHS_DIR}{{dhs_file}}_sorted_wl{MATRIX_COLUMNS}_downsampled.bed", dhs_file=DHS_FILES)
    resources:
        runtime=5,
        mem_mb=300
    group: "downsample_dhs"
    script:
        "../scripts/downsample_dhs.py"

rule train_sort_fragments:
    input:
        fragment=f"{INPUT_FRAGS_DIR}{{sample}}.hg38.frag.gz"
    output:
        fragment_sorted=f"{SORTED_FRAGS_DIR}{{sample}}_sorted.hg38.frag.gz"
    resources:
        runtime=60,
        mem_mb=200
    group: "sort_frag"
    shell:
        '''
        zcat {input.fragment} | sort -k1,1V -k2,2n | gzip -c > {output.fragment_sorted}
        '''

rule train_preprocess_fragments:
    input:
        fragment=f"{SORTED_FRAGS_DIR}{{sample}}_sorted.hg38.frag.gz",
        dhs=f"{TRAIN_SORTED_DHS_DIR}{{dhs_file}}_sorted_wl{MATRIX_COLUMNS}_downsampled.bed"
    output:
        raw=f"{TRAIN_OUTPUT_DIR}{{sample}}__{{dhs_file}}.npy",
        cov=f"{TRAIN_OUTPUT_DIR}{{sample}}__{{dhs_file}}.cov.txt"
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

rule calculate_min_coverage:
    input:
        covs=expand(f"{TRAIN_OUTPUT_DIR}{{sample}}__{{dhs_file}}.cov.txt", sample=SAMPLES, dhs_file=DHS_FILES)
    output:
        MIN_COV_FILE
    resources:
        runtime=10,
        mem_mb=500,
    script:
        "../scripts/calculate_min_coverage.py"

rule train_downsample_matrices:
    input:
        raw=f"{TRAIN_OUTPUT_DIR}{{sample}}__{{dhs_file}}.npy",
        mincov=MIN_COV_FILE
    output:
        f"{TRAIN_OUTPUT_DIR}{{sample}}__{{dhs_file}}_downsampled.npy"
    resources:
        runtime=10,
        mem_mb=150
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
        n_rebin_rows=MATRIX["n_rebin_rows"]
    resources:
        runtime=30,
        mem_mb=8000
    script:
        "../scripts/compute_bin_edges.py"

rule rebin_matrices:
    input:
        matrix = f"{TRAIN_OUTPUT_DIR}{{sample}}__{{dhs_file}}_downsampled.npy",
        bin_edges = BIN_EDGES_FILE
    output:
        f"{TRAIN_OUTPUT_DIR}{{sample}}__{{dhs_file}}_rebinned.npy"
    resources:
        runtime=10,
        mem_mb=150
    group: "rebin_matrices"
    script:
        "../scripts/rebin_matrices.py"
