SAMPLES = [
    os.path.basename(os.path.dirname(p))
    for p in glob.glob(f"{INPUT_FRAGS_DIR}*/{FRAG_FILENAME}")
]

if GENERATE_BASE_MATRICES:
    rule train_preprocess_fragments:
        input:
            fragment=f"{INPUT_FRAGS_DIR}{{sample}}/{FRAG_FILENAME}",
            dhs=expand(
                f"{TRAIN_DHS_DIR}{{dhs_file}}_wl{MATRIX_COLUMNS}.bed",
                dhs_file=DHS_FILES,
            )
        output:
            raw=[
                f"{TRAIN_BASE_MATRICES_DIR}{{sample}}__{dhs_file}.npy"
                for dhs_file in DHS_FILES
            ],
            cov=[
                f"{TRAIN_BASE_MATRICES_DIR}{{sample}}__{dhs_file}.cov.txt"
                for dhs_file in DHS_FILES
            ]
        params:
            matrix_rows=MATRIX_ROWS,
            matrix_columns=MATRIX_COLUMNS,
            matrix_shift=MATRIX_SHIFT
        resources:
            runtime=50
        group: "prep_frag"
        script:
            "../scripts/preprocess_fragments.py"

if COVERAGE_HANDLING == "downsample":
    rule calculate_min_coverage:
        input:
            covs=expand(f"{TRAIN_BASE_MATRICES_DIR}{{sample}}__{{dhs_file}}.cov.txt", sample=SAMPLES, dhs_file=DHS_FILES)
        output:
            MIN_COV_FILE
        params:
            user_min_cov=MIN_COV_OVERRIDE
        resources:
            runtime=10
        script:
            "../scripts/calculate_min_coverage.py"

    rule train_downsample_matrices:
        input:
            raw=f"{TRAIN_BASE_MATRICES_DIR}{{sample}}__{{dhs_file}}.npy",
            mincov=MIN_COV_FILE
        output:
            f"{TRAIN_OUTPUT_DIR}{{sample}}__{{dhs_file}}_downsampled.npy"
        resources:
            runtime=5
        group: "downsample_matrices"
        script:
            "../scripts/downsample_matrices.py"

if COVERAGE_HANDLING == "normalize":
    rule train_calculate_sample_coverage:
        input:
            covs=expand(f"{TRAIN_BASE_MATRICES_DIR}{{{{sample}}}}__{{dhs_file}}.cov.txt", dhs_file=DHS_FILES)
        output:
            f"{TRAIN_OUTPUT_DIR}{{sample}}_sample_coverage.txt"
        resources:
            runtime=2
        group: "calc_sample_coverage"
        script:
            "../scripts/calculate_sample_coverage.py"

    rule train_normalize_matrices:
        input:
            raw=f"{TRAIN_BASE_MATRICES_DIR}{{sample}}__{{dhs_file}}.npy",
            sample_cov=f"{TRAIN_OUTPUT_DIR}{{sample}}_sample_coverage.txt"
        output:
            f"{TRAIN_OUTPUT_DIR}{{sample}}__{{dhs_file}}_normalized.npy"
        resources:
            runtime=5
        group: "normalize_matrices"
        script:
            "../scripts/normalize_matrices.py"

rule compute_bin_edges:
    input:
        matrices=expand(f"{TRAIN_OUTPUT_DIR}{{sample}}__{{dhs_file}}{COVERAGE_SUFFIX}.npy", sample=SAMPLES, dhs_file=DHS_FILES)
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
        matrix = f"{TRAIN_OUTPUT_DIR}{{sample}}__{{dhs_file}}{COVERAGE_SUFFIX}.npy",
        bin_edges = BIN_EDGES_FILE
    output:
        f"{TRAIN_OUTPUT_DIR}{{sample}}__{{dhs_file}}_rebinned.npy"
    resources:
        runtime=2
    group: "rebin_matrices"
    script:
        "../scripts/rebin_matrices.py"
