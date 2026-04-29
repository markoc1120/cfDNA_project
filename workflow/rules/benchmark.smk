accessibility_inputs = {}

if STAGES.get("accessibility_scores", False):
    accessibility_inputs = dict(
        lwps_inputs=expand(
            f"{ACCESSIBILITY_DIR}{{sample}}__{{dhs_file}}_{COVERAGE_SUFFIX}_lwps.npy",
            sample=INFERENCE_SAMPLES, dhs_file=INFERENCE_DHS_FILES,
        ),
        ocf_inputs=expand(
            f"{ACCESSIBILITY_DIR}{{sample}}__{{dhs_file}}_{COVERAGE_SUFFIX}_ocf.npy",
            sample=INFERENCE_SAMPLES, dhs_file=INFERENCE_DHS_FILES,
        ),
        fdi_inputs=expand(
            f"{ACCESSIBILITY_DIR}{{sample}}__{{dhs_file}}_{COVERAGE_SUFFIX}_fdi.npz",
            sample=INFERENCE_SAMPLES, dhs_file=INFERENCE_DHS_FILES,
        ),
        ifs_inputs=expand(
            f"{ACCESSIBILITY_DIR}{{sample}}__{{dhs_file}}_{COVERAGE_SUFFIX}_ifs.npz",
            sample=INFERENCE_SAMPLES, dhs_file=INFERENCE_DHS_FILES,
        ),
        pfe_inputs=expand(
            f"{ACCESSIBILITY_DIR}{{sample}}__{{dhs_file}}_{COVERAGE_SUFFIX}_pfe.npz",
            sample=INFERENCE_SAMPLES, dhs_file=INFERENCE_DHS_FILES,
        ),
    )


rule build_feature_matrices:
    input:
        model_inputs=expand(
            f"{ACCESSIBILITY_DIR}{{sample}}__{{dhs_file}}_{INFERENCE_OUTPUT_SUFFIX}",
            sample=INFERENCE_SAMPLES, dhs_file=INFERENCE_DHS_FILES,
        ),
        metadata=DATA["inference_metadata_path"],
        config="confs/thesis.yaml",
        **accessibility_inputs
    params:
        dhs_files=INFERENCE_DHS_FILES,
    output:
        matrices=expand(
            f"{FINAL_MATRICES_DIR}feature_matrix_{{stat}}.parquet",
            stat=BENCH_STATS,
        ),
    resources:
        runtime=30
    script:
        "../scripts/build_matrices_joint.py"
