accessibility_inputs = {}

if STAGES.get("accessibility_scores", False):
    accessibility_inputs = dict(
        lwps_inputs=expand(
            f"{ACCESSIBILITY_DIR}{{sample}}__{{dhs_file}}_downsampled_lwps.npy",
            sample=INFERENCE_SAMPLES, dhs_file=INFERENCE_DHS_FILES,
        ),
        ocf_inputs=expand(
            f"{ACCESSIBILITY_DIR}{{sample}}__{{dhs_file}}_downsampled_ocf.npy",
            sample=INFERENCE_SAMPLES, dhs_file=INFERENCE_DHS_FILES,
        ),
        fdi_inputs=expand(
            f"{ACCESSIBILITY_DIR}{{sample}}__{{dhs_file}}_downsampled_fdi.npz",
            sample=INFERENCE_SAMPLES, dhs_file=INFERENCE_DHS_FILES,
        ),
        ifs_inputs=expand(
            f"{ACCESSIBILITY_DIR}{{sample}}__{{dhs_file}}_downsampled_ifs.npz",
            sample=INFERENCE_SAMPLES, dhs_file=INFERENCE_DHS_FILES,
        ),
        pfe_inputs=expand(
            f"{ACCESSIBILITY_DIR}{{sample}}__{{dhs_file}}_downsampled_pfe.npz",
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
        runtime=30,
        mem_mb=8000
    script:
        "../scripts/build_matrices_joint.py"
