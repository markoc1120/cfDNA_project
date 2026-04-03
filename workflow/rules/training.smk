rule train_model:
    input:
        matrices=expand(
            f"{TRAIN_OUTPUT_DIR}{{sample}}__{{dhs_file}}_rebinned.npy",
            sample=SAMPLES, dhs_file=DHS_FILES,
        ),
    params:
        input_type=INPUT_TYPE,
    output:
        checkpoint=MODEL["checkpoint"],
    resources:
        runtime=300,
        mem_mb=16000
    script:
        "../scripts/train_model.py"
