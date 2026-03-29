rule generate_negative_dhs:
    output:
        touch(f"{TRAIN_DHS_DIR}.negatives_done")
    params:
        training_dhs_dir=TRAIN_DHS_DIR,
        hg38_2bit_file=DATA["hg_38_2bit_file"],
        gc_bias_window=NEGATIVE_DHS.get("gc_bias_window", 200),
        matrix_columns=MATRIX["columns"],
        max_tries_multiplier=NEGATIVE_DHS.get("max_tries_multiplier", 50),
        n_quantile_bins=NEGATIVE_DHS.get("n_quantile_bins", 20),
    resources:
        runtime=120,
        mem_mb=4000
    script:
        "../scripts/generate_negative_dhs.py"


rule train_model:
    input:
        matrices=expand(
            f"{TRAIN_OUTPUT_DIR}{{sample}}__{{dhs_file}}_rebinned.npy",
            sample=SAMPLES, dhs_file=DHS_FILES,
        ),
    output:
        checkpoint=TRAINING.get("save_checkpoint", MODEL["checkpoint"]),
    resources:
        runtime=300,
        mem_mb=16000
    script:
        "../scripts/train_model.py"
