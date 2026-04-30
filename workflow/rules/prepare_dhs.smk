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
        downsampled_dhs=expand(f"{TRAIN_DHS_DIR}{{dhs_file}}_wl{MATRIX_COLUMNS}_downsampled.bed", dhs_file=DHS_FILES)
    resources:
        runtime=5
    group: "downsample_dhs"
    script:
        "../scripts/downsample_dhs.py"

