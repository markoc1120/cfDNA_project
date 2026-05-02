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
