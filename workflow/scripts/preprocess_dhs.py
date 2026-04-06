from cfdna.preprocessing.dhs import preprocess_dhs

if 'snakemake' in globals():
    preprocess_dhs(
        snakemake.input.dhs,
        snakemake.output.dhs_preprocessed,
        matrix_columns=snakemake.params.matrix_columns,
    )
