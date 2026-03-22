from cfdna.preprocessing.dhs import preprocess_dhs

if 'snakemake' in globals():
    preprocess_dhs(
        snakemake.input.dhs_sorted,
        snakemake.output.dhs_sorted_preprocessed,
        matrix_columns=snakemake.params.matrix_columns,
    )
