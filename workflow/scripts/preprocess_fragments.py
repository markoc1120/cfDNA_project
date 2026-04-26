from cfdna.preprocessing.fragments import Preprocessor

if 'snakemake' in globals():
    preprocessor = Preprocessor(
        snakemake.input.fragment,
        list(snakemake.input.dhs),
        list(snakemake.output.raw),
        list(snakemake.output.cov),
        matrix_rows=snakemake.params.matrix_rows,
        matrix_columns=snakemake.params.matrix_columns,
        matrix_shift=snakemake.params.matrix_shift,
    )
    preprocessor.generate_matrix()
