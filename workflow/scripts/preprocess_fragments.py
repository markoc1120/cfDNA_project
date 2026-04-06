from cfdna.preprocessing.fragments import Preprocessor

if 'snakemake' in globals():
    preprocessor = Preprocessor(
        snakemake.input.fragment,
        snakemake.input.dhs,
        snakemake.output.raw,
        snakemake.output.cov,
        matrix_rows=snakemake.params.matrix_rows,
        matrix_columns=snakemake.params.matrix_columns,
        matrix_shift=snakemake.params.matrix_shift,
    )
    preprocessor.generate_matrix()
