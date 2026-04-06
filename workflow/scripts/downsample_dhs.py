from cfdna.preprocessing.dhs import downsample_dhs_files

if 'snakemake' in globals():
    inputs = list(map(str, snakemake.input.dhs))
    outputs = list(map(str, snakemake.output.downsampled_dhs))
    downsample_dhs_files(inputs, outputs)
