import pandas as pd
import subprocess


def downsample_matrix(input_file, target_cov):
    df = pd.read_csv(input_file, sep='\t', names=['chr', 'start', 'end'])
    df = df.sample(n=target_cov, random_state=42).sort_index()
    return df

if 'snakemake' in globals():
    inputs = list(map(str, snakemake.input.dhs))
    outputs = list(map(str, snakemake.output.downsampled_dhs))
    
    coverages = []
    for inp in inputs:
        coverages.append(
            (inp, int(subprocess.check_output(['wc', '-l', inp]).split()[0]))
        )
    min_inp, min_cov = min(coverages, key=lambda x: x[1])
    
    for inp, out in zip(inputs, outputs):
        if inp == min_inp:
            subprocess.call(['cp', inp, out])
            continue
            
        downsampled_df = downsample_matrix(inp, min_cov)
        downsampled_df.to_csv(out, sep='\t', header=False, index=False)