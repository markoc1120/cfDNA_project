if 'snakemake' in globals():
    cov_txt_files = snakemake.input.covs
    
    vals = []
    for cov in cov_txt_files:
        with open(cov) as f:
            vals.append(int(f.read().strip()))

    output_min_cov = snakemake.output[0]
    min_cov = min(vals)
    with open(output_min_cov, 'w') as f:
        f.write(str(min_cov) + '\n')
