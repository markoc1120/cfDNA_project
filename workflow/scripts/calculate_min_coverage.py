from cfdna.preprocessing.matrices import calculate_min_coverage

if 'snakemake' in globals():
    cov_txt_files = snakemake.input.covs
    min_cov = calculate_min_coverage(cov_txt_files)

    output_path = snakemake.output[0]
    with open(output_path, 'w') as f:
        f.write(str(min_cov) + '\n')
