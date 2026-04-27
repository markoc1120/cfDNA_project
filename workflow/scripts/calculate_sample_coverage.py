from cfdna.preprocessing.matrices import sum_coverage_files

if 'snakemake' in globals():
    cov_txt_files = snakemake.input.covs
    sample_total = sum_coverage_files(cov_txt_files)

    output_path = snakemake.output[0]
    with open(output_path, 'w') as f:
        f.write(str(sample_total) + '\n')
