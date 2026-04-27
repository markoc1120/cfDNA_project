from cfdna.preprocessing.matrices import calculate_min_coverage

if 'snakemake' in globals():
    user_min_cov = snakemake.params.get('user_min_cov')
    if user_min_cov is not None:
        min_cov = float(user_min_cov)
    else:
        min_cov = calculate_min_coverage(snakemake.input.covs)

    output_path = snakemake.output[0]
    with open(output_path, 'w') as f:
        f.write(str(min_cov) + '\n')
