# run workflow locally
[group('local-run')]
local-run:
    snakemake --snakefile workflow/Snakefile --profile workflow/profiles/local --configfile confs/thesis.yaml --until generate_negative_dhs
    snakemake --snakefile workflow/Snakefile --profile workflow/profiles/local --configfile confs/thesis.yaml
