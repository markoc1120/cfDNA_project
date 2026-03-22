# run workflow locally
[group('local-run')]
local-run:
    snakemake --snakefile workflow/Snakefile --profile workflow/profiles/local --configfile confs/thesis.yaml
