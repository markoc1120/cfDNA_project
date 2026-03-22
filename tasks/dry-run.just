# dry-run workflow
[group('dry-run')]
dry-run:
    snakemake --snakefile workflow/Snakefile --configfile confs/thesis.yaml -n
