# run workflow via slurm
[group('slurm-run')]
slurm-run:
    snakemake --snakefile workflow/Snakefile --profile workflow/profiles/slurm --configfile confs/thesis.yaml
