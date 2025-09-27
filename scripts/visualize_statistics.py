import numpy as np
import yaml

from test_statistics import create_statistic


def load_statistic_data(file_path):
    return np.load(file_path)


if 'snakemake' in globals():
    matrix_path = snakemake.input.matrix
    statistic_path = snakemake.input.statistic
    config_path = snakemake.input.config
    
    output_paths = snakemake.output[0]
    
    statistic_name = snakemake.params.statistic
    sample_name = snakemake.params.get('sample', None)

    with open(matrix_path, 'rb') as f:
        matrix = np.load(f)

    statistic_data = load_statistic_data(statistic_path)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    statistic = create_statistic(statistic_name, config.get(statistic_name, {}))
    
    statistic.visualize(statistic_data, output_paths)
