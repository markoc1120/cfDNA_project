import numpy as np
import yaml

from test_statistics import create_statistic


def save_statistic_data(data, file_path):
    np.save(file_path, data)


if 'snakemake' in globals():
    matrix_path = snakemake.input.matrix
    config_path = snakemake.input.config
    
    output_path = snakemake.output[0]
    
    statistic_name = snakemake.params.statistic

    with open(matrix_path, 'rb') as f:
        matrix = np.load(f)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    statistic = create_statistic(statistic_name, config.get(statistic_name, {}))
    result = statistic.calculate(matrix)
    
    save_statistic_data(result, output_path)
