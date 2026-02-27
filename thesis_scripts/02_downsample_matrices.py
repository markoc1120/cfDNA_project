import numpy as np


def downsample_matrix(matrix: np.ndarray, target_sum: int) -> np.ndarray:
    current_sum = np.sum(matrix)
    
    if current_sum == 0:
        return matrix
    if current_sum <= target_sum:
        return matrix
        
    flat_matrix = matrix.flatten()
    probabilities = flat_matrix / current_sum
    
    downsampled_flat = np.random.multinomial(target_sum, probabilities)
    return downsampled_flat.reshape(matrix.shape)


if 'snakemake' in globals():
    mincov_txt_path = snakemake.input['mincov']
    with open(mincov_txt_path) as f:
        hardcut = int(f.read().strip())
    
    matrix = np.load(snakemake.input['raw'])
    downsampled_matrix = downsample_matrix(matrix, hardcut)
    np.save(snakemake.output[0], downsampled_matrix)