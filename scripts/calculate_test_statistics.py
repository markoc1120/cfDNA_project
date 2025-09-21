import numpy as np
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


# TODO: export window_size to constants.py
def calculate_lwps(matrix: np.ndarray, window_size=120) -> np.ndarray:
    num_positions = matrix.shape[1]
    lwps = np.zeros(num_positions)
    
    # sliding window -> calculating lwps for each positions 1,2,3..,1999,2000
    for pos in range(num_positions):
        if not (pos % 100) or pos == num_positions - 1:
            logger.info("{}%".format(round(pos/num_positions * 100)))
        
        # for position 0 -> window [-60, 60]
        window_start = pos - window_size // 2
        window_end = pos + window_size // 2
        
        # fragments which are outside of this [-60, 60], starts before -60 and ends after 60
        spanning_count = 0
        # fragments those either start or end in the window
        internal_endpoints = 0
        
        for fragment_length in range(matrix.shape[0]):
            for rel_midpoint in range(matrix.shape[1]):
                count = matrix[fragment_length, rel_midpoint]
                if count > 0:
                    # calculate fragment start and end from midpoint and length
                    frag_start = rel_midpoint - fragment_length // 2
                    frag_end = rel_midpoint + fragment_length // 2
                    
                    # count spanning fragments
                    if frag_start <= window_start and frag_end >= window_end:
                        spanning_count += count
                    
                    # count internal endpoints
                    if window_start <= frag_start <= window_end:  # starting in the window
                        internal_endpoints += count
                    if window_start <= frag_end <= window_end:  # ending in the window
                        internal_endpoints += count
        
        lwps[pos] = spanning_count - internal_endpoints
    
    return lwps


if 'snakemake' in globals():
    matrix_file_path = snakemake.input.matrix
    
    output_path = snakemake.output[0]
    
    with open(matrix_file_path, 'rb') as f:
        matrix = np.load(f)
        
    lwps = calculate_lwps(matrix)
    np.save(output_path, lwps)
        
    