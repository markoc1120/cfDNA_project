import gzip
import numpy as np
import logging
import yaml
import os

from collections import deque


config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

MATRIX_ROWS = config['matrix_rows']
MATRIX_COLUMNS = config['matrix_columns']
HARDCUT_OFF_LOWER = config['hardcut_off_lower']


logger = logging.getLogger(__name__)


MATRIX_COLUMNS_HALF = MATRIX_COLUMNS // 2


class Preprocessor():
    def __init__(self, fragment_file, dhs_file, output_file):
        self.fragment_file = fragment_file
        self.dhs_file = dhs_file
        self.output_file = output_file
        self.DHS_sites = None
        self.initial_DHS_length = None
        

    def read_dhs_to_memory(self):
        sites = deque()
        with open(self.dhs_file, 'rt') as f:
            for line in f:
                chr, start, end = line.split('\t')
                midpoint = (int(end) + int(start)) // 2
                sites.append((midpoint, chr))
        return sites, len(sites)


    def get_curr_dhs(self) -> tuple:
        if not self.DHS_sites:
            return None, None, None

        curr_dhs_midpoint, chr = self.DHS_sites.popleft()
        return (
            curr_dhs_midpoint - MATRIX_COLUMNS_HALF, 
            curr_dhs_midpoint + MATRIX_COLUMNS_HALF,
            chr
        )
    

    def parse_fragment(self, line: str) -> tuple:
        parsed_fragment = line.strip().split('\t')
        chr, start, end = parsed_fragment[0:3]
        return chr, int(start), int(end)
    
    
    def downsample_matrix(self, matrix: np.ndarray, target_sum: int) -> np.ndarray:
        current_sum = np.sum(matrix)
        
        if current_sum == 0:
            return matrix
        if current_sum <= target_sum:
            return matrix
            
        flat_matrix = matrix.flatten()
        probabilities = flat_matrix / current_sum
        
        downsampled_flat = np.random.multinomial(target_sum, probabilities)
        
        return downsampled_flat.reshape(matrix.shape)
    
    # length vs fragments' relative midpoint
    def generate_matrix(self, should_save=True) -> np.ndarray:
        self.DHS_sites, self.initial_DHS_length = self.read_dhs_to_memory()
        result = np.zeros((MATRIX_ROWS, MATRIX_COLUMNS))
        curr_dhs_start, curr_dhs_end, curr_chr = self.get_curr_dhs()
    
        with gzip.open(self.fragment_file, 'rt') as f:
            for line in f:
                chr, start, end = self.parse_fragment(line)
                fragment_midpoint, fragment_length = (start + end) // 2, end - start

                # if the fragment is too long skip and log it for now
                if fragment_length >= MATRIX_ROWS:
                    logger.warning(f'Skipped fragment due to too high length:\nstart:{start}\nend:{end}')
                    continue

                # move dhs until to the fragments' chromosome is reached
                while curr_dhs_end and chr != curr_chr:
                    curr_dhs_start, curr_dhs_end, curr_chr = self.get_curr_dhs()
                    if curr_dhs_end is None:
                        logger.warning('No more DHS sites')
                        break

                # move dhs until we have overlapping fragments
                while curr_dhs_end and chr == curr_chr and fragment_midpoint > curr_dhs_end:
                    curr_dhs_start, curr_dhs_end, curr_chr = self.get_curr_dhs()
                    if curr_dhs_end is None:
                        logger.warning('No more DHS sites')
                        break

                # break if no more dhs sites
                if curr_dhs_end is None:
                    logger.warning('No more DHS sites')
                    break

                # move fragments that are not overlapping and in the previous chromosome from the dhs point of view
                if chr != curr_chr:
                    continue
                
                rel_midpoint = fragment_midpoint - curr_dhs_start

                # only track fragments those are in our boundaries
                if rel_midpoint >= 0 and rel_midpoint < MATRIX_COLUMNS:
                    result[fragment_length, rel_midpoint] += 1
        
        total_sum = np.sum(result)
        if total_sum >= HARDCUT_OFF_LOWER:
            result = self.downsample_matrix(result, HARDCUT_OFF_LOWER)
            if should_save:
                logger.info(f"Saving matrix for {self.output_file}")
                np.save(self.output_file, result)
        else:
            logger.warning(f"Total sum {total_sum} is below threshold.")
            if should_save:
                np.save(self.output_file, result)
                open(self.output_file + '.skip', 'w').close()
        
        # skip those fragmetns which are under the HARDCUT_OFF_LOWER, 
        # the problem is that workflow rule expects files for all samples
#         if should_save:
#                 logger.info(f"Saving matrix for {self.output_file}")
#                 np.save(self.output_file, result)
        return result
            
if 'snakemake' in globals():
    fragment_file = snakemake.input.fragment
    dhs_file = snakemake.input.dhs
    output_file = snakemake.output[0]

    preprocessor = Preprocessor(
        fragment_file,
        dhs_file,
        output_file
    )
    preprocessor.generate_matrix()