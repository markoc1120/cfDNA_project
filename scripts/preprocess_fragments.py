import gzip
import numpy as np
import logging

from collections import deque
try:
    from constants import MATRIX_ROWS, MATRIX_COLUMNS
except ImportError:
    # fallbacks
    MATRIX_ROWS = 264
    MATRIX_COLUMNS = 2000


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
                    
        if should_save:
            np.save(self.output_file, result)
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