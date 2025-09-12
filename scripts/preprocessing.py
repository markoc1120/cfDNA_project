import gzip
import numpy as np
import logging

from collections import deque
from constants import MATRIX_ROWS, MATRIX_COLUMNS


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
        with gzip.open(self.dhs_file, 'rt') as f:
            for line in f:
                start, end = line.split('\t')
                sites.append((int(end)+int(start))//2)
        return sites, len(sites)


    def get_curr_dhs(self) -> tuple:
        if not self.DHS_sites:
            return None, None, None, None

        curr_dhs = self.DHS_sites.popleft()
        return (
            curr_dhs, 
            curr_dhs - MATRIX_COLUMNS_HALF, 
            curr_dhs + MATRIX_COLUMNS_HALF, 
            self.initial_DHS_length - len(self.DHS_sites) - 1
        )
    

    def parse_fragment(self, line: str) -> tuple:
        parsed_fragment = line.strip().split('\t')
        start, end = parsed_fragment[1:3]
        return int(start), int(end)
    
    
    def generate_matrix(self, should_save=True) -> np.ndarray:
        self.DHS_sites, self.initial_DHS_length = self.read_dhs_to_memory()
        result = np.zeros((self.initial_DHS_length, MATRIX_ROWS, MATRIX_COLUMNS))
        curr_dhs, curr_dhs_start, curr_dhs_end, curr_dhs_index = self.get_curr_dhs()
        
        with gzip.open(self.fragment_file, 'rt') as f:
            for line in f:
                start, end = self.parse_fragment(line)
                fragment_length = end - start

                # if the fragment is too long skip and log it for now
                if fragment_length >= MATRIX_ROWS:
                    logger.warning(f'Skipped fragment due to too high length:\nstart:{start}\nend:{end}')
                    continue

                # move dhs until we have overlapping fragments
                while curr_dhs and end >= curr_dhs_end:
                    curr_dhs, curr_dhs_start, curr_dhs_end, curr_dhs_index = self.get_curr_dhs()
                    if curr_dhs is None:
                        logger.warning('No more DHS sites')
                        break

                if curr_dhs is None:
                    logger.warning('No more DHS sites')
                    break


                if start >= curr_dhs_start and end <= curr_dhs_end:
                    rel_start = start - curr_dhs_start
                    rel_end = end - curr_dhs_start

                    # take care boundaries so we ain't updating nonexistent rows or columns
                    rel_start = max(0, rel_start)
                    rel_end = min(MATRIX_COLUMNS - 1, rel_end)

                    if rel_start < rel_end:
                        result[curr_dhs_index, fragment_length, rel_start:rel_end+1] += 1
        if should_save:
            np.save(self.output_file, result)
        return result
            

fragment_file = snakemake.input.fragment
dhs_file = snakemake.input.dhs
output_file = snakemake.output[0]

preprocessor = Preprocessor(
    fragment_file,
    dhs_file,
    output_file
)
preprocessor.generate_matrix()