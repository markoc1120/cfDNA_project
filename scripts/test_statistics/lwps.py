import numpy as np
import logging
import matplotlib.pyplot as plt
from .base import TestStatistic

logger = logging.getLogger(__name__)


class LWPSStatistic(TestStatistic):
    
    @property
    def name(self):
        return 'lwps'
        
    def calculate(self, matrix):
        # get parameters from config or use constants
        LWPS_WINDOW_SIZE = self.config.get('window_size', 120)
        LWPS_LOWER_THRESHOLD = self.config.get('lower_threshold', 120)
        LWPS_UPPER_THRESHOLD = self.config.get('upper_threshold', 180)
        MATRIX_SHIFT = self.config.get('matrix_shift', 250)
        LWPS_NUM_POSITIONS = matrix.shape[1] - 2 * MATRIX_SHIFT
        
        lwps = np.zeros(LWPS_NUM_POSITIONS)
        
        # precompute fragment_data to avoid O(n^3)
        # TODO: maybe numpy matrix operations would be faster
        fragment_data = []
        for fragment_length in range(matrix.shape[0]):
            # filtering out fragments for 120-180 bp length range
            if LWPS_LOWER_THRESHOLD <= fragment_length <= LWPS_UPPER_THRESHOLD:
                continue
                
            for rel_midpoint in range(matrix.shape[1]):
                count = matrix[fragment_length, rel_midpoint]
                if count > 0:
                    frag_start = rel_midpoint - fragment_length // 2
                    frag_end = rel_midpoint + fragment_length // 2
                    fragment_data.append({
                        'start': frag_start,
                        'end': frag_end,
                        'count': count,
                    })
        
        # sliding window algo O(n^2) -> calculating lwps for each positions 180,181,...,1818, 1819 O(n^2)
        for pos in range(MATRIX_SHIFT, LWPS_NUM_POSITIONS + MATRIX_SHIFT):
            # matrix indexing starts from 0
            pos_idx = pos - MATRIX_SHIFT
            
            if pos_idx % 100 == 0 or pos_idx == LWPS_NUM_POSITIONS - 1:
                progress = round(pos_idx / LWPS_NUM_POSITIONS * 100)
                logger.info(f'Progress: {progress}%')
            
            # for position 180 -> window [120, 240]
            window_start = pos - LWPS_WINDOW_SIZE // 2
            window_end = pos + LWPS_WINDOW_SIZE // 2
            
            # fragments which are outside of this [-60, 60], starts before -60 and ends after 60
            spanning_count = 0
            # fragments those either start or end in the window
            internal_endpoints = 0
            
            for frag in fragment_data:
                frag_start, frag_end, count = frag['start'], frag['end'], frag['count']
                
                # count spanning fragments
                if frag_start <= window_start and frag_end >= window_end:
                    spanning_count += count
                
                # count internal endpoints
                if window_start <= frag_start <= window_end:  # starting in the window
                    internal_endpoints += count
                if window_start <= frag_end <= window_end:    # ending in the window
                    internal_endpoints += count
            
            lwps[pos_idx] = spanning_count - internal_endpoints
        
        return lwps
        
    def visualize(self, statistic_data, output_paths):
        LWPS_NUM_POSITIONS = len(statistic_data)
        
        fig = plt.figure(figsize=(8, 4))
        x_positions = np.arange(LWPS_NUM_POSITIONS)
        plt.plot(x_positions, statistic_data)
        plt.xlabel('Relative midpoint positions')
        plt.ylabel('L-WPS score')
        plt.title('Relative midpoint positions VS L-WPS score')
        plt.show()
        fig.savefig(output_paths, dpi=300)
        plt.close()
