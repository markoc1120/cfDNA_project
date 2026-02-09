import numpy as np
import logging
import matplotlib.pyplot as plt
from .base import TestStatistic

logger = logging.getLogger(__name__)

class OCFStatistic(TestStatistic):
    @property
    def name(self):
        return 'ocf'
    
    def preprocess_matrix(self, matrix: np.ndarray):
        matrix_rows, matrix_columns = matrix.shape
        fragment_starts, fragment_ends = np.zeros(matrix_columns), np.zeros(matrix_columns)

        for fragment_length in range(matrix_rows):
            # skip empty rows
            if not np.any(matrix[fragment_length, :]):
                continue
                
            for rel_midpoint in range(matrix_columns):
                count = matrix[fragment_length, rel_midpoint]
                if count > 0:
                    # calculate relative fragment start and end
                    frag_start = rel_midpoint - fragment_length // 2
                    # calculate frag_end as exclusive
                    frag_end = frag_start + fragment_length

                    if 0 <= frag_start < matrix_columns:
                        fragment_starts[frag_start] += count
                        
                    if 0 <= frag_end < matrix_columns:
                         fragment_ends[frag_end] += count
                        
        logger.info("Finished preprocessing OCF matrix.")
        return fragment_starts, fragment_ends
        

    def calculate(self, matrix: np.ndarray):
        fragment_starts, fragment_ends = self.preprocess_matrix(matrix)
        
        matrix_columns = len(fragment_starts)
        matrix_shift = self.config.get('matrix_shift', 250)
        window_half = self.config.get('window_half', 10)
        center_offset = self.config.get('center_offset', 60)

        num_scores = matrix_columns - 2 * matrix_shift
        ocf_scores = np.zeros(num_scores)

        for pos in range(matrix_shift, matrix_columns - matrix_shift):
            pos_idx = pos - matrix_shift
            
            if pos_idx % 100:
                progress = round((pos_idx + 1) / num_scores * 100)
                logger.info(f'OCF calculation progress: {progress}%')
            
            left_start = pos - center_offset - window_half
            left_end = pos - center_offset + window_half + 1
            
            right_start = pos + center_offset - window_half
            right_end = pos + center_offset + window_half + 1

            if left_start < 0 or right_end > matrix_columns:
                ocf_scores[pos_idx] = 0
                continue
                
            starts_in_left_window = np.sum(fragment_starts[max(0, left_start):min(matrix_columns, left_end)])
            ends_in_left_window = np.sum(fragment_ends[max(0, left_start):min(matrix_columns, left_end)])
            
            starts_in_right_window = np.sum(fragment_starts[max(0, right_start):min(matrix_columns, right_end)])
            ends_in_right_window = np.sum(fragment_ends[max(0, right_start):min(matrix_columns, right_end)])
            
            true_signal = ends_in_left_window + starts_in_right_window
            background = starts_in_left_window + ends_in_right_window
                
            # ocf_scores[pos_idx] = (true_signal - background) / (true_signal + background)
            ocf_scores[pos_idx] = true_signal - background

        return ocf_scores

    def visualize(self, statistic_data, output_paths):
        fig = plt.figure(figsize=(8, 4))
        x_positions = np.arange(len(statistic_data))
        plt.plot(x_positions, statistic_data)
        plt.xlabel('Relative midpoint positions')
        plt.ylabel('OCF score')
        plt.title('Relative midpoint positions VS OCF score')
        plt.show()
        fig.savefig(output_paths, dpi=300)
        plt.close()
