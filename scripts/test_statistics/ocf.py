import numpy as np
import logging
import matplotlib.pyplot as plt
from .base import TestStatistic

logger = logging.getLogger(__name__)

class OCFStatistic(TestStatistic):
    @property
    def name(self):
        return 'ocf'

    def calculate(self, matrix: np.ndarray):
        matrix_rows, matrix_columns = matrix.shape
        matrix_shift = self.config.get('matrix_shift', 250)
        window_half = self.config.get('window_half', 10)
        center_offset = self.config.get('center_offset', 60)

        ocf_scores = np.zeros(matrix_columns)

        for pos in range(matrix_shift, matrix_columns - matrix_shift):
            left_start = pos - center_offset - window_half
            left_end = pos - center_offset + window_half
            right_start = pos + center_offset - window_half
            right_end = pos + center_offset + window_half

            if left_start < 0 or right_end >= matrix_columns:
                continue

            # U and D endpoints should be based on + and - strand???
            # need to have a matrix that contains +, and - strand information
            

            if pos % 100 == 0 or pos == matrix_columns - matrix_shift - 1:
                progress = round(pos / matrix_columns * 100)
                logger.info(f'OCF calculation progress: {progress}%')

        return ocf_scores[matrix_shift:-matrix_shift]

    def visualize(self, statistic_data, output_paths):
        pass
