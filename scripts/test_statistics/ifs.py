import numpy as np
import logging
import matplotlib.pyplot as plt
from .base import TestStatistic

logger = logging.getLogger(__name__)

class IFSStatistic(TestStatistic):
    @property
    def name(self):
        return 'ifs'

    def calculate(self, matrix: np.ndarray):
        window_size = self.config.get('window_size', 125)
        matrix_shift = self.config.get('matrix_shift', 250)

        matrix_rows, matrix_columns = matrix.shape
        lengths = np.arange(matrix_rows)

        total_counts = matrix.sum()
        if total_counts == 0:
            L = 1
            total_counts = 1
        else:
            counts_per_length = matrix.sum(axis=1)
            L = np.average(lengths, weights=counts_per_length)

        num_windows = matrix_columns // window_size
        positions, ifs_scores = [], []

        for i in range(num_windows):
            start, end = i * window_size, (i + 1) * window_size

            if not i % 10 or i == num_windows - 1:
                progress = int(i / max(1, num_windows - 1) * 100)
                logger.info(f'IFS calculation progress: {progress}%')

            region = matrix[:, start:end]
            n = region.sum()
            if n > 0:
                n_norm = n / total_counts
                l = np.average(lengths, weights=region.sum(axis=1))
                ifs = n_norm * (1.0 + l / L)
            else:
                ifs = 0.0

            if start >= matrix_shift and end <= (matrix_columns - matrix_shift):
                positions.append((start - matrix_shift, end - matrix_shift))
                ifs_scores.append(ifs)

        return {
            'positions': np.array(positions, dtype=np.int32),
            'ifs_scores': np.array(ifs_scores),
            'global_avg_length': np.array([L]),
        }

    def visualize(self, statistic_data, output_paths):
        fig = plt.figure(figsize=(8, 4))
        pos = statistic_data['positions']
        centers = [(s + e) // 2 for s, e in pos]
        plt.plot(centers, statistic_data['ifs_scores'], label='IFS')

        plt.axvline(x=2000, color='red', linestyle='--', linewidth=2, label='DHS site at 2000')
        plt.xlabel('Relative midpoint positions')
        plt.ylabel('IFS score')
        plt.title(f'IFS across windows (window_size={self.config.get("window_size", 125)})')
        plt.legend()
        plt.tight_layout()
        plt.show()
        fig.savefig(output_paths, dpi=300)
        plt.close()
