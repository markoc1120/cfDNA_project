import numpy as np
import logging
import matplotlib.pyplot as plt
from .base import TestStatistic

logger = logging.getLogger(__name__)

class PFEStatistic(TestStatistic):
    @property
    def name(self):
        return 'pfe'

    def calculate(self, matrix: np.ndarray):
        window_size = self.config.get('window_size', 2000)
        min_len = self.config.get('min_length', 100)
        max_len = self.config.get('max_length', 300)
        matrix_shift = self.config.get('matrix_shift', 250)

        _, matrix_columns = matrix.shape
        start_min, end_max = matrix_shift, matrix_columns - matrix_shift

        total_windows = (end_max - start_min) // window_size

        positions, pfe_scores = [], []
        for idx, start in enumerate(range(start_min, start_min + total_windows * window_size, window_size)):
            end = start + window_size

            if end > end_max:
                break

            # fragment counts per length across this window
            region_counts = matrix[:, start:end]
            counts_per_length = np.sum(region_counts, axis=1)

            # restrict to specified length range
            counts_len_range = counts_per_length[min_len:max_len + 1]
            total = np.sum(counts_len_range)
            if total > 0:
                probs = counts_len_range / total
                mask = probs > 0  # avoid entropy being nan due to nan values
                entropy = -np.sum(probs[mask] * np.log2(probs[mask]))
            else:
                entropy = 0.0

            positions.append((start - matrix_shift, end - matrix_shift))
            pfe_scores.append(entropy)

        return {
            'positions': np.array(positions, dtype=np.int32),
            'pfe_scores': np.array(pfe_scores),
            'length_range': (min_len, max_len),
            'window_size': window_size,
        }

    def visualize(self, statistic_data, output_paths):
        pos = statistic_data['positions']
        scores = statistic_data['pfe_scores']

        table_data = []
        for (s, e), score in zip(pos, scores):
            table_data.append([int(s), int(e), round(float(score), 3)])

        fig = plt.figure(figsize=(8, 4))
        ax = plt.gca()
        ax.axis('off')

        col_labels = ['Window start', 'Window end', 'PFE']
        table = plt.table(
            cellText=table_data,
            colLabels=col_labels,
            loc='center',
            cellLoc='center'
        )

        plt.title('PFE summary table', fontweight='bold')        
        plt.show()
        fig.savefig(output_paths, dpi=300)
        plt.close()
