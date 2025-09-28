import numpy as np
import logging
import matplotlib.pyplot as plt
from .base import TestStatistic

logger = logging.getLogger(__name__)


class FDIStatistic(TestStatistic):
    
    @property
    def name(self):
        return "fdi"
    
    def calculate(self, matrix):
        # Get parameters from config
        X = self.config.get('x', 0.999)
        ENDPOINT_WINDOW = self.config.get('endpoint_window', 10)
        WINDOW_SIZE = self.config.get('window_size', 200)
        STEP_SIZE = self.config.get('step_size', 20)

        logger.info(
            f"calculating FDI with x={X}, endpoint_window={ENDPOINT_WINDOW}, window_size={WINDOW_SIZE}, step_size={STEP_SIZE}"
        )

        # convert matrix to reads format
        reads = self.matrix_to_reads(matrix)
        logger.info(f"converted matrix to {len(reads)} reads")

        # calculate coverage array
        coverage = self.calculate_coverage(matrix)

        # calculate endpoint dispersion matrix
        dispersion_matrix = self.calculate_endpoint_dispersion(reads, matrix.shape[1], X, ENDPOINT_WINDOW)

        # calculate FDI in sliding windows
        fdi_results = self.calculate_windowed_fdi(
            coverage, dispersion_matrix, WINDOW_SIZE, STEP_SIZE
        )

        return fdi_results

    # # TODO: same logic as in LWPSStatistic
    def matrix_to_reads(self, matrix):    
        reads = []

        for fragment_length in range(matrix.shape[0]):
            # filtering out fragments based on lengths, maybe it makes sense
            # if 120 <= fragment_length <= 180:
            #     continue

            for rel_midpoint in range(matrix.shape[1]):
                count = matrix[fragment_length, rel_midpoint]

                if count > 0:
                    frag_start = rel_midpoint - fragment_length // 2
                    frag_end = rel_midpoint + fragment_length // 2

                    for _ in range(int(count)):
                        reads.append({
                            'start': frag_start,
                            'end': frag_end,
                            'count': count,
                        })
        return reads

    # TODO: same logic as in visualize_matrix.py
    def calculate_coverage(self, matrix: np.ndarray) -> np.ndarray:
        matrix_columns = matrix.shape[1]

        coverage = np.zeros(matrix_columns)

        for fragment_length in range(matrix.shape[0]):
            for rel_midpoint in range(matrix_columns):
                count = matrix[fragment_length, rel_midpoint]
                if count > 0:
                    # calculate start and end positions from midpoint and length
                    start_pos = rel_midpoint - fragment_length // 2
                    end_pos = rel_midpoint + fragment_length // 2

                    # make sure we stay in our boundaries
                    start_pos = max(0, start_pos)
                    end_pos = min(matrix_columns, end_pos)

                    # update coverage
                    if start_pos < end_pos:
                        coverage[start_pos:end_pos] += count

        return coverage


    def calculate_endpoint_dispersion(self, reads, matrix_columns, x, endpoint_window):
        # dispersion_matrix[:, 0] = endpoint counts, dispersion_matrix[:, 1] = dispersion values
        dispersion_matrix = np.zeros((matrix_columns, 2))

        # count endpoints at each position
        for read in reads:
            start, end, fragment_length = read['start'], read['end'], read['count']

            # adjust positions to valid range
            start = max(endpoint_window, min(start, matrix_columns - endpoint_window - 1))
            end = max(endpoint_window, min(end, matrix_columns - endpoint_window - 1))

            # count endpoints
            dispersion_matrix[start, 0] += 1
            dispersion_matrix[end, 0] += 1

        # calculate dispersion values
        for read in reads:
            start, end, fragment_length = read['start'], read['end'], read['count']

            # Adjust positions to valid range
            start = max(endpoint_window, min(start, matrix_columns - endpoint_window - 1))
            end = max(endpoint_window, min(end, matrix_columns - endpoint_window - 1))

            # calculate local density and dispersion for start endpoint
            local_density_start = np.sum(
                dispersion_matrix[start-endpoint_window:start+endpoint_window+1, 0]
            ) - 1
            dispersion_matrix[start, 1] += x ** local_density_start

            # calculate local density and dispersion for end endpoint  
            local_density_end = np.sum(
                dispersion_matrix[end-endpoint_window:end+endpoint_window+1, 0]
            ) - 1
            dispersion_matrix[end, 1] += x ** local_density_end

        return dispersion_matrix

    def calculate_windowed_fdi(self, coverage, dispersion_matrix, window_size, step_size):
        matrix_columns = len(coverage)
        num_windows = (matrix_columns - window_size) // step_size + 1

        positions, fdi_scores, coverage_stds, endpoint_dispersions = [], [], [], []
        for i in range(num_windows):
            start = i * step_size
            end = start + window_size + 1

            if not i % 10:
                progress = round(i / (num_windows - 1) * 100)
                logger.info(f'FDI calculation progress: {progress}%')

            if end > matrix_columns:
                break

            # calculate coverage standard deviation in window
            window_coverage = coverage[start:end]
            coverage_std = np.std(window_coverage)

            # calculate endpoint dispersion in window
            window_endpoint_counts = np.sum(dispersion_matrix[start:end, 0])
            if window_endpoint_counts == 0:  # avoid 0 division
                window_endpoint_counts = 1

            # calculate average endpoint dispersion
            window_dispersion_sum = np.sum(dispersion_matrix[start:end, 1])
            avg_endpoint_dispersion = window_dispersion_sum / window_endpoint_counts

            # calculate FDI score
            fdi_score = coverage_std * avg_endpoint_dispersion

            positions.append((start, end))
            fdi_scores.append(fdi_score)
            coverage_stds.append(coverage_std)
            endpoint_dispersions.append(avg_endpoint_dispersion)

        return {
            'positions': positions,
            'fdi_scores': np.array(fdi_scores),
            'coverage_std': np.array(coverage_stds),
            'endpoint_dispersion': np.array(endpoint_dispersions)
        }
        
    def visualize(self, statistic_data, output_paths):
        fig = plt.figure(figsize=(8, 4))
        
        window_centers = [(start + end) // 2 for start, end in statistic_data['positions']]

        plt.plot(window_centers, statistic_data['fdi_scores'], 'b-', marker='o', markersize=4, linewidth=1)
        plt.axvline(x=1000, color='red', linestyle='--', linewidth=2, label='DHS site at 1000') # mark the DHS site
        plt.yscale('log') # use log scale for y-axis since values are very small
        plt.xlabel('Relative midpoint positions')
        plt.ylabel('FDI score (log scale)')
        plt.title(
            f'FDI scores across sliding windows '
            f'(window_size={self.config["window_size"]}, step_size={self.config["step_size"]})'
        )
        plt.legend()
        plt.show()
        fig.savefig(output_paths, dpi=300)
        plt.close()
