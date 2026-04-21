import gzip
import logging
from collections import deque
from typing import Deque

import numpy as np

logger = logging.getLogger(__name__)


class Preprocessor:
    def __init__(
        self,
        fragment_file: str,
        dhs_file: str,
        output_file: str,
        output_gc_file: str,
        output_cov: str,
        matrix_rows: int,
        matrix_columns: int,
        matrix_shift: int,
    ):
        self.fragment_file = fragment_file
        self.dhs_file = dhs_file
        self.output_file = output_file
        self.output_gc_file = output_gc_file
        self.output_cov = output_cov
        self.DHS_sites = None
        self.initial_DHS_length = None
        self.matrix_rows = matrix_rows
        self.matrix_columns = matrix_columns
        self.matrix_columns_half = matrix_columns // 2
        self.matrix_shift = matrix_shift

    def read_dhs_to_memory(self):
        sites: Deque[tuple[int, str]] = deque()
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
            curr_dhs_midpoint - self.matrix_columns_half,
            curr_dhs_midpoint + self.matrix_columns_half,
            chr,
        )

    def parse_fragment(self, line: str) -> tuple:
        parsed_fragment = line.strip().split('\t')
        chr, start, end = parsed_fragment[0:3]
        gc_fraction = parsed_fragment[-1]
        return chr, int(start), int(end), float(gc_fraction)

    # length vs fragments' relative midpoint
    def generate_matrix(self, should_save=True) -> tuple[np.ndarray, np.ndarray]:
        self.DHS_sites, self.initial_DHS_length = self.read_dhs_to_memory()
        counts = np.zeros((self.matrix_rows, self.matrix_columns), dtype=np.int64)
        gc_sums = np.zeros((self.matrix_rows, self.matrix_columns), dtype=np.float64)
        curr_dhs_start, curr_dhs_end, curr_chr = self.get_curr_dhs()

        with gzip.open(self.fragment_file, 'rt') as f:
            for line in f:
                chr, start, end, gc_fraction = self.parse_fragment(line)
                if chr not in {f'chr{i}' for i in range(1, 23)} | {'chrX', 'chrY'}:
                    continue

                fragment_midpoint, fragment_length = (start + end) // 2, end - start

                # if the fragment is too long skip and log it for now
                if fragment_length >= self.matrix_rows:
                    logger.warning(
                        f'Skipped fragment due to too high length:\nstart:{start}\nend:{end}'
                    )
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
                if rel_midpoint >= 0 and rel_midpoint < self.matrix_columns:
                    counts[fragment_length, rel_midpoint] += 1
                    gc_sums[fragment_length, rel_midpoint] += gc_fraction

        if should_save:
            logger.info(
                f'Saving counts to {self.output_file}, gc sums to {self.output_gc_file}, '
                f'cov to {self.output_cov}'
            )
            col_slice = slice(self.matrix_shift, self.matrix_columns - self.matrix_shift)
            counts = counts[:, col_slice]
            gc_sums = gc_sums[:, col_slice]
            total_cov = int(counts.sum())
            np.save(self.output_file, counts)
            np.save(self.output_gc_file, gc_sums)
            with open(self.output_cov, 'w') as cov_f:
                cov_f.write(str(total_cov))
        return counts, gc_sums
