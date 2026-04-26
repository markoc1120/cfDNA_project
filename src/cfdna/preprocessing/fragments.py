import gzip
import logging
from collections import deque
from typing import Deque

import numpy as np

logger = logging.getLogger(__name__)

VALID_CHRS = frozenset({f'chr{i}' for i in range(1, 23)} | {'chrX', 'chrY'})


class Preprocessor:
    def __init__(
        self,
        fragment_file: str,
        dhs_files: list[str] | str,
        output_files: list[str] | str,
        output_covs: list[str] | str,
        matrix_rows: int,
        matrix_columns: int,
        matrix_shift: int,
    ):
        self.fragment_file = fragment_file
        self.dhs_files = [dhs_files] if isinstance(dhs_files, str) else list(dhs_files)
        self.output_files = [output_files] if isinstance(output_files, str) else list(output_files)
        self.output_covs = [output_covs] if isinstance(output_covs, str) else list(output_covs)
        if not (len(self.dhs_files) == len(self.output_files) == len(self.output_covs)):
            raise ValueError('dhs_files, output_files, output_covs must be the same length')

        self.matrix_rows = matrix_rows
        self.matrix_columns = matrix_columns
        self.matrix_columns_half = matrix_columns // 2
        self.matrix_shift = matrix_shift

        self.DHS_sites = []
        self.initial_DHS_lengths = []

    def read_dhs_to_memory(self, dhs_file):
        sites: Deque[tuple[int, str]] = deque()
        with open(dhs_file, 'rt') as f:
            for line in f:
                chr, start, end = line.split('\t')
                midpoint = (int(end) + int(start)) // 2
                sites.append((midpoint, chr))
        return sites, len(sites)

    def get_curr_dhs(self, idx: int) -> tuple:
        if not self.DHS_sites[idx]:
            return None, None, None
        curr_dhs_midpoint, chr = self.DHS_sites[idx].popleft()
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
    def generate_matrix(self, should_save=True) -> np.ndarray:
        n = len(self.dhs_files)
        self.DHS_sites = []
        self.initial_DHS_lengths = []
        for dhs in self.dhs_files:
            sites, length = self.read_dhs_to_memory(dhs)
            self.DHS_sites.append(sites)
            self.initial_DHS_lengths.append(length)

        results = [np.zeros((self.matrix_rows, self.matrix_columns)) for _ in range(n)]
        curr_dhs_starts = [None] * n
        curr_dhs_ends = [None] * n
        curr_chrs = [None] * n
        for i in range(n):
            curr_dhs_starts[i], curr_dhs_ends[i], curr_chrs[i] = self.get_curr_dhs(i)

        with gzip.open(self.fragment_file, 'rt') as f:
            for line in f:
                chr, start, end, gc_fraction = self.parse_fragment(line)
                if chr not in VALID_CHRS:
                    continue

                fragment_midpoint, fragment_length = (start + end) // 2, end - start

                # if the fragment is too long skip and log it for now
                if fragment_length >= self.matrix_rows:
                    logger.warning(
                        f'Skipped fragment due to too high length:\nstart:{start}\nend:{end}'
                    )
                    continue

                all_exhausted = True
                for i in range(n):
                    if curr_dhs_ends[i] is None:
                        continue
                    all_exhausted = False

                    # move dhs until to the fragments' chromosome is reached
                    while curr_dhs_ends[i] and chr != curr_chrs[i]:
                        curr_dhs_starts[i], curr_dhs_ends[i], curr_chrs[i] = self.get_curr_dhs(i)
                        if curr_dhs_ends[i] is None:
                            # logger.warning('No more DHS sites')
                            break

                    # move dhs until we have overlapping fragments
                    while (
                        curr_dhs_ends[i]
                        and chr == curr_chrs[i]
                        and fragment_midpoint > curr_dhs_ends[i]
                    ):
                        curr_dhs_starts[i], curr_dhs_ends[i], curr_chrs[i] = self.get_curr_dhs(i)
                        if curr_dhs_ends[i] is None:
                            # logger.warning('No more DHS sites')
                            break

                    # skip if no more dhs sites for this DHS file
                    if curr_dhs_ends[i] is None:
                        continue

                    # move fragments that are not overlapping and in the previous chromosome from the dhs point of view
                    if chr != curr_chrs[i]:
                        continue

                    rel_midpoint = fragment_midpoint - curr_dhs_starts[i]

                    # only track fragments those are in our boundaries
                    if rel_midpoint >= 0 and rel_midpoint < self.matrix_columns:
                        results[i][fragment_length, rel_midpoint] += gc_fraction

                # break if all DHS files are exhausted
                if all_exhausted:
                    break

        if should_save:
            shift = self.matrix_shift
            end_col = self.matrix_columns - shift
            for i in range(n):
                logger.info(
                    f'Saving matrix for {self.output_files[i]} and cov for {self.output_covs[i]}'
                )
                sliced_result = results[i][:, shift:end_col]
                np.save(self.output_files[i], sliced_result)
                with open(self.output_covs[i], 'w') as cov_f:
                    cov_f.write(str(float(np.sum(sliced_result))))
            return results[0][:, shift:end_col]

        return results[0]
