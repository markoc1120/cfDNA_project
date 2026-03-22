import logging
import subprocess

import pandas as pd

logger = logging.getLogger(__name__)


def preprocess_dhs(input_file: str, output_file: str, matrix_columns: int):
    with open(input_file, 'rt') as f_in, open(output_file, 'w') as f_out:
        # keeping track of last_midpoint to decide whether the next DHS is inside the window or not,
        # as well as curr_chr, because if we change chr then we need to reset last_midpoint
        last_midpoint, curr_chr = float('-inf'), None

        # line by line iteration
        for i, line in enumerate(f_in):
            chr, start, end = line.split('\t')

            # reset variables
            if chr != curr_chr:
                last_midpoint, curr_chr = float('-inf'), chr

            # parse string -> int
            start, end = int(start), int(end)
            midpoint = (end + start) // 2

            # if there is not enough diff between midpoint (current) and last_midpoint -> overlapping -> continue
            if midpoint - last_midpoint <= matrix_columns:
                logger.info('skip - overlapping')
                continue

            # write line
            f_out.write(line.strip() + '\n')
            # set last_midpoint to midpoint (current)
            last_midpoint = midpoint


def downsample_dhs_files(inputs: list[str], outputs: list[str]):
    coverages = []
    for inp in inputs:
        coverages.append((inp, int(subprocess.check_output(['wc', '-l', inp]).split()[0])))
    min_inp, min_cov = min(coverages, key=lambda x: x[1])

    for inp, out in zip(inputs, outputs):
        if inp == min_inp:
            subprocess.call(['cp', inp, out])
            continue
        df = pd.read_csv(inp, sep='\t', names=['chr', 'start', 'end'])
        df = df.sample(n=min_cov, random_state=42).sort_index()
        df.to_csv(out, sep='\t', header=False, index=False)
