from constants import MATRIX_COLUMNS


def preprocess_dhs(input_file: str, output_file: str):
    with open(input_file, 'rt') as f_in, open(output_file, 'w') as f_out:
        # keeping track of last_midpoint to decide whether the next DHS is inside the window or not, 
        #as well as curr_chr, because if we change chr then we need to reset last_midpoint
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
            if midpoint - last_midpoint <= MATRIX_COLUMNS:
                logger.info('skip - overlapping')
                continue
            
            # write line
            f_out.write(line.strip() + '\n')
            # set last_midpoint to midpoint (current)
            last_midpoint = midpoint

if 'snakemake' in globals():
    dhs_sorted_file = snakemake.input.dhs_sorted
    dhs_sorted_preprocessed_file = snakemake.output.dhs_sorted_preprocessed

    preprocess_dhs(dhs_sorted_file, dhs_sorted_preprocessed_file)