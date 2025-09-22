ENV_NAME = "cfDNA_project"

INPUT_DIR = "../raw_data/test/"
INPUT_DHS_DIR = "../raw_data/dhs/"

RESULT_DIR = "../data/test/"
RESULT_SORTED_DIR = "../data/sorted/"


# zcat EE87920.hg38.frag.gz | awk -F'\t' '{print $3 - $2}' | sort -nr | head -n 1 -> 262
MATRIX_ROWS = int(262 * 1.5)  # add 50% threshold
MATRIX_COLUMNS = 2000

LWPS_WINDOW_SIZE = 120