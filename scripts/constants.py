ENV_NAME = "cfDNA_project"
RESULT_DIR = "../data/test/"
INPUT_DIR = "../raw_data/test/"

# zcat EE87920.hg38.frag.gz | awk -F'\t' '{print $3 - $2}' | sort -nr | head -n 1 -> 262
MATRIX_ROWS = int(262 * 1.5)  # add 50% threshold
MATRIX_COLUMNS = 2000
