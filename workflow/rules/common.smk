DATA = config['data']
MATRIX = config['matrix']
MODEL = config['model']
STAGES = config.get('stages', {})
TRAINING = config.get('training', {})
PREPROCESSING = config.get('preprocessing', {})
NEGATIVE_DHS = config.get('negative_dhs', {})

# preprocessing
INPUT_FRAGS_DIR = DATA["input_frags_dir"]
TRAIN_DHS_DIR = DATA["training_dhs_dir"]
TRAIN_OUTPUT_DIR = DATA["training_output_dir"]
MIN_COV_FILE = DATA["training_min_coverage_file"]
BIN_EDGES_FILE = DATA["training_bin_edges_file"]
MATRIX_COLUMNS = MATRIX["columns"]
MATRIX_ROWS = MATRIX["rows"]
MATRIX_SHIFT = MATRIX["shift"]

# inference
INFERENCE_DHS_DIR = DATA["inference_dhs_dir"]
INFERENCE_FRAGS_DIR = DATA["inference_frags_dir"]
INFERENCE_OUTPUT_DIR = DATA["inference_output_dir"]
INFERENCE_USE_REBINNED = MODEL.get("use_rebinned", True)

INPUT_TYPE = 'rebinned' if INFERENCE_USE_REBINNED else 'downsampled'

# accessibility score
ACCESSIBILITY_DIR = DATA["accessibility_scores_dir"]

# benchmark
FINAL_MATRICES_DIR = DATA["final_matrices_dir"]

ACCESSIBILITY_STATS = ['pfe', 'lwps', 'ifs', 'fdi', 'ocf']
BENCH_STATS = [MODEL['name']]

if STAGES.get('accessibility_scores', False):
    BENCH_STATS.extend(ACCESSIBILITY_STATS)
