DATA = config['data']
MATRIX = config['matrix']
MODEL = config['model']
STAGES = config.get('stages', {})
TRAINING = config.get('training', {})
PREPROCESSING = config.get('preprocessing', {})
NEGATIVE_DHS = config.get('negative_dhs', {})

# preprocessing
INPUT_FRAGS_DIR = DATA["input_frags_dir"]
FRAG_FILENAME = DATA["fragment_filename"]
TRAINING_METADATA_PATH = DATA["training_metadata_path"]
TRAIN_DHS_DIR = DATA["training_dhs_dir"]
TRAIN_OUTPUT_DIR = DATA["training_output_dir"]
TRAIN_BASE_MATRICES_DIR = DATA["training_base_matrices"]
MIN_COV_FILE = DATA["training_min_coverage_file"]
MIN_COV_OVERRIDE = PREPROCESSING.get("min_cov")
BIN_EDGES_FILE = DATA["training_bin_edges_file"]
MATRIX_COLUMNS = MATRIX["columns"]
MATRIX_ROWS = MATRIX["rows"]
MATRIX_SHIFT = MATRIX["shift"]

GENERATE_BASE_MATRICES = STAGES.get("generate_base_matrices", True)

COVERAGE_HANDLING = PREPROCESSING.get("coverage_handling", "downsample")
_COVERAGE_SUFFIX_BY_MODE = {
    "downsample": "downsampled",
    "normalize": "normalized",
    "none": "",
}
if COVERAGE_HANDLING not in _COVERAGE_SUFFIX_BY_MODE:
    raise ValueError(
        f"preprocessing.coverage_handling must be one of "
        f"{list(_COVERAGE_SUFFIX_BY_MODE)}, got {COVERAGE_HANDLING}"
    )
COVERAGE_SUFFIX = _COVERAGE_SUFFIX_BY_MODE[COVERAGE_HANDLING]

# inference
INFERENCE_DHS_DIR = DATA["inference_dhs_dir"]
INFERENCE_FRAGS_DIR = DATA["inference_frags_dir"]
INFERENCE_METADATA_PATH = DATA["inference_metadata_path"]
INFERENCE_OUTPUT_DIR = DATA["inference_output_dir"]
INFERENCE_BASE_MATRICES_DIR = DATA["inference_base_matrices"]
INFERENCE_USE_REBINNED = MODEL.get("use_rebinned", True)
INFERENCE_OUTPUT_SUFFIX = "latent.npz" if MODEL["name"] == "vae" else "score.txt"

if INFERENCE_USE_REBINNED:
    INPUT_TYPE = 'rebinned'
elif COVERAGE_HANDLING == 'none':
    raise ValueError("coverage_handling='none' requires use_rebinned=true")
else:
    INPUT_TYPE = COVERAGE_SUFFIX

# accessibility score
ACCESSIBILITY_DIR = DATA["accessibility_scores_dir"]

# benchmark
FINAL_MATRICES_DIR = DATA["final_matrices_dir"]

ACCESSIBILITY_STATS = ['pfe', 'lwps', 'ifs', 'fdi', 'ocf']
BENCH_STATS = [MODEL['name']]

if STAGES.get('accessibility_scores', False):
    BENCH_STATS.extend(ACCESSIBILITY_STATS)
