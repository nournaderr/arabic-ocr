from pathlib import Path

# ── Root paths ──────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT_DIR / "data"
MODELS_DIR   = ROOT_DIR / "models"
OUTPUT_DIR   = ROOT_DIR / "output"
PREPROCESS_DIR = OUTPUT_DIR / "preprocess"
SEGMENT_DIR  = OUTPUT_DIR / "segment"

for _d in (DATA_DIR, MODELS_DIR, OUTPUT_DIR, PREPROCESS_DIR, SEGMENT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Debug ────────────────────────────────────────────────────────────────────
DEBUG = False

# ── Preprocessing ────────────────────────────────────────────────────────────
SAUVOLA_WINDOW  = 25
MORPH_KERNEL    = 2
SKEW_ANGLE_MAX  = 45

# ── Segmentation ─────────────────────────────────────────────────────────────
AH_HEIGHT_MIN   = 0.3   # fraction of average character height
AH_HEIGHT_MAX   = 2.0
MIN_CHAR_WIDTH  = 0.1
MAX_CHAR_WIDTH  = 3.0
AH_AREA_MIN     = 0.05
CHOP_MIN_VALLEY = 0.70  # valley must be below this fraction of local max

# ── Features ──────────────────────────────────────────────────────────────────
NORM_SIZE       = 32
GRID_ROWS       = 8
GRID_COLS       = 8
OUTLINE_SAMPLES = 64
HOG_BINS        = 9

# ── Classifier ───────────────────────────────────────────────────────────────
TOP_K    = 5
MIN_CONF = 0.4
HIGH_CONF = 0.85

# ── Language model ───────────────────────────────────────────────────────────
BIGRAM_WEIGHT = 0.3
DAWG_BOOST    = 0.25

# ── Arabic Unicode range ──────────────────────────────────────────────────────
ARABIC_UNICODE_START = 0x0600
ARABIC_UNICODE_END   = 0x06FF

# Letters that carry distinguishing dots (body classification ignores these dots)
DOT_CHARS = set("بتثنيجحخذرزسشصضطظعغفقكلمنهوي")
