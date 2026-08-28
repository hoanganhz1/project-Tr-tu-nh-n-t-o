# app/config/constants.py
# ================================================================
# HẰNG SỐ TOÀN CỤC
# ================================================================

# ============================================================
# THRESHOLD
# ============================================================

DEFAULT_COSINE_THRESHOLD = 0.35
MIN_COSINE_THRESHOLD = 0.10
MAX_COSINE_THRESHOLD = 0.80

DEFAULT_NHAN_DANG_THRESHOLD = 0.35
DEFAULT_XAC_MINH_THRESHOLD = 0.30

# ============================================================
# MODEL
# ============================================================

EMBEDDING_DIMENSION = 512
FACE_IMAGE_SIZE = 160
FACE_MIN_SIZE = 20
MTCNN_THRESHOLDS = [0.6, 0.7, 0.7]

# ============================================================
# CAMERA
# ============================================================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_BUFFER_SIZE = 1

# ============================================================
# REGISTRATION
# ============================================================

DEFAULT_SAMPLE_COUNT = 20
MIN_SAMPLE_COUNT = 5
MAX_SAMPLE_COUNT = 50

# ============================================================
# DATABASE
# ============================================================

DB_VERSION = 1
DB_MODEL_NAME = "InceptionResnetV1"
DB_PRETRAINED = "vggface2"
DB_DISTANCE_METRIC = "cosine"

# ============================================================
# UI
# ============================================================

WINDOW_MIN_WIDTH = 1200
WINDOW_MIN_HEIGHT = 750
SIDEBAR_WIDTH = 270
CAMERA_DISPLAY_WIDTH = 640
CAMERA_DISPLAY_HEIGHT = 480

# ============================================================
# MÀU SẮC KHUNG
# ============================================================

COLOR_SUCCESS = (0, 255, 0)
COLOR_FAIL = (0, 0, 255)
COLOR_DETECT = (0, 255, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_BG = (0, 0, 0, 128)