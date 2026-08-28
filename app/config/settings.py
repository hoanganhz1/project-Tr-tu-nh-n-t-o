# app/config/settings.py
# ================================================================
# CẤU HÌNH HỆ THỐNG - HỖ TRỢ GPU 60FPS + TỐI ƯU TỐC ĐỘ
# ================================================================

import os
import json
import torch

# ============================================================
# THƯ MỤC
# ============================================================

THU_MUC_GOC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THU_MUC_DATA = os.path.join(THU_MUC_GOC, "data")
TEP_CSDL = os.path.join(THU_MUC_DATA, "face_database.json")
TEP_THRESHOLD = os.path.join(THU_MUC_DATA, "threshold.json")

# ============================================================
# GPU / CUDA - TỰ ĐỘNG PHÁT HIỆN
# ============================================================

CUDA_AVAILABLE = torch.cuda.is_available()
THIET_BI = torch.device("cuda:0" if CUDA_AVAILABLE else "cpu")

if CUDA_AVAILABLE:
    print(f"[Config] ✅ Sử dụng GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Config] ✅ CUDA Version: {torch.version.cuda}")
    print(f"[Config] ✅ PyTorch Version: {torch.__version__}")
else:
    print("[Config] ⚠️ Không tìm thấy GPU, sử dụng CPU")

# Batch size cho embedding (tối ưu GPU)
BATCH_SIZE_EMBEDDING = 16 if CUDA_AVAILABLE else 1

# FPS mặc định
DEFAULT_FPS = 60 if CUDA_AVAILABLE else 30

# ============================================================
# TỐI ƯU HIỆU NĂNG
# ============================================================

# Scale cho MTCNN detect (giảm xuống 0.5 để tăng tốc)
MTCNN_DETECT_SCALE = 0.5

# Sử dụng căn chỉnh nâng cao (tắt để tăng tốc)
USE_ADVANCED_ALIGNMENT = True  

# Sử dụng ensemble embedding (tắt để tăng tốc)
USE_ENSEMBLE = False

# Số embedding tối thiểu để đăng ký
SO_EMBEDDING_TOI_THIEU = 5

# ============================================================
# MODEL
# ============================================================

TEN_MODEL = "vggface2"
CHIEU_EMBEDDING = 512

# ============================================================
# THRESHOLD
# ============================================================

NGUONG_NHAN_DANG = 0.35
NGUONG_XAC_MINH = 0.30
NGUONG_COSINE_DISTANCE = NGUONG_NHAN_DANG

# ============================================================
# CÀI ĐẶT NÂNG CAO
# ============================================================

SU_DUNG_CLAHE = True
SU_DUNG_DENOISE = True
SU_DUNG_SHARPEN = True
ALPHA_CONTRAST = 1.2
BETA_BRIGHTNESS = 10

SU_DUNG_ENSEMBLE = False
SO_ANH_ENSEMBLE = 3
GOC_XOAY = [-5, 5]

DO_SANG_TOI_THIEU = 10
KICH_THUOC_TOI_THIEU = 50
NORM_TOI_THIEU = 0.8
NORM_TOI_DA = 1.2
PHUONG_PHAP_SO_SANH = "cosine"

# ============================================================
# LOAD/SAVE THRESHOLD
# ============================================================

def load_threshold():
    global NGUONG_NHAN_DANG, NGUONG_XAC_MINH, NGUONG_COSINE_DISTANCE
    try:
        if os.path.exists(TEP_THRESHOLD):
            with open(TEP_THRESHOLD, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "nhan_dang" in data:
                    NGUONG_NHAN_DANG = data["nhan_dang"]
                if "xac_minh" in data:
                    NGUONG_XAC_MINH = data["xac_minh"]
                NGUONG_COSINE_DISTANCE = NGUONG_NHAN_DANG
                print(f"[Config] Đã tải threshold: 1N={NGUONG_NHAN_DANG}, 11={NGUONG_XAC_MINH}")
                return True
    except Exception as e:
        print(f"[Config] Lỗi tải threshold: {e}")
    return False

def save_threshold(nhan_dang=None, xac_minh=None):
    global NGUONG_NHAN_DANG, NGUONG_XAC_MINH, NGUONG_COSINE_DISTANCE
    try:
        os.makedirs(THU_MUC_DATA, exist_ok=True)
        if nhan_dang is not None:
            NGUONG_NHAN_DANG = nhan_dang
        if xac_minh is not None:
            NGUONG_XAC_MINH = xac_minh
        NGUONG_COSINE_DISTANCE = NGUONG_NHAN_DANG
        data = {"nhan_dang": NGUONG_NHAN_DANG, "xac_minh": NGUONG_XAC_MINH, "version": 1}
        with open(TEP_THRESHOLD, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"[Config] Lỗi lưu threshold: {e}")
        return False

load_threshold()
os.makedirs(THU_MUC_DATA, exist_ok=True)