# app/config/settings.py
# ================================================================
# CẤU HÌNH HỆ THỐNG - NÂNG CAO
# ================================================================

import os
import json
import torch


# ============================================================
# THƯ MỤC
# ============================================================

THU_MUC_GOC = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

THU_MUC_DATA = os.path.join(
    THU_MUC_GOC,
    "data"
)

TEP_CSDL = os.path.join(
    THU_MUC_DATA,
    "face_database.json"
)

TEP_THRESHOLD = os.path.join(
    THU_MUC_DATA,
    "threshold.json"
)


# ============================================================
# MODEL
# ============================================================

TEN_MODEL = "vggface2"
CHIEU_EMBEDDING = 512

THIET_BI = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# NHẬN DIỆN - THRESHOLD
# ============================================================

NGUONG_NHAN_DANG = 0.35
NGUONG_XAC_MINH = 0.30
NGUONG_COSINE_DISTANCE = NGUONG_NHAN_DANG


# ============================================================
# ✅ CÀI ĐẶT NÂNG CAO
# ============================================================

# Chuẩn hóa ảnh
SU_DUNG_CLAHE = True              # Cân bằng histogram
SU_DUNG_DENOISE = True            # Giảm nhiễu
SU_DUNG_SHARPEN = True            # Làm sắc nét
ALPHA_CONTRAST = 1.2              # Độ tương phản
BETA_BRIGHTNESS = 10              # Độ sáng

# Ensemble embedding
SU_DUNG_ENSEMBLE = True           # Gộp nhiều embedding
SO_ANH_ENSEMBLE = 5               # Số ảnh cho ensemble
GOC_XOAY = [-5, 5]                # Góc xoay

# Lọc chất lượng
DO_SANG_TOI_THIEU = 10            # Độ sáng tối thiểu
KICH_THUOC_TOI_THIEU = 50         # Kích thước tối thiểu
NORM_TOI_THIEU = 0.8              # Norm tối thiểu
NORM_TOI_DA = 1.2                 # Norm tối đa
SO_EMBEDDING_TOI_THIEU = 5        # Số embedding tối thiểu để đăng ký

# Cách so sánh
PHUONG_PHAP_SO_SANH = "cosine"    # "cosine", "euclidean", "manhattan"


# ============================================================
# HÀM LOAD/SAVE THRESHOLD
# ============================================================

def load_threshold():
    """Tải threshold từ file"""
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
        else:
            print("[Config] Chưa có file threshold, dùng giá trị mặc định")
            return False
            
    except Exception as e:
        print(f"[Config] Lỗi tải threshold: {e}")
        return False


def save_threshold(nhan_dang=None, xac_minh=None):
    """Lưu threshold vào file"""
    global NGUONG_NHAN_DANG, NGUONG_XAC_MINH, NGUONG_COSINE_DISTANCE
    
    try:
        os.makedirs(THU_MUC_DATA, exist_ok=True)
        
        if nhan_dang is not None:
            NGUONG_NHAN_DANG = nhan_dang
        if xac_minh is not None:
            NGUONG_XAC_MINH = xac_minh
            
        NGUONG_COSINE_DISTANCE = NGUONG_NHAN_DANG
        
        data = {
            "nhan_dang": NGUONG_NHAN_DANG,
            "xac_minh": NGUONG_XAC_MINH,
            "version": 1
        }
        
        with open(TEP_THRESHOLD, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"[Config] Đã lưu threshold: 1N={NGUONG_NHAN_DANG}, 11={NGUONG_XAC_MINH}")
        return True
        
    except Exception as e:
        print(f"[Config] Lỗi lưu threshold: {e}")
        return False


# ============================================================
# TỰ ĐỘNG LOAD
# ============================================================

load_threshold()
os.makedirs(THU_MUC_DATA, exist_ok=True)