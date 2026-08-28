# app/services/identification.py
# ================================================================
# NHẬN DẠNG 1:N - CHỈ 1 KHUÔN MẶT (ỔN ĐỊNH)
# ================================================================

import time
from app.config.settings import NGUONG_COSINE_DISTANCE, MTCNN_DETECT_SCALE
from app.utils.logger import logger


class DichVuNhanDang:

    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.detector = recognizer.embedder.detector
        self.embedder = recognizer.embedder

    def nhan_dang(self, anh_bgr, threshold=NGUONG_COSINE_DISTANCE):
        """
        Nhận dạng 1:N - CHỈ LẤY 1 KHUÔN MẶT LỚN NHẤT
        """
        t0 = time.time()
        
        # ============================================================
        # BƯỚC 1: PHÁT HIỆN KHUÔN MẶT (DÙNG NHANH)
        # ============================================================
        box, prob = self.detector.phat_hien_nhanh(
            anh_bgr, 
            scale=MTCNN_DETECT_SCALE
        )
        
        if box is None:
            # Thử lại với detect chuẩn nếu detect nhanh thất bại
            box, prob = self.detector.phat_hien(anh_bgr)
            if box is None:
                return {
                    "success": False,
                    "message": "Không phát hiện được khuôn mặt.",
                    "results": []
                }
        
        # ============================================================
        # BƯỚC 2: CẮT ROI VỚI MARGIN
        # ============================================================
        x1, y1, x2, y2 = box
        margin = int((x2 - x1) * 0.1)
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(anh_bgr.shape[1], x2 + margin)
        y2 = min(anh_bgr.shape[0], y2 + margin)
        
        face_roi = anh_bgr[y1:y2, x1:x2]
        
        if face_roi.size == 0:
            return {
                "success": False,
                "message": "Không thể cắt khuôn mặt.",
                "results": []
            }
        
        # ============================================================
        # BƯỚC 3: TRÍCH XUẤT EMBEDDING
        # ============================================================
        embedding = self.embedder.trich_xuat_nhan_dien(
            face_roi,
            use_advanced=True,
            use_ensemble=False
        )
        
        if embedding is None:
            return {
                "success": False,
                "message": "Không trích xuất được embedding.",
                "results": []
            }

        # ============================================================
        # BƯỚC 4: NHẬN DẠNG
        # ============================================================
        ket_qua = self.recognizer.identify(embedding, threshold)
        ket_qua["processing_time_ms"] = (time.time() - t0) * 1000
        
        return ket_qua