# app/services/verification.py
# ================================================================
# XÁC MINH 1:1 - ỔN ĐỊNH
# ================================================================

import time
from app.config.settings import NGUONG_COSINE_DISTANCE, MTCNN_DETECT_SCALE
from app.utils.logger import logger


class DichVuXacMinh:
    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.embedder = recognizer.embedder
        self.detector = recognizer.embedder.detector

    def xac_minh(self, anh_bgr, user_id, threshold=NGUONG_COSINE_DISTANCE):
        """
        Xác minh 1:1 - ỔN ĐỊNH
        """
        t0 = time.time()
        
        # PHÁT HIỆN KHUÔN MẶT
        box, prob = self.detector.phat_hien_nhanh(
            anh_bgr, 
            scale=MTCNN_DETECT_SCALE
        )
        
        if box is None:
            box, prob = self.detector.phat_hien(anh_bgr)
            if box is None:
                return {
                    "success": False,
                    "message": "Không phát hiện được khuôn mặt.",
                    "results": []
                }
        
        # CẮT ROI VỚI MARGIN
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
        
        # TRÍCH XUẤT EMBEDDING
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

        # XÁC MINH
        ket_qua = self.recognizer.verify(embedding, user_id, threshold)
        ket_qua["processing_time_ms"] = (time.time() - t0) * 1000
        
        return ket_qua

    def xac_minh_normalized(self, anh_bgr, user_id, threshold=NGUONG_COSINE_DISTANCE, use_advanced=True):
        """Xác minh với chuẩn hóa nâng cao"""
        return self.xac_minh(anh_bgr, user_id, threshold)