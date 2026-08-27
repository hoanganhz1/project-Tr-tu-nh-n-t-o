# app/services/verification.py
# ================================================================
# XÁC MINH 1:1 - CÓ CHUẨN HÓA
# ================================================================

from app.config.settings import NGUONG_COSINE_DISTANCE


class DichVuXacMinh:
    def __init__(self, recognizer):
        self.recognizer = recognizer
        
        # ✅ Lấy embedder để chuẩn hóa
        self.embedder = recognizer.embedder

    # ============================================================
    # XÁC MINH CƠ BẢN (KHÔNG CHUẨN HÓA THÊM)
    # ============================================================

    def xac_minh(
        self,
        anh_bgr,
        user_id,
        threshold=NGUONG_COSINE_DISTANCE
    ):
        """Xác minh cơ bản - không chuẩn hóa thêm"""
        
        embedding = self.recognizer.embedder.trich_xuat(
            anh_bgr,
            use_advanced=False  # ❌ Không chuẩn hóa
        )

        if embedding is None:
            return {
                "success": False,
                "message": "Không phát hiện khuôn mặt.",
                "results": []
            }

        return self.recognizer.verify(
            embedding,
            user_id,
            threshold
        )

    # ============================================================
    # ✅ XÁC MINH CÓ CHUẨN HÓA NÂNG CAO
    # ============================================================

    def xac_minh_normalized(
        self,
        anh_bgr,
        user_id,
        threshold=NGUONG_COSINE_DISTANCE,
        use_advanced=True
    ):
        """
        Xác minh với chuẩn hóa nâng cao
        
        Args:
            anh_bgr: Ảnh BGR
            user_id: ID cần xác minh
            threshold: Ngưỡng
            use_advanced: Có dùng chuẩn hóa nâng cao không
        """
        
        # ✅ Trích xuất embedding với chuẩn hóa
        if use_advanced:
            # Dùng detector để chuẩn hóa ảnh trước
            from app.core.detector import PhatHienKhuonMat
            detector = PhatHienKhuonMat()
            
            # Chuẩn hóa ảnh
            anh_normalized = detector.chuan_hoa_anh(anh_bgr)
            
            # Trích xuất embedding từ ảnh đã chuẩn hóa
            embedding = self.recognizer.embedder.trich_xuat(
                anh_normalized,
                use_advanced=True
            )
        else:
            embedding = self.recognizer.embedder.trich_xuat(
                anh_bgr,
                use_advanced=False
            )

        if embedding is None:
            return {
                "success": False,
                "message": "Không phát hiện khuôn mặt.",
                "results": []
            }

        return self.recognizer.verify(
            embedding,
            user_id,
            threshold
        )