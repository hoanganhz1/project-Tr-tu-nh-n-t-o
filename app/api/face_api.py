# app/api/face_api.py
# ================================================================
# FACE API - CHỈ HỖ TRỢ 1 KHUÔN MẶT
# ================================================================

from app.config.settings import NGUONG_NHAN_DANG, NGUONG_XAC_MINH
from app.core.embedder import FaceEmbedder


class FaceAPI:
    """API cho các chức năng nhận diện khuôn mặt"""

    def __init__(
        self,
        registration_service,
        identification_service,
        verification_service
    ):
        self.registration_service = registration_service
        self.identification_service = identification_service
        self.verification_service = verification_service

    def register(self, thong_tin, danh_sach_embedding):
        return self.registration_service.tao_nguoi_dung(
            thong_tin,
            danh_sach_embedding
        )

    def identify(
        self,
        anh_bgr,
        threshold=None
    ):
        """Nhận dạng 1:N - CHỈ 1 KHUÔN MẶT"""
        if threshold is None:
            threshold = NGUONG_NHAN_DANG

        return self.identification_service.nhan_dang(
            anh_bgr,
            threshold=threshold
        )

    def verify(
        self,
        anh_bgr,
        user_id,
        threshold=None
    ):
        if threshold is None:
            threshold = NGUONG_XAC_MINH

        return self.verification_service.xac_minh(
            anh_bgr,
            user_id,
            threshold=threshold
        )

    def verify_normalized(
        self,
        anh_bgr,
        user_id,
        threshold=None,
        use_advanced=True
    ):
        if threshold is None:
            threshold = NGUONG_XAC_MINH

        return self.verification_service.xac_minh_normalized(
            anh_bgr,
            user_id,
            threshold=threshold,
            use_advanced=use_advanced
        )