# ================================================================
# app/main.py
# FACESECURE
# ĐIỂM KHỞI CHẠY CHƯƠNG TRÌNH
# ================================================================

import sys

from PyQt5.QtWidgets import QApplication


# ================================================================
# DATABASE
# ================================================================

from app.database.repository import CoSoDuLieu


# ================================================================
# CORE
# ================================================================

from app.core.detector import PhatHienKhuonMat
from app.core.embedder import FaceEmbedder
from app.core.matcher import FaceMatcher
from app.core.recognizer import FaceRecognizer


# ================================================================
# SERVICES
# ================================================================

from app.services.registration import DichVuDangKy
from app.services.identification import DichVuNhanDang
from app.services.verification import DichVuXacMinh


# ================================================================
# API
# ================================================================

from app.api.face_api import FaceAPI
from app.api.database_api import DatabaseAPI


# ================================================================
# UI
# ================================================================

from app.ui.main_window import FaceSecureApp


# ================================================================
# KHỞI TẠO HỆ THỐNG
# ================================================================

def khoi_tao_he_thong():

    print("=" * 60)
    print("FACESECURE")
    print("DANG KHOI TAO HE THONG...")
    print("=" * 60)

    # ------------------------------------------------------------
    # 1. DATABASE
    # ------------------------------------------------------------

    print("[1] Khoi tao co so du lieu...")

    database = CoSoDuLieu()

    print("    OK")


    # ------------------------------------------------------------
    # 2. MTCNN - PHÁT HIỆN KHUÔN MẶT
    # ------------------------------------------------------------

    print("[2] Khoi tao MTCNN...")

    detector = PhatHienKhuonMat()

    print("    OK")


    # ------------------------------------------------------------
    # 3. FACENET - TRÍCH XUẤT EMBEDDING
    # ------------------------------------------------------------

    print("[3] Khoi tao FaceNet...")

    embedder = FaceEmbedder(
        detector
    )

    print("    OK")


    # ------------------------------------------------------------
    # 4. COSINE MATCHER
    # ------------------------------------------------------------

    print("[4] Khoi tao Face Matcher...")

    matcher = FaceMatcher()

    print("    OK")


    # ------------------------------------------------------------
    # 5. RECOGNIZER
    # ------------------------------------------------------------

    print("[5] Khoi tao Face Recognizer...")

    recognizer = FaceRecognizer(
        embedder,
        matcher,
        database
    )

    print("    OK")


    # ------------------------------------------------------------
    # 6. SERVICES
    # ------------------------------------------------------------

    print("[6] Khoi tao Services...")

    dich_vu_dang_ky = DichVuDangKy(
        embedder,
        database
    )

    dich_vu_nhan_dang = DichVuNhanDang(
        recognizer
    )

    dich_vu_xac_minh = DichVuXacMinh(
        recognizer
    )

    print("    Dang ky       : OK")
    print("    Nhan dang 1:N : OK")
    print("    Xac minh 1:1  : OK")


    # ------------------------------------------------------------
    # 7. FACE API
    # ------------------------------------------------------------

    print("[7] Khoi tao Face API...")

    face_api = FaceAPI(
        dich_vu_dang_ky,
        dich_vu_nhan_dang,
        dich_vu_xac_minh
    )

    print("    OK")


    # ------------------------------------------------------------
    # 8. DATABASE API
    # ------------------------------------------------------------

    print("[8] Khoi tao Database API...")

    database_api = DatabaseAPI(
        database
    )

    print("    OK")


    print("=" * 60)
    print("KHOI TAO HE THONG THANH CONG")
    print("=" * 60)


    return (
        face_api,
        database_api,
        detector,
        embedder,
        database
    )


# ================================================================
# MAIN
# ================================================================

def main():

    # ------------------------------------------------------------
    # Tạo ứng dụng PyQt
    # ------------------------------------------------------------

    ung_dung = QApplication(
        sys.argv
    )


    # ------------------------------------------------------------
    # Khởi tạo toàn bộ hệ thống
    # ------------------------------------------------------------

    (
        face_api,
        database_api,
        detector,
        embedder,
        database
    ) = khoi_tao_he_thong()


    # ------------------------------------------------------------
    # Khởi tạo giao diện chính
    # ------------------------------------------------------------

    cua_so = FaceSecureApp(
        face_api=face_api,
        database_api=database_api,
        detector=detector,
        embedder=embedder,
        database=database
    )


    # ------------------------------------------------------------
    # Hiển thị giao diện
    # ------------------------------------------------------------

    cua_so.showMaximized()


    # ------------------------------------------------------------
    # Chạy chương trình
    # ------------------------------------------------------------

    sys.exit(
        ung_dung.exec_()
    )


# ================================================================
# ĐIỂM CHẠY
# ================================================================

if __name__ == "__main__":
    main()