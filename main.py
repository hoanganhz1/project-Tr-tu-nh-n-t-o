# app/main.py
# ================================================================
# FACESECURE - ĐIỂM KHỞI CHẠY
# ================================================================

import sys
import os
import traceback

# Thêm path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt

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
# UTILS
# ================================================================

from app.utils.logger import logger


# ================================================================
# KHỞI TẠO HỆ THỐNG
# ================================================================

# Biến global để lưu recognizer
_recognizer = None
# app/main.py
# ================================================================
# FACESECURE - ĐIỂM KHỞI CHẠY
# ================================================================

# ... (imports khác)

# ================================================================
# KHỞI TẠO HỆ THỐNG
# ================================================================

def khoi_tao_he_thong():
    """Khởi tạo toàn bộ hệ thống"""
    global _recognizer

    logger.info("=" * 60)
    logger.info("FACESECURE - KHỞI TẠO HỆ THỐNG")
    logger.info("=" * 60)

    try:
        # 1. DATABASE
        logger.info("[1] Khởi tạo cơ sở dữ liệu...")
        
        def on_database_change():
            logger.info("[Database] Phát hiện thay đổi, refresh cache...")
            if _recognizer is not None:
                _recognizer.refresh_cache()
                logger.info("[Database] Cache đã được refresh")
        
        database = CoSoDuLieu(on_change_callback=on_database_change)
        logger.info(f"    Đã tải {len(database.lay_tat_ca_nguoi())} người dùng")

        # 2. MTCNN
        logger.info("[2] Khởi tạo MTCNN...")
        detector = PhatHienKhuonMat()
        logger.info("    OK")

        # 3. FACENET
        logger.info("[3] Khởi tạo FaceNet...")
        embedder = FaceEmbedder(detector)
        logger.info("    OK")

        # 4. MATCHER
        logger.info("[4] Khởi tạo Face Matcher...")
        matcher = FaceMatcher()
        logger.info("    OK")

        # 5. RECOGNIZER
        logger.info("[5] Khởi tạo Face Recognizer...")
        recognizer = FaceRecognizer(embedder, matcher, database)
        _recognizer = recognizer
        logger.info("    OK")

        # 6. SERVICES - Sử dụng FaceProcessor bên trong
        logger.info("[6] Khởi tạo Services...")
        dich_vu_dang_ky = DichVuDangKy(embedder, database)
        dich_vu_nhan_dang = DichVuNhanDang(recognizer)
        dich_vu_xac_minh = DichVuXacMinh(recognizer)
        logger.info("    Đăng ký      : OK")
        logger.info("    Nhận dạng 1:N : OK")
        logger.info("    Xác minh 1:1  : OK")

        # 7. FACE API
        logger.info("[7] Khởi tạo Face API...")
        face_api = FaceAPI(dich_vu_dang_ky, dich_vu_nhan_dang, dich_vu_xac_minh)
        logger.info("    OK")

        # 8. DATABASE API
        logger.info("[8] Khởi tạo Database API...")
        database_api = DatabaseAPI(database)
        logger.info("    OK")

        logger.info("=" * 60)
        logger.info("✅ KHỞI TẠO HỆ THỐNG THÀNH CÔNG")
        logger.info("=" * 60)

        return (face_api, database_api, detector, embedder, database)

    except Exception as loi:
        logger.error(f"❌ LỖI KHỞI TẠO: {loi}")
        traceback.print_exc()
        return None

# ================================================================
# MAIN
# ================================================================

def main():
    """Điểm chạy chính"""

    logger.info("🔄 Bắt đầu chương trình...")

    # Tạo ứng dụng PyQt
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    app.setStyleSheet("""
        QMainWindow {
            background-color: #F8FAFC;
        }
    """)

    try:
        he_thong = khoi_tao_he_thong()
        
        if he_thong is None:
            QMessageBox.critical(
                None,
                "Lỗi khởi tạo",
                "Không thể khởi tạo hệ thống.\nVui lòng kiểm tra log."
            )
            sys.exit(1)

        (face_api, database_api, detector, embedder, database) = he_thong

        logger.info("🔄 Đang khởi tạo giao diện...")
        cua_so = FaceSecureApp(
            face_api=face_api,
            database_api=database_api,
            detector=detector,
            embedder=embedder,
            database=database
        )

        cua_so.showMaximized()
        logger.info("✅ Giao diện đã hiển thị")

        sys.exit(app.exec_())

    except KeyboardInterrupt:
        logger.info("⚠️ Chương trình bị dừng bởi người dùng")
        sys.exit(0)
    except Exception as loi:
        logger.error(f"❌ LỖI CHƯƠNG TRÌNH: {loi}")
        traceback.print_exc()
        
        QMessageBox.critical(
            None,
            "Lỗi chương trình",
            f"Đã xảy ra lỗi:\n{loi}"
        )
        sys.exit(1)


# ================================================================
# ĐIỂM CHẠY
# ================================================================

if __name__ == "__main__":
    main()