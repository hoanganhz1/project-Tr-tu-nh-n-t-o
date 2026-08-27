# app/ui/main_window.py
# ================================================================
# CỬA SỔ CHÍNH - QUẢN LÝ CAMERA 30FPS
# ================================================================

from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QStackedWidget,
    QFrame
)
from PyQt5.QtCore import Qt

from app.ui.registration_page import RegistrationPage
from app.ui.recognition_page import RecognitionPage
from app.ui.database_page import DatabasePage
from app.ui.settings_page import SettingsPage

from app.utils.camera_manager import CameraManager
from app.utils.logger import logger


class FaceSecureApp(QMainWindow):
    """Cửa sổ chính - Quản lý camera 30fps"""

    def __init__(self, face_api, database_api, detector, embedder, database):
        super().__init__()

        logger.info("[UI] Bắt đầu khởi tạo MainWindow...")

        self.face_api = face_api
        self.database_api = database_api
        self.detector = detector
        self.embedder = embedder
        self.database = database

        self.camera_manager = CameraManager()
        self.camera_manager.camera_error.connect(self.xu_ly_loi_camera)

        self.setWindowTitle("FaceSecure - Hệ thống nhận diện khuôn mặt")
        self.setMinimumSize(1200, 750)

        self.tao_giao_dien()

        logger.info("[UI] MainWindow khởi tạo thành công!")

    def xu_ly_loi_camera(self, error):
        logger.error(f"[Camera] {error}")

    def tao_giao_dien(self):
        logger.info("[UI] Đang tạo giao diện...")
        giao_dien_trung_tam = QWidget()
        self.setCentralWidget(giao_dien_trung_tam)

        bo_cuc_chinh = QHBoxLayout(giao_dien_trung_tam)
        bo_cuc_chinh.setContentsMargins(0, 0, 0, 0)
        bo_cuc_chinh.setSpacing(0)

        sidebar = self.tao_sidebar()
        bo_cuc_chinh.addWidget(sidebar)

        self.khu_vuc_noi_dung = QStackedWidget()
        bo_cuc_chinh.addWidget(self.khu_vuc_noi_dung)

        logger.info("[UI] Tạo trang đăng ký...")
        self.trang_dang_ky = RegistrationPage(
            face_api=self.face_api,
            embedder=self.embedder
        )

        logger.info("[UI] Tạo trang nhận dạng...")
        self.trang_nhan_dang = RecognitionPage(
            face_api=self.face_api
        )

        logger.info("[UI] Tạo trang quản lý...")
        self.trang_quan_ly = DatabasePage(
            database_api=self.database_api
        )

        logger.info("[UI] Tạo trang cài đặt...")
        self.trang_cai_dat = SettingsPage()
        self.trang_cai_dat.threshold_changed.connect(
            self.cap_nhat_threshold
        )

        self.khu_vuc_noi_dung.addWidget(self.trang_dang_ky)
        self.khu_vuc_noi_dung.addWidget(self.trang_nhan_dang)
        self.khu_vuc_noi_dung.addWidget(self.trang_quan_ly)
        self.khu_vuc_noi_dung.addWidget(self.trang_cai_dat)

        self.ket_noi_su_kien()
        self.chuyen_trang(0, self.nut_dang_ky)

        logger.info("[UI] Giao diện đã tạo xong")

    def tao_sidebar(self):
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(270)

        bo_cuc = QVBoxLayout(sidebar)
        bo_cuc.setContentsMargins(0, 35, 0, 30)

        logo = QLabel("FACESECURE")
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet("QLabel { color: #2563EB; font-size: 24px; font-weight: 900; }")
        bo_cuc.addWidget(logo)

        mo_ta = QLabel("Hệ thống nhận diện\nkhuôn mặt")
        mo_ta.setAlignment(Qt.AlignCenter)
        mo_ta.setStyleSheet("QLabel { color: #64748B; font-size: 13px; }")
        bo_cuc.addWidget(mo_ta)
        bo_cuc.addSpacing(40)

        self.nut_dang_ky = self.tao_nut_menu("👤  Đăng ký khuôn mặt")
        self.nut_nhan_dang = self.tao_nut_menu("🔍  Nhận dạng & Xác minh")
        self.nut_quan_ly = self.tao_nut_menu("🗄️  Quản lý dữ liệu")
        self.nut_cai_dat = self.tao_nut_menu("⚙️  Cài đặt")

        bo_cuc.addWidget(self.nut_dang_ky)
        bo_cuc.addWidget(self.nut_nhan_dang)
        bo_cuc.addWidget(self.nut_quan_ly)
        bo_cuc.addWidget(self.nut_cai_dat)
        bo_cuc.addStretch()

        thong_tin = QLabel("FaceSecure v1.0\nFaceNet + MTCNN\nEmbedding 512D")
        thong_tin.setAlignment(Qt.AlignCenter)
        thong_tin.setStyleSheet("QLabel { color: #94A3B8; font-size: 11px; }")
        bo_cuc.addWidget(thong_tin)

        sidebar.setStyleSheet("""
            #Sidebar {
                background-color: #FFFFFF;
                border-right: 1px solid #E2E8F0;
            }
            QPushButton {
                text-align: left;
                padding: 14px 20px;
                margin: 4px 15px;
                border: none;
                border-radius: 8px;
                color: #64748B;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #F1F5F9;
                color: #0F172A;
            }
            QPushButton:checked {
                background-color: #EFF6FF;
                color: #2563EB;
            }
        """)

        return sidebar

    def tao_nut_menu(self, noi_dung):
        nut = QPushButton(noi_dung)
        nut.setCheckable(True)
        nut.setCursor(Qt.PointingHandCursor)
        return nut

    def ket_noi_su_kien(self):
        self.nut_dang_ky.clicked.connect(lambda: self.chuyen_trang(0, self.nut_dang_ky))
        self.nut_nhan_dang.clicked.connect(lambda: self.chuyen_trang(1, self.nut_nhan_dang))
        self.nut_quan_ly.clicked.connect(lambda: self.chuyen_trang(2, self.nut_quan_ly))
        self.nut_cai_dat.clicked.connect(lambda: self.chuyen_trang(3, self.nut_cai_dat))

    def chuyen_trang(self, chi_so, nut_dang_chon):
        self.khu_vuc_noi_dung.setCurrentIndex(chi_so)

        danh_sach_nut = [
            self.nut_dang_ky,
            self.nut_nhan_dang,
            self.nut_quan_ly,
            self.nut_cai_dat
        ]
        for nut in danh_sach_nut:
            nut.setChecked(False)
        nut_dang_chon.setChecked(True)

        if chi_so == 0:
            self.trang_nhan_dang.set_active(False)
            self.trang_dang_ky.set_active(True)
        elif chi_so == 1:
            self.trang_dang_ky.set_active(False)
            self.trang_nhan_dang.set_active(True)
        else:
            self.trang_dang_ky.set_active(False)
            self.trang_nhan_dang.set_active(False)
            self.camera_manager.pause()
            logger.info("[UI] Camera paused")

        if chi_so == 2:
            self.trang_quan_ly.tai_du_lieu()

    def cap_nhat_threshold(self, nhan_dang, xac_minh):
        try:
            from app.config import settings
            settings.NGUONG_NHAN_DANG = nhan_dang
            settings.NGUONG_XAC_MINH = xac_minh
            settings.NGUONG_COSINE_DISTANCE = nhan_dang
            self.trang_nhan_dang.cap_nhat_threshold(nhan_dang, xac_minh)
            logger.info(f"[Settings] Đã cập nhật: 1N={nhan_dang}, 11={xac_minh}")
        except Exception as loi:
            logger.error(f"[Settings] Lỗi cập nhật: {loi}")

    def closeEvent(self, su_kien):
        logger.info("[UI] Đang đóng cửa sổ...")
        self.camera_manager.stop()
        self.trang_dang_ky.active = False
        self.trang_nhan_dang.active = False
        su_kien.accept()