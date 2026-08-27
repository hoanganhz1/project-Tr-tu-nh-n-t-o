# app/ui/registration_page.py
# ================================================================
# ĐĂNG KÝ KHUÔN MẶT - ĐỒNG BỘ 30FPS
# ================================================================

import cv2
import time

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFrame,
    QMessageBox,
    QProgressBar
)
from PyQt5.QtCore import Qt, QThreadPool, QTimer, QMutex, QMutexLocker
from PyQt5.QtGui import QImage, QPixmap

from app.utils.camera_manager import CameraManager
from app.utils.worker import TrichXuatWorker
from app.utils.logger import logger


class RegistrationPage(QWidget):
    """Trang đăng ký khuôn mặt - Đồng bộ 30FPS"""

    def __init__(self, face_api, embedder):
        super().__init__()

        self.face_api = face_api
        self.embedder = embedder
        self.detector = embedder.detector

        # Camera Manager
        self.camera_manager = CameraManager()
        self.active = False
        self.camera_manager.frame_ready.connect(self.cap_nhat_camera)

        # Buffer và mutex
        self.frame_buffer = None
        self.buffer_mutex = QMutex()
        self.display_timer = QTimer()
        self.display_timer.timeout.connect(self._update_display)
        self.display_timer.start(33)  # ~30fps

        # Trạng thái đăng ký
        self.dang_quay = False
        self.bo_dem_anh = 0
        self.danh_sach_embedding = []
        self.SO_MAU_CAN = 20

        # Flag xử lý
        self.dang_xu_ly_anh = False
        self.lan_cuoi_xu_ly = 0
        self.KHOANG_CACH_TOI_THIEU = 66  # ms (~15 lần/giây)

        # ThreadPool
        self.threadpool = QThreadPool.globalInstance()

        # Tạo giao diện
        self.tao_giao_dien()

        logger.info("[Registration] Đã khởi tạo (Đồng bộ 30FPS)")

    # ============================================================
    # GIAO DIỆN (GIỮ NGUYÊN)
    # ============================================================

    def tao_giao_dien(self):
        """Tạo giao diện"""
        bo_cuc = QVBoxLayout(self)
        bo_cuc.setContentsMargins(40, 35, 40, 35)

        bo_cuc.addWidget(QLabel("<h1>📸 ĐĂNG KÝ KHUÔN MẶT</h1>"))

        noi_dung = QHBoxLayout()

        # FORM THÔNG TIN
        khung_form = QFrame()
        khung_form.setStyleSheet("""
            QFrame {
                background: white;
                border: 1px solid #E2E8F0;
                border-radius: 16px;
            }
        """)

        form = QVBoxLayout(khung_form)
        form.setContentsMargins(30, 30, 30, 30)

        form.addWidget(QLabel("<b>Thông tin người dùng</b>"))
        form.addSpacing(10)

        form.addWidget(QLabel("Họ và tên *"))
        self.o_ten = QLineEdit()
        self.o_ten.setPlaceholderText("Nhập họ và tên")
        form.addWidget(self.o_ten)

        form.addWidget(QLabel("Tuổi"))
        self.o_tuoi = QLineEdit()
        self.o_tuoi.setPlaceholderText("Nhập tuổi")
        form.addWidget(self.o_tuoi)

        form.addWidget(QLabel("Quê quán"))
        self.o_que_quan = QLineEdit()
        self.o_que_quan.setPlaceholderText("Nhập quê quán")
        form.addWidget(self.o_que_quan)

        form.addWidget(QLabel("Lớp học"))
        self.o_lop = QLineEdit()
        self.o_lop.setPlaceholderText("Nhập lớp học")
        form.addWidget(self.o_lop)

        form.addWidget(QLabel("Ngành học"))
        self.o_nganh = QLineEdit()
        self.o_nganh.setPlaceholderText("Nhập ngành học")
        form.addWidget(self.o_nganh)

        form.addStretch()

        self.nut_dang_ky = QPushButton("📷 Bắt đầu thu thập")
        self.nut_dang_ky.clicked.connect(self.bat_dau_dang_ky)
        self.nut_dang_ky.setStyleSheet("""
            QPushButton {
                background-color: #2563EB;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1D4ED8; }
            QPushButton:disabled { background-color: #94A3B8; }
        """)
        form.addWidget(self.nut_dang_ky)

        self.nut_lam_moi = QPushButton("🧹 Làm mới")
        self.nut_lam_moi.clicked.connect(self.lam_moi)
        self.nut_lam_moi.setStyleSheet("""
            QPushButton {
                background-color: #64748B;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #475569; }
        """)
        form.addWidget(self.nut_lam_moi)

        noi_dung.addWidget(khung_form, 1)

        # CAMERA
        khung_camera = QFrame()
        khung_camera.setStyleSheet("""
            QFrame {
                background: white;
                border: 1px solid #E2E8F0;
                border-radius: 16px;
            }
        """)

        camera_layout = QVBoxLayout(khung_camera)

        self.hien_thi_camera = QLabel("📷 Camera")
        self.hien_thi_camera.setAlignment(Qt.AlignCenter)
        self.hien_thi_camera.setMinimumSize(500, 400)
        self.hien_thi_camera.setStyleSheet("""
            QLabel {
                background: #0F172A;
                color: white;
                border-radius: 12px;
                font-size: 20px;
            }
        """)
        camera_layout.addWidget(self.hien_thi_camera)

        self.nhan_thong_bao = QLabel("🔴 Chưa bắt đầu thu thập")
        self.nhan_thong_bao.setAlignment(Qt.AlignCenter)
        self.nhan_thong_bao.setStyleSheet("""
            QLabel {
                font-size: 14px;
                padding: 8px;
                background: #F1F5F9;
                border-radius: 8px;
                color: #475569;
            }
        """)
        camera_layout.addWidget(self.nhan_thong_bao)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, self.SO_MAU_CAN)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #2563EB;
                border-radius: 8px;
            }
        """)
        camera_layout.addWidget(self.progress_bar)

        noi_dung.addWidget(khung_camera, 2)

        bo_cuc.addLayout(noi_dung)

    # ============================================================
    # QUẢN LÝ TRANG ACTIVE
    # ============================================================

    def set_active(self, active: bool):
        self.active = active
        if active:
            if not self.camera_manager.is_opened():
                self.camera_manager.start(fps=30)
            self.camera_manager.resume()
            if not self.display_timer.isActive():
                self.display_timer.start(33)
            self.hien_thi_camera.setText("📷 Camera")
            logger.info("[Registration] Active: BẬT (30fps)")
        else:
            logger.info("[Registration] Active: TẮT")
            self.camera_manager.pause()
            self.hien_thi_camera.setText("⏸️ Camera tạm dừng")

    # ============================================================
    # XỬ LÝ FRAME TỪ CAMERA MANAGER
    # ============================================================

    def cap_nhat_camera(self, anh_bgr):
        """Nhận frame từ CameraManager - Lưu vào buffer"""
        if not self.active or anh_bgr is None:
            return
        
        with QMutexLocker(self.buffer_mutex):
            self.frame_buffer = anh_bgr.copy()

    # ============================================================
    # HIỂN THỊ (GỌI BỞI TIMER)
    # ============================================================

    def _update_display(self):
        """Cập nhật hiển thị từ buffer (30fps)"""
        if not self.active:
            return
        
        with QMutexLocker(self.buffer_mutex):
            if self.frame_buffer is None:
                return
            anh = self.frame_buffer.copy()
        
        # Vẽ khung
        box, _ = self.detector.phat_hien(anh)
        if box is not None:
            mau = (0, 255, 255) if self.dang_quay else (100, 200, 100)
            do_day = 2 if self.dang_quay else 1
            anh = self.detector.ve_khung(anh, box, mau, do_day)
        
        self._hien_thi_anh(anh)
        
        # Xử lý đăng ký (tần suất thấp)
        if self.dang_quay and not self.dang_xu_ly_anh:
            current_time = time.time() * 1000
            if current_time - self.lan_cuoi_xu_ly >= self.KHOANG_CACH_TOI_THIEU:
                self.xu_ly_anh_dang_ky()

    def _hien_thi_anh(self, anh_bgr):
        """Hiển thị ảnh lên QLabel"""
        if anh_bgr is None:
            return
        anh_rgb = cv2.cvtColor(anh_bgr, cv2.COLOR_BGR2RGB)
        cao, rong, kenh = anh_rgb.shape
        anh_qt = QImage(anh_rgb.data, rong, cao, kenh * rong, QImage.Format_RGB888)
        self.hien_thi_camera.setPixmap(
            QPixmap.fromImage(anh_qt).scaled(
                self.hien_thi_camera.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    # ============================================================
    # XỬ LÝ ĐĂNG KÝ (DÙNG WORKER)
    # ============================================================

    def xu_ly_anh_dang_ky(self):
        """Xử lý ảnh để trích xuất embedding"""
        with QMutexLocker(self.buffer_mutex):
            if self.frame_buffer is None:
                return
            anh = self.frame_buffer.copy()

        box, _ = self.detector.phat_hien(anh)
        if box is None:
            return

        self.dang_xu_ly_anh = True
        self.lan_cuoi_xu_ly = time.time() * 1000

        worker = TrichXuatWorker(self.embedder, anh, use_advanced=True)
        worker.signals.result.connect(self.nhan_embedding)
        worker.signals.error.connect(self.xu_ly_loi_trich_xuat)
        worker.signals.finished.connect(self.don_dep_luong)
        self.threadpool.start(worker)

    def nhan_embedding(self, embedding):
        self.dang_xu_ly_anh = False
        if not self.dang_quay or embedding is None:
            return

        self.danh_sach_embedding.append(embedding)
        self.bo_dem_anh = len(self.danh_sach_embedding)
        self.nhan_thong_bao.setText(f"📸 Đang thu thập: {self.bo_dem_anh}/{self.SO_MAU_CAN}")
        self.nhan_thong_bao.setStyleSheet("""
            QLabel {
                font-size: 14px;
                padding: 8px;
                background: #EFF6FF;
                border-radius: 8px;
                color: #2563EB;
            }
        """)
        self.progress_bar.setValue(self.bo_dem_anh)

        if self.bo_dem_anh >= self.SO_MAU_CAN:
            self.ket_thuc_thu_thap()

    def xu_ly_loi_trich_xuat(self, error):
        self.dang_xu_ly_anh = False
        logger.error(f"[Registration] Lỗi trích xuất: {error}")

    def don_dep_luong(self):
        pass

    # ============================================================
    # ĐĂNG KÝ
    # ============================================================

    def bat_dau_dang_ky(self):
        """Bắt đầu quá trình đăng ký"""
        if not self.o_ten.text().strip():
            QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng nhập họ và tên.")
            return

        if not self.camera_manager.is_opened():
            self.camera_manager.start()
            if not self.camera_manager.is_opened():
                QMessageBox.critical(self, "Lỗi camera", "Không thể mở camera.")
                return

        self.danh_sach_embedding = []
        self.bo_dem_anh = 0
        self.dang_quay = True
        self.dang_xu_ly_anh = False

        self.nut_dang_ky.setEnabled(False)
        self.nut_lam_moi.setEnabled(False)

        self.progress_bar.setValue(0)
        self.nhan_thong_bao.setText("📸 Đang thu thập: 0/20")
        self.nhan_thong_bao.setStyleSheet("""
            QLabel {
                font-size: 14px;
                padding: 8px;
                background: #EFF6FF;
                border-radius: 8px;
                color: #2563EB;
            }
        """)

        logger.info("[Registration] Bắt đầu đăng ký")

    def ket_thuc_thu_thap(self):
        """Kết thúc thu thập và lưu dữ liệu"""
        self.dang_quay = False
        self.dang_xu_ly_anh = False

        self.nhan_thong_bao.setText("⏳ Đang lưu dữ liệu...")
        self.nhan_thong_bao.setStyleSheet("""
            QLabel {
                font-size: 14px;
                padding: 8px;
                background: #FEF3C7;
                border-radius: 8px;
                color: #D97706;
            }
        """)

        if len(self.danh_sach_embedding) < 5:
            self.nhan_thong_bao.setText("❌ Không đủ embedding chất lượng")
            self.nhan_thong_bao.setStyleSheet("""
                QLabel {
                    font-size: 14px;
                    padding: 8px;
                    background: #FEF2F2;
                    border-radius: 8px;
                    color: #DC2626;
                }
            """)
            QMessageBox.warning(
                self,
                "Không đủ dữ liệu",
                f"Chỉ thu được {len(self.danh_sach_embedding)}/20 embedding chất lượng.\nVui lòng thử lại."
            )
            self.nut_dang_ky.setEnabled(True)
            self.nut_lam_moi.setEnabled(True)
            return

        thong_tin = {
            "name": self.o_ten.text().strip(),
            "age": self.o_tuoi.text().strip(),
            "home": self.o_que_quan.text().strip(),
            "class_name": self.o_lop.text().strip(),
            "major": self.o_nganh.text().strip()
        }

        try:
            ket_qua = self.face_api.register(thong_tin, self.danh_sach_embedding)

            self.nhan_thong_bao.setText("✅ Đăng ký thành công!")
            self.nhan_thong_bao.setStyleSheet("""
                QLabel {
                    font-size: 14px;
                    padding: 8px;
                    background: #F0FDF4;
                    border-radius: 8px;
                    color: #16A34A;
                }
            """)

            QMessageBox.information(
                self,
                "Thành công",
                f"✅ Đã đăng ký khuôn mặt cho {thong_tin['name']}\n"
                f"📸 Số ảnh chất lượng: {len(self.danh_sach_embedding)}"
            )

            logger.info(f"[Registration] Đăng ký thành công")
            self.lam_moi()

        except Exception as loi:
            logger.error(f"[LỖI ĐĂNG KÝ] {loi}")
            self.nhan_thong_bao.setText(f"❌ Lỗi: {str(loi)[:50]}...")
            self.nhan_thong_bao.setStyleSheet("""
                QLabel {
                    font-size: 14px;
                    padding: 8px;
                    background: #FEF2F2;
                    border-radius: 8px;
                    color: #DC2626;
                }
            """)
            QMessageBox.critical(self, "Lỗi đăng ký", str(loi))
            self.nut_dang_ky.setEnabled(True)
            self.nut_lam_moi.setEnabled(True)

    # ============================================================
    # LÀM MỚI
    # ============================================================

    def lam_moi(self):
        """Làm mới form"""
        self.dang_quay = False
        self.dang_xu_ly_anh = False
        self.danh_sach_embedding = []
        self.bo_dem_anh = 0

        self.o_ten.clear()
        self.o_tuoi.clear()
        self.o_que_quan.clear()
        self.o_lop.clear()
        self.o_nganh.clear()

        self.nut_dang_ky.setEnabled(True)
        self.nut_lam_moi.setEnabled(True)

        self.progress_bar.setValue(0)
        self.nhan_thong_bao.setText("🔴 Chưa bắt đầu thu thập")
        self.nhan_thong_bao.setStyleSheet("""
            QLabel {
                font-size: 14px;
                padding: 8px;
                background: #F1F5F9;
                border-radius: 8px;
                color: #475569;
            }
        """)

        logger.info("[Registration] Đã làm mới form")

    # ============================================================
    # ĐÓNG
    # ============================================================

    def closeEvent(self, su_kien):
        self.active = False
        su_kien.accept()