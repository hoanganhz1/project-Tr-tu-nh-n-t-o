# app/ui/recognition_page.py
# ================================================================
# NHẬN DẠNG & XÁC MINH - THÊM THÔNG BÁO GIỌNG NÓI
# ================================================================

import cv2
import numpy as np
import time
import torch
import os

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QRadioButton,
    QButtonGroup,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
    QCheckBox,
    QSizePolicy,
    QSplitter,
    QComboBox
)
from PyQt5.QtCore import Qt, QThreadPool, QTimer, QMutex, QMutexLocker
from PyQt5.QtGui import QImage, QPixmap

from app.config import settings
from app.config.constants import DEFAULT_NHAN_DANG_THRESHOLD, DEFAULT_XAC_MINH_THRESHOLD
from app.config.settings import CUDA_AVAILABLE, DEFAULT_FPS
from app.utils.camera_manager import CameraManager
from app.utils.worker import NhanDangWorker, XacMinhWorker
from app.utils.tts_simple import speak  # ✅ THÊM IMPORT
from app.utils.logger import logger


class RecognitionPage(QWidget):
    """Trang nhận dạng - Có thông báo giọng nói"""

    def __init__(self, face_api):
        super().__init__()


        self.active_workers = []
        self.worker_mutex = QMutex()
        self.face_api = face_api

        # Lấy detector từ face_api
        self.detector = face_api.identification_service.recognizer.embedder.detector
        self.embedder = face_api.identification_service.recognizer.embedder

        # Lấy recognizer để refresh cache
        self.recognizer = face_api.identification_service.recognizer

        # Camera Manager
        self.camera_manager = CameraManager()
        self.active = False
        self.camera_manager.frame_ready.connect(self.cap_nhat_camera)

        # Buffer
        self.frame_buffer = None
        self.buffer_mutex = QMutex()

        # Cấu hình tốc độ
        self.has_gpu = CUDA_AVAILABLE
        self.FPS_CAMERA = DEFAULT_FPS
        self.DISPLAY_INTERVAL = 16 if self.has_gpu else 33
        self.min_process_interval = 500

        # Timer hiển thị
        self.display_timer = QTimer()
        self.display_timer.timeout.connect(self._update_display)
        self.display_timer.start(self.DISPLAY_INTERVAL)

        # Timer xử lý nhận dạng
        self.process_timer = QTimer()
        self.process_timer.timeout.connect(self._process_identification)
        self.process_timer.start(self.min_process_interval)

        # Trạng thái
        self.dang_xu_ly = False
        self.tu_dong_quet = True
        self.che_do_hien_tai = "1:N"
        self.cach_khop = "embedding"
        self.last_process_time = 0

        # Threshold
        self.threshold_nhan_dang = getattr(settings, 'NGUONG_NHAN_DANG', DEFAULT_NHAN_DANG_THRESHOLD)
        self.threshold_xac_minh = getattr(settings, 'NGUONG_XAC_MINH', DEFAULT_XAC_MINH_THRESHOLD)

        # ThreadPool
        self.threadpool = QThreadPool.globalInstance()
        self.active_workers = []
        self.worker_mutex = QMutex()

        # Kết quả hiện tại
        self.current_result = None
        
        # ✅ THÊM: Biến để tránh thông báo trùng lặp
        self.last_notification = ""
        self.last_notification_time = 0
        # ✅ THÊM: Cache kết quả nhận dạng để tránh xử lý lặp
        self._last_result_cache = {}
        self._cache_frame_hash = None
        # Load font Unicode
        self._load_unicode_font()

        # Tạo giao diện
        self.tao_giao_dien()

        logger.info(f"[Recognition] Đã khởi tạo (GPU={self.has_gpu}, FPS={self.FPS_CAMERA})")

    # ============================================================
    # SET ACTIVE - QUẢN LÝ TRANG
    # ============================================================
# Trong class RecognitionPage, thêm method này
    def _cancel_all_workers(self):
        """Hủy tất cả worker đang chạy"""
        with QMutexLocker(self.worker_mutex):
            for worker in self.active_workers:
                try:
                    if hasattr(worker, 'cancel'):
                        worker.cancel()
                except:
                    pass
            self.active_workers.clear()

    def _on_worker_finished(self, worker):
        """Dọn dẹp worker khi hoàn thành"""
        with QMutexLocker(self.worker_mutex):
            if worker in self.active_workers:
                self.active_workers.remove(worker)

    def toggle_tu_dong_quet(self, checked):
        """Bật/tắt tự động quét"""
        self.tu_dong_quet = checked
        if checked:
            self.nhan_trang_thai.setText("🔄 Đang tự động quét...")
            logger.info("[Recognition] Bật tự động quét")
        else:
            self.nhan_trang_thai.setText("⏸️ Tạm dừng quét")
            logger.info("[Recognition] Tắt tự động quét")

    def set_active(self, active: bool):
        """Bật/tắt trang"""
        self.active = active
        if active:
            if not self.camera_manager.is_opened():
                self.camera_manager.start(fps=self.FPS_CAMERA)
            self.camera_manager.resume()
            if not self.display_timer.isActive():
                self.display_timer.start(self.DISPLAY_INTERVAL)
            if not self.process_timer.isActive():
                self.process_timer.start(self.min_process_interval)
            self.hien_thi_camera.setText("📷 Camera")
            logger.info(f"[Recognition] Active: BẬT")
        else:
            logger.info("[Recognition] Active: TẮT")
            self._cancel_all_workers()
            self.camera_manager.pause()
            self.hien_thi_camera.setText("⏸️ Camera tạm dừng")
            """✅ THÊM: Phương thức để bật/tắt trang"""
            self.active = active
            if active:
                if not self.camera_manager.is_opened():
                    self.camera_manager.start(fps=self.FPS_CAMERA)
                self.camera_manager.resume()
                if not self.display_timer.isActive():
                    self.display_timer.start(self.DISPLAY_INTERVAL)
                if not self.process_timer.isActive():
                    self.process_timer.start(self.min_process_interval)
                self.hien_thi_camera.setText("📷 Camera")
                logger.info(f"[Recognition] Active: BẬT ({self.FPS_CAMERA}fps)")
            else:
                logger.info("[Recognition] Active: TẮT")
                self._cancel_all_workers()
                self.camera_manager.pause()
                self.hien_thi_camera.setText("⏸️ Camera tạm dừng")

    def _cancel_all_workers(self):
        """Hủy tất cả worker đang chạy"""
        with QMutexLocker(self.worker_mutex):
            for worker in self.active_workers:
                try:
                    if hasattr(worker, 'cancel'):
                        worker.cancel()
                except:
                    pass
            self.active_workers.clear()

    # ============================================================
    # LOAD FONT UNICODE
    # ============================================================

    def _load_unicode_font(self):
        """Tải font Unicode để hiển thị tiếng Việt"""
        self.font_path = None
        font_paths = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for path in font_paths:
            if os.path.exists(path):
                self.font_path = path
                logger.info(f"[Recognition] Đã tìm thấy font: {path}")
                return
        logger.warning("[Recognition] Không tìm thấy font Unicode")

    def _draw_unicode_text(self, img, text, x, y, color, font_size=16):
        """Vẽ text Unicode lên ảnh"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil, "RGBA")
            
            if self.font_path and os.path.exists(self.font_path):
                font = ImageFont.truetype(self.font_path, font_size)
            else:
                font = ImageFont.load_default()
            
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            padding = 8
            draw.rectangle(
                [x - padding, y - text_h - padding,
                 x + text_w + padding, y + padding],
                fill=(0, 0, 0, 200)
            )
            
            color_rgb = (color[2], color[1], color[0])
            draw.text((x, y - text_h), text, font=font, fill=color_rgb)
            
            img[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            return True
        except Exception as e:
            cv2.putText(img, text.encode('ascii', 'ignore').decode('ascii'),
                       (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            return False

    # ============================================================
    # GIAO DIỆN
    # ============================================================

    def tao_giao_dien(self):
        bo_cuc = QVBoxLayout(self)
        bo_cuc.setContentsMargins(20, 20, 20, 20)

        header = QHBoxLayout()

        header.addWidget(QLabel("<h2>🔍 NHẬN DẠNG & XÁC MINH</h2>"))

        self.check_tu_dong = QCheckBox("🔄 Tự động quét")
        self.check_tu_dong.setChecked(True)
        self.check_tu_dong.toggled.connect(self.toggle_tu_dong_quet)
        self.check_tu_dong.setStyleSheet("""
            QCheckBox {
                font-size: 13px;
                font-weight: bold;
                color: #2563EB;
            }
        """)
        header.addWidget(self.check_tu_dong)

        self.label_performance = QLabel("⚡ 0ms")
        self.label_performance.setStyleSheet("""
            QLabel {
                color: #64748B;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        header.addWidget(self.label_performance)

        self.nut_1_n = QRadioButton("1:N Nhận dạng")
        self.nut_1_1 = QRadioButton("1:1 Xác minh")
        self.nut_1_n.setChecked(True)

        self.nhom_che_do = QButtonGroup(self)
        self.nhom_che_do.addButton(self.nut_1_n)
        self.nhom_che_do.addButton(self.nut_1_1)

        header.addWidget(self.nut_1_n)
        header.addWidget(self.nut_1_1)

        self.o_id_xac_minh = QLineEdit()
        self.o_id_xac_minh.setPlaceholderText("Nhập ID")
        self.o_id_xac_minh.setFixedWidth(120)
        self.o_id_xac_minh.setEnabled(False)
        header.addWidget(self.o_id_xac_minh)

        header.addWidget(QLabel("Khớp:"))
        self.cbx_cach_khop = QComboBox()
        self.cbx_cach_khop.addItems(["📐 Vị trí", "🆔 ID", "🧠 Embedding"])
        self.cbx_cach_khop.setCurrentIndex(2)
        self.cbx_cach_khop.currentIndexChanged.connect(self.doi_cach_khop)
        self.cbx_cach_khop.setFixedWidth(150)
        self.cbx_cach_khop.setStyleSheet("""
            QComboBox {
                padding: 4px 8px;
                border: 1px solid #E2E8F0;
                border-radius: 6px;
                background: white;
            }
        """)
        header.addWidget(self.cbx_cach_khop)

        self.nut_bat_dau = QPushButton("🔍 Đối sánh")
        self.nut_bat_dau.clicked.connect(self.bat_dau_xu_ly_thu_cong)
        self.nut_bat_dau.setFixedSize(100, 30)
        self.nut_bat_dau.setStyleSheet("""
            QPushButton {
                background-color: #2563EB;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1D4ED8; }
            QPushButton:disabled { background-color: #94A3B8; }
        """)
        header.addWidget(self.nut_bat_dau)

        self.nut_lam_moi = QPushButton("🔄 Làm mới")
        self.nut_lam_moi.clicked.connect(self.lam_moi)
        self.nut_lam_moi.setFixedSize(100, 30)
        self.nut_lam_moi.setStyleSheet("""
            QPushButton {
                background-color: #64748B;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #475569; }
        """)
        header.addWidget(self.nut_lam_moi)

        header.addStretch()

        self.label_threshold = QLabel(
            f"📏 1N:{self.threshold_nhan_dang:.2f} | 11:{self.threshold_xac_minh:.2f}"
        )
        self.label_threshold.setStyleSheet("""
            QLabel {
                color: #64748B;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        header.addWidget(self.label_threshold)

        bo_cuc.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)

        khung_camera = QFrame()
        khung_camera.setStyleSheet("""
            QFrame {
                background: #0F172A;
                border-radius: 12px;
            }
        """)
        layout_camera = QVBoxLayout(khung_camera)
        layout_camera.setContentsMargins(5, 5, 5, 5)

        self.hien_thi_camera = QLabel("📷 Camera")
        self.hien_thi_camera.setAlignment(Qt.AlignCenter)
        self.hien_thi_camera.setMinimumHeight(450)
        self.hien_thi_camera.setStyleSheet("""
            QLabel {
                background: transparent;
                color: white;
                font-size: 24px;
            }
        """)
        self.hien_thi_camera.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout_camera.addWidget(self.hien_thi_camera)

        footer_camera = QHBoxLayout()
        self.nhan_trang_thai = QLabel("⏳ Chưa nhận diện")
        self.nhan_trang_thai.setStyleSheet("""
            QLabel {
                color: #94A3B8;
                font-size: 13px;
                padding: 5px 10px;
                background: rgba(255,255,255,0.05);
                border-radius: 6px;
            }
        """)
        footer_camera.addWidget(self.nhan_trang_thai)
        footer_camera.addStretch()
        self.nhan_so_face = QLabel("👤 0 khuôn mặt")
        self.nhan_so_face.setStyleSheet("""
            QLabel {
                color: #94A3B8;
                font-size: 13px;
            }
        """)
        footer_camera.addWidget(self.nhan_so_face)
        layout_camera.addLayout(footer_camera)

        splitter.addWidget(khung_camera)

        # Panel kết quả 1:1
        self.panel_ket_qua = QFrame()
        self.panel_ket_qua.setStyleSheet("""
            QFrame {
                background: white;
                border: 1px solid #E2E8F0;
                border-radius: 12px;
            }
        """)
        self.panel_ket_qua.setFixedWidth(350)
        self.panel_ket_qua.hide()

        layout_ket_qua = QVBoxLayout(self.panel_ket_qua)
        layout_ket_qua.setContentsMargins(20, 20, 20, 20)

        layout_ket_qua.addWidget(QLabel("<b>📊 KẾT QUẢ XÁC MINH</b>"))

        self.nhan_trang_thai_ket_qua = QLabel("⏳ Chưa xác minh")
        self.nhan_trang_thai_ket_qua.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                background: #F1F5F9;
                border-radius: 8px;
            }
        """)
        layout_ket_qua.addWidget(self.nhan_trang_thai_ket_qua)

        khung_nguoi = QFrame()
        khung_nguoi.setStyleSheet("""
            QFrame {
                background: #F8FAFC;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        layout_nguoi = QVBoxLayout(khung_nguoi)
        self.nhan_ten_ket_qua = QLabel("👤 Tên: ---")
        self.nhan_ten_ket_qua.setStyleSheet("font-size: 15px; font-weight: bold;")
        layout_nguoi.addWidget(self.nhan_ten_ket_qua)
        self.nhan_lop_ket_qua = QLabel("📚 Lớp: ---")
        layout_nguoi.addWidget(self.nhan_lop_ket_qua)
        self.nhan_nganh_ket_qua = QLabel("🎓 Ngành: ---")
        layout_nguoi.addWidget(self.nhan_nganh_ket_qua)
        self.nhan_id_ket_qua = QLabel("🆔 ID: ---")
        layout_nguoi.addWidget(self.nhan_id_ket_qua)
        layout_ket_qua.addWidget(khung_nguoi)

        khung_diem = QHBoxLayout()
        khung_distance = QFrame()
        khung_distance.setStyleSheet("""
            QFrame {
                background: #EFF6FF;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        layout_distance = QVBoxLayout(khung_distance)
        layout_distance.addWidget(QLabel("📏 Distance"))
        self.nhan_distance_ket_qua = QLabel("---")
        self.nhan_distance_ket_qua.setStyleSheet("font-size: 18px; font-weight: bold; color: #2563EB;")
        layout_distance.addWidget(self.nhan_distance_ket_qua)
        khung_diem.addWidget(khung_distance)

        khung_similarity = QFrame()
        khung_similarity.setStyleSheet("""
            QFrame {
                background: #F0FDF4;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        layout_similarity = QVBoxLayout(khung_similarity)
        layout_similarity.addWidget(QLabel("🎯 Similarity"))
        self.nhan_similarity_ket_qua = QLabel("---")
        self.nhan_similarity_ket_qua.setStyleSheet("font-size: 18px; font-weight: bold; color: #16A34A;")
        layout_similarity.addWidget(self.nhan_similarity_ket_qua)
        khung_diem.addWidget(khung_similarity)
        layout_ket_qua.addLayout(khung_diem)

        layout_ket_qua.addWidget(QLabel("<b>📋 So sánh với các user khác</b>"))
        self.bang_so_sanh_ket_qua = QTableWidget(0, 3)
        self.bang_so_sanh_ket_qua.setHorizontalHeaderLabels(["ID", "Họ tên", "Distance"])
        self.bang_so_sanh_ket_qua.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.bang_so_sanh_ket_qua.verticalHeader().setVisible(False)
        self.bang_so_sanh_ket_qua.setMaximumHeight(150)
        layout_ket_qua.addWidget(self.bang_so_sanh_ket_qua)

        layout_ket_qua.addStretch()
        splitter.addWidget(self.panel_ket_qua)
        splitter.setSizes([700, 350])
        bo_cuc.addWidget(splitter, 1)

        self.nut_1_1.toggled.connect(self.thay_doi_che_do)
        self.nut_1_n.toggled.connect(self.thay_doi_che_do)

    # ============================================================
    # NHẬN FRAME TỪ CAMERA
    # ============================================================

    def cap_nhat_camera(self, anh_bgr):
        if not self.active or anh_bgr is None:
            return

        with QMutexLocker(self.buffer_mutex):
            self.frame_buffer = anh_bgr.copy()

    # ============================================================
    # XỬ LÝ NHẬN DẠNG
    # ============================================================

    def _process_identification(self):
        if not self.active or not self.tu_dong_quet:
            return
        if self.che_do_hien_tai != "1:N":
            return
        if self.dang_xu_ly:
            return

        with QMutexLocker(self.buffer_mutex):
            if self.frame_buffer is None:
                return
            anh = self.frame_buffer.copy()

        if anh is None:
            return

        current_time = time.time() * 1000
        if current_time - self.last_process_time < self.min_process_interval:
            return
        self.last_process_time = current_time

        self.dang_xu_ly = True
        start_time = time.time()
        # ✅ THÊM: Kiểm tra nhanh xem có khuôn mặt không trước khi xử lý
        box, _ = self.detector.phat_hien_nhanh(anh, scale=0.3)
        if box is None:
            self.dang_xu_ly = False
            return

        worker = NhanDangWorker(
            face_api=self.face_api,
            anh_bgr=anh,
            threshold=self.threshold_nhan_dang
        )
        
        with QMutexLocker(self.worker_mutex):
            self.active_workers.append(worker)
        
        worker.signals.result.connect(
            lambda ket_qua: self._on_nhan_dang_result(ket_qua, start_time)
        )
        worker.signals.error.connect(self._on_nhan_dang_error)
        worker.signals.finished.connect(
            lambda: self._on_worker_finished(worker)
        )
        self.threadpool.start(worker)

    def _on_nhan_dang_result(self, ket_qua, start_time):
        self.dang_xu_ly = False

        process_time = (time.time() - start_time) * 1000
        self._update_performance_label(process_time)

        best_match = ket_qua.get("best_match")
        if best_match:
            user = best_match.get("user")
            dist = best_match.get("distance")
            
            if user:
                if hasattr(user, 'to_dict'):
                    user_dict = user.to_dict()
                else:
                    user_dict = user
                
                if dist is not None and dist <= self.threshold_nhan_dang:
                    name = user_dict.get("name", "Unknown")
                    self.current_result = {
                        "name": name,
                        "distance": dist,
                        "status": "success"
                    }
                    self.nhan_trang_thai.setText(f"{name}")
                    self.nhan_trang_thai.setStyleSheet("""
                        QLabel {
                            color: #16A34A;
                            font-size: 13px;
                            padding: 5px 10px;
                            background: rgba(22, 163, 74, 0.15);
                            border-radius: 6px;
                        }
                    """)
                    
                    # ✅ SỬA: Chỉ thông báo khi có sự thay đổi
                    notification = f"nhận diện thành công {name}"
                    self._speak_notification(notification)
                else:
                    self.current_result = {
                        "name": "Người lạ",
                        "distance": dist,
                        "status": "fail"
                    }
                    self.nhan_trang_thai.setText(f"❌ Người lạ - {dist:.3f}")
                    self.nhan_trang_thai.setStyleSheet("""
                        QLabel {
                            color: #DC2626;
                            font-size: 13px;
                            padding: 5px 10px;
                            background: rgba(220, 38, 38, 0.15);
                            border-radius: 6px;
                        }
                    """)
                    
                    # ✅ THÊM: Thông báo người lạ
                    self._speak_notification("Người lạ, không xác định")
                    
            else:
                self.current_result = {
                    "name": "Người lạ",
                    "distance": None,
                    "status": "fail"
                }
                self.nhan_trang_thai.setText("❌ Người lạ")
                self.nhan_trang_thai.setStyleSheet("""
                    QLabel {
                        color: #DC2626;
                        font-size: 13px;
                        padding: 5px 10px;
                        background: rgba(220, 38, 38, 0.15);
                        border-radius: 6px;
                    }
                """)
                self._speak_notification("Người lạ, không xác định")
        else:
            self.current_result = {
                "name": "Không xác định",
                "distance": None,
                "status": "unknown"
            }
            self.nhan_trang_thai.setText("ℹ️ Không nhận dạng được")
            self.nhan_trang_thai.setStyleSheet("""
                QLabel {
                    color: #94A3B8;
                    font-size: 13px;
                    padding: 5px 10px;
                    background: rgba(255,255,255,0.05);
                    border-radius: 6px;
                }
            """)

        try:
            self.recognizer._get_all_users()
        except Exception as e:
            logger.debug(f"[Recognition] Refresh cache: {e}")
    def _speak_notification(self, text):
        """
        Phát thông báo bằng giọng nói - CÓ GIỚI HẠN TẦN SUẤT
        """
        current_time = time.time()
        
        # ✅ SỬA: Tăng thời gian chờ lên 5 giây để tránh spam
        if text == self.last_notification and (current_time - self.last_notification_time) < 5:
            return
        
        self.last_notification = text
        self.last_notification_time = current_time
        
        # Phát âm thanh
        try:
            speak(text, voice="vi-VN")
            logger.info(f"[Recognition] Đã phát giọng nói: {text}")
        except Exception as e:
            logger.error(f"[Recognition] Lỗi phát giọng nói: {e}")

    def _on_nhan_dang_error(self, error):
        self.dang_xu_ly = False
        logger.error(f"[Recognition] Lỗi nhận dạng: {error}")

    def _on_worker_finished(self, worker):
        with QMutexLocker(self.worker_mutex):
            if worker in self.active_workers:
                self.active_workers.remove(worker)

    def _update_performance_label(self, process_time):
        """Cập nhật label hiệu suất"""
        self.label_performance.setText(f"⚡ {process_time:.0f}ms")
        if process_time < 100:  # ✅ SỬA: Ngưỡng 100ms
            self.label_performance.setStyleSheet("color: #16A34A; font-size: 12px; font-weight: bold;")
        elif process_time < 300:
            self.label_performance.setStyleSheet("color: #F59E0B; font-size: 12px; font-weight: bold;")
        else:
            self.label_performance.setStyleSheet("color: #DC2626; font-size: 12px; font-weight: bold;")

    # ============================================================
    # THÔNG BÁO GIỌNG NÓI
    # ============================================================

    def _speak_notification(self, text):
        """
        Phát thông báo bằng giọng nói
        Tránh thông báo trùng lặp trong 3 giây
        """
        current_time = time.time()
        
        # Tránh thông báo trùng lặp trong 3 giây
        if text == self.last_notification and (current_time - self.last_notification_time) < 3:
            return
        
        self.last_notification = text
        self.last_notification_time = current_time
        
        # Phát âm thanh
        try:
            speak(text, voice="vi-VN")
            logger.info(f"[Recognition] Đã phát giọng nói: {text}")
        except Exception as e:
            logger.error(f"[Recognition] Lỗi phát giọng nói: {e}")

    # ============================================================
    # HIỂN THỊ
    # ============================================================

    def _update_display(self):
        if not self.active:
            return

        with QMutexLocker(self.buffer_mutex):
            if self.frame_buffer is None:
                return
            anh = self.frame_buffer.copy()

        if anh is None:
            return

        # Phát hiện 1 khuôn mặt lớn nhất
        box, _ = self.detector.phat_hien_nhanh(anh, scale=0.5)
        
        if box is not None:
            self.nhan_so_face.setText("👤 1 khuôn mặt")
            
            if self.current_result:
                info = self.current_result
                status = info.get("status", "unknown")
                
                if status == "success":
                    color = (0, 255, 0)
                    name = info.get("name", "Unknown")
                    dist = info.get("distance")
                    label = f"✅ {name}" + (f" - {dist:.3f}" if dist else "")
                elif status == "fail":
                    color = (0, 0, 255)
                    name = info.get("name", "Người lạ")
                    dist = info.get("distance")
                    label = f"❌ {name}" + (f" - {dist:.3f}" if dist else "")
                else:
                    color = (0, 255, 255)
                    label = "🔄 Đang nhận dạng..."
                
                x1, y1, x2, y2 = box
                cv2.rectangle(anh, (x1, y1), (x2, y2), color, 2)
                self._draw_unicode_text(anh, label, x1, y1 - 10, color, font_size=16)
        else:
            self.nhan_so_face.setText("👤 0 khuôn mặt")
            self.current_result = None

        self._hien_thi_anh(anh)

    def _hien_thi_anh(self, anh_bgr):
        if anh_bgr is None:
            return
        try:
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
        except Exception as e:
            logger.error(f"[Recognition] Lỗi hiển thị: {e}")

    # ============================================================
    # XÁC MINH 1:1
    # ============================================================

    def bat_dau_xac_minh_thu_cong(self):
        if self.che_do_hien_tai != "1:1":
            return

        with QMutexLocker(self.buffer_mutex):
            if self.frame_buffer is None:
                QMessageBox.warning(self, "Chưa có ảnh", "Vui lòng đợi camera.")
                return
            anh = self.frame_buffer.copy()

        user_id = self.o_id_xac_minh.text().strip()
        if not user_id:
            QMessageBox.warning(self, "Thiếu ID", "Vui lòng nhập ID cần xác minh.")
            self.o_id_xac_minh.setFocus()
            return

        self.dang_xu_ly = True
        self.nut_bat_dau.setEnabled(False)
        self.nhan_trang_thai.setText(f"🔄 Đang xác minh ID {user_id}...")
        self.nhan_trang_thai_ket_qua.setText(f"🔄 Đang xác minh ID {user_id}...")
        self.nhan_trang_thai_ket_qua.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                background: #EFF6FF;
                border-radius: 8px;
                color: #2563EB;
            }
        """)

        worker = XacMinhWorker(
            face_api=self.face_api,
            anh_bgr=anh,
            user_id=user_id,
            threshold=self.threshold_xac_minh,
            use_normalize=True
        )
        worker.signals.result.connect(self._on_xac_minh_result)
        worker.signals.error.connect(self._on_xac_minh_error)
        worker.signals.finished.connect(self._on_xac_minh_finished)
        self.threadpool.start(worker)

    def _on_xac_minh_result(self, ket_qua):
        self.dang_xu_ly = False
        self.nut_bat_dau.setEnabled(True)

        thanh_cong = ket_qua.get("success", False)
        message = ket_qua.get("message", "")

        if thanh_cong:
            self.nhan_trang_thai_ket_qua.setText(f"✅ {message}")
            self.nhan_trang_thai_ket_qua.setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    font-weight: bold;
                    padding: 10px;
                    background: #F0FDF4;
                    border-radius: 8px;
                    color: #16A34A;
                }
            """)
            self.nhan_trang_thai.setText(f"✅ {message}")
            
            # ✅ THÊM: Thông báo xác minh thành công
            best_match = ket_qua.get("best_match", {})
            user_info = best_match.get("user") if best_match else None
            if user_info:
                if hasattr(user_info, 'to_dict'):
                    user_dict = user_info.to_dict()
                else:
                    user_dict = user_info
                name = user_dict.get("name", "Người dùng")
                self._speak_notification(f"Xác minh thành công,{name}")
            else:
                self._speak_notification("Xác minh thành công")
                
        else:
            self.nhan_trang_thai_ket_qua.setText(f"❌ {message}")
            self.nhan_trang_thai_ket_qua.setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    font-weight: bold;
                    padding: 10px;
                    background: #FEF2F2;
                    border-radius: 8px;
                    color: #DC2626;
                }
            """)
            self.nhan_trang_thai.setText(f"❌ {message}")
            
            # ✅ THÊM: Thông báo xác minh thất bại
            self._speak_notification("Xác minh thất bại")

        # Cập nhật chi tiết
        best_match = ket_qua.get("best_match", {})
        user_info = best_match.get("user") if best_match else None
        distance = best_match.get("distance") if best_match else None
        similarity = best_match.get("similarity") if best_match else None

        if user_info:
            if hasattr(user_info, 'to_dict'):
                user_dict = user_info.to_dict()
            else:
                user_dict = user_info
            self.nhan_ten_ket_qua.setText(f"👤 Tên: {user_dict.get('name', '---')}")
            self.nhan_lop_ket_qua.setText(f"📚 Lớp: {user_dict.get('class_name', '---')}")
            self.nhan_nganh_ket_qua.setText(f"🎓 Ngành: {user_dict.get('major', '---')}")
            self.nhan_id_ket_qua.setText(f"🆔 ID: {user_dict.get('id', '---')}")
            self.nhan_ten_ket_qua.setStyleSheet(
                "font-size: 15px; font-weight: bold; color: #16A34A;"
                if thanh_cong else "font-size: 15px; font-weight: bold; color: #DC2626;"
            )
        else:
            self.nhan_ten_ket_qua.setText("👤 Tên: Không xác định")
            self.nhan_lop_ket_qua.setText("📚 Lớp: ---")
            self.nhan_nganh_ket_qua.setText("🎓 Ngành: ---")
            self.nhan_id_ket_qua.setText("🆔 ID: ---")
            self.nhan_ten_ket_qua.setStyleSheet("font-size: 15px; font-weight: bold; color: #64748B;")

        if distance is not None:
            self.nhan_distance_ket_qua.setText(f"{distance:.4f}")
            self.nhan_distance_ket_qua.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: #16A34A;"
                if distance <= self.threshold_xac_minh else "font-size: 18px; font-weight: bold; color: #DC2626;"
            )
        else:
            self.nhan_distance_ket_qua.setText("---")
            self.nhan_distance_ket_qua.setStyleSheet("font-size: 18px; font-weight: bold; color: #94A3B8;")

        if similarity is not None:
            self.nhan_similarity_ket_qua.setText(f"{similarity:.2%}")
            self.nhan_similarity_ket_qua.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: #16A34A;"
                if similarity >= 0.5 else "font-size: 18px; font-weight: bold; color: #DC2626;"
            )
        else:
            self.nhan_similarity_ket_qua.setText("---")
            self.nhan_similarity_ket_qua.setStyleSheet("font-size: 18px; font-weight: bold; color: #94A3B8;")

        results = ket_qua.get("results", [])
        self.bang_so_sanh_ket_qua.setRowCount(0)
        for item in results:
            if isinstance(item, dict):
                user = item.get("user")
                dist = item.get("distance")
                if user and dist is not None:
                    if hasattr(user, 'to_dict'):
                        user_dict = user.to_dict()
                    else:
                        user_dict = user
                    row = self.bang_so_sanh_ket_qua.rowCount()
                    self.bang_so_sanh_ket_qua.insertRow(row)
                    self.bang_so_sanh_ket_qua.setItem(row, 0, QTableWidgetItem(str(user_dict.get("id", "-"))))
                    self.bang_so_sanh_ket_qua.setItem(row, 1, QTableWidgetItem(user_dict.get("name", "-")))
                    self.bang_so_sanh_ket_qua.setItem(row, 2, QTableWidgetItem(f"{dist:.4f}"))

    def _on_xac_minh_error(self, error):
        self.dang_xu_ly = False
        self.nut_bat_dau.setEnabled(True)
        logger.error(f"[Recognition] Lỗi xác minh: {error}")
        self.nhan_trang_thai.setText(f"❌ Lỗi: {error[:50]}")
        self.nhan_trang_thai_ket_qua.setText(f"❌ Lỗi: {error[:50]}")
        self.nhan_trang_thai_ket_qua.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                background: #FEF2F2;
                border-radius: 8px;
                color: #DC2626;
            }
        """)

    def _on_xac_minh_finished(self):
        self.nut_bat_dau.setEnabled(True)

    # ============================================================
    # CÁC HÀM HỖ TRỢ
    # ============================================================

    def thay_doi_che_do(self):
        is_1_1 = self.nut_1_1.isChecked()
        self.che_do_hien_tai = "1:1" if is_1_1 else "1:N"

        if is_1_1:
            self.current_result = None
            self.tu_dong_quet = False
            self.check_tu_dong.setChecked(False)
            self.panel_ket_qua.show()
            self.o_id_xac_minh.setEnabled(True)
            self.o_id_xac_minh.setFocus()
            self.nhan_trang_thai_ket_qua.setText("🔍 Nhập ID và bấm Đối sánh")
            self.nhan_trang_thai_ket_qua.setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    font-weight: bold;
                    padding: 10px;
                    background: #EFF6FF;
                    border-radius: 8px;
                    color: #2563EB;
                }
            """)
            self.nhan_trang_thai.setText("⏸️ Tạm dừng nhận dạng (đang xác minh)")
        else:
            if self.check_tu_dong.isChecked():
                self.tu_dong_quet = True
            self.panel_ket_qua.hide()
            self.o_id_xac_minh.clear()
            self.o_id_xac_minh.setEnabled(False)

        self.lam_moi()

    def toggle_tu_dong_quet(self, checked):
        self.tu_dong_quet = checked
        if checked:
            self.nhan_trang_thai.setText("🔄 Đang tự động quét...")
            logger.info("[Recognition] Bật tự động quét")
        else:
            self.nhan_trang_thai.setText("⏸️ Tạm dừng quét")
            logger.info("[Recognition] Tắt tự động quét")

    def doi_cach_khop(self, index):
        cac_cach = ["position", "id", "embedding"]
        self.cach_khop = cac_cach[index]
        logger.info(f"[Recognition] Đổi cách khớp sang: {self.cach_khop}")

    def bat_dau_xu_ly_thu_cong(self):
        if self.che_do_hien_tai == "1:N":
            with QMutexLocker(self.buffer_mutex):
                if self.frame_buffer is None:
                    QMessageBox.warning(self, "Chưa có ảnh", "Vui lòng đợi camera.")
                    return
                anh = self.frame_buffer.copy()
            if anh is not None:
                self.dang_xu_ly = True
                start_time = time.time()
                worker = NhanDangWorker(
                    face_api=self.face_api,
                    anh_bgr=anh,
                    threshold=self.threshold_nhan_dang
                )
                worker.signals.result.connect(
                    lambda ket_qua: self._on_nhan_dang_result(ket_qua, start_time)
                )
                worker.signals.error.connect(self._on_nhan_dang_error)
                self.threadpool.start(worker)
        else:
            self.bat_dau_xac_minh_thu_cong()

    def cap_nhat_threshold(self, nhan_dang, xac_minh):
        self.threshold_nhan_dang = nhan_dang
        self.threshold_xac_minh = xac_minh
        self.label_threshold.setText(f"📏 1N:{nhan_dang:.2f} | 11:{xac_minh:.2f}")

    def lam_moi(self):
        self.nhan_trang_thai.setText("⏳ Chưa nhận diện")
        self.nhan_trang_thai.setStyleSheet("""
            QLabel {
                color: #94A3B8;
                font-size: 13px;
                padding: 5px 10px;
                background: rgba(255,255,255,0.05);
                border-radius: 6px;
            }
        """)
        self.nhan_so_face.setText("👤 0 khuôn mặt")
        
        if self.che_do_hien_tai == "1:1":
            self.nhan_trang_thai_ket_qua.setText("⏳ Chưa xác minh")
            self.nhan_trang_thai_ket_qua.setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    font-weight: bold;
                    padding: 10px;
                    background: #F1F5F9;
                    border-radius: 8px;
                    color: #475569;
                }
            """)
            self.nhan_ten_ket_qua.setText("👤 Tên: ---")
            self.nhan_ten_ket_qua.setStyleSheet("font-size: 15px; font-weight: bold; color: #64748B;")
            self.nhan_lop_ket_qua.setText("📚 Lớp: ---")
            self.nhan_nganh_ket_qua.setText("🎓 Ngành: ---")
            self.nhan_id_ket_qua.setText("🆔 ID: ---")
            self.nhan_distance_ket_qua.setText("---")
            self.nhan_distance_ket_qua.setStyleSheet("font-size: 18px; font-weight: bold; color: #94A3B8;")
            self.nhan_similarity_ket_qua.setText("---")
            self.nhan_similarity_ket_qua.setStyleSheet("font-size: 18px; font-weight: bold; color: #94A3B8;")
            self.bang_so_sanh_ket_qua.setRowCount(0)
        
        self.label_performance.setText("⚡ 0ms")
        self.label_performance.setStyleSheet("color: #64748B; font-size: 12px; font-weight: bold;")
        self.current_result = None
        
        # Reset thông báo giọng nói
        self.last_notification = ""
        self.last_notification_time = 0

    def closeEvent(self, su_kien):
        self.active = False
        self._cancel_all_workers()
        su_kien.accept()