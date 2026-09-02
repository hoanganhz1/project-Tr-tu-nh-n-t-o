# app/ui/settings_page.py
# ================================================================
# CÀI ĐẶT - DÙNG LOAD/SAVE TỪ CONFIG + ĐỔI CAMERA
# ================================================================

import os
import json
import torch

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QFrame,
    QPushButton,
    QMessageBox,
    QGroupBox,
    QComboBox  # ✅ THÊM
)
from PyQt5.QtCore import Qt, pyqtSignal

from app.config import settings
from app.config.constants import (
    DEFAULT_NHAN_DANG_THRESHOLD,
    DEFAULT_XAC_MINH_THRESHOLD,
    MIN_COSINE_THRESHOLD,
    MAX_COSINE_THRESHOLD
)
from app.config.settings import CUDA_AVAILABLE, THIET_BI
from app.utils.logger import logger
from app.utils.camera_manager import CameraManager  # ✅ THÊM


class SettingsPage(QWidget):
    """Trang cài đặt - TỰ ĐỘNG LOAD/SAVE + ĐỔI CAMERA"""

    threshold_changed = pyqtSignal(float, float)
    camera_changed = pyqtSignal(int)  # ✅ THÊM

    def __init__(self):
        super().__init__()

        self.threshold_nhan_dang = getattr(
            settings,
            'NGUONG_NHAN_DANG',
            DEFAULT_NHAN_DANG_THRESHOLD
        )
        self.threshold_xac_minh = getattr(
            settings,
            'NGUONG_XAC_MINH',
            DEFAULT_XAC_MINH_THRESHOLD
        )

        # ✅ THÊM: Camera Manager
        self.camera_manager = CameraManager()
        self.available_cameras = []

        self.tao_giao_dien()
        
        # ✅ THÊM: Quét camera khi mở
        self.quet_camera()

    def tao_giao_dien(self):
        bo_cuc = QVBoxLayout(self)
        bo_cuc.setContentsMargins(40, 35, 40, 35)

        bo_cuc.addWidget(QLabel("<h1>⚙️ CÀI ĐẶT THUẬT TOÁN</h1>"))

        # ============================================================
        # THRESHOLD NHẬN DẠNG (GIỮ NGUYÊN)
        # ============================================================
        khung_nhan_dang = QFrame()
        khung_nhan_dang.setStyleSheet("""
            QFrame {
                background: white;
                border: 1px solid #E2E8F0;
                border-radius: 16px;
            }
        """)

        layout_nhan_dang = QVBoxLayout(khung_nhan_dang)
        layout_nhan_dang.setContentsMargins(30, 20, 30, 20)

        header_nhan_dang = QHBoxLayout()
        header_nhan_dang.addWidget(QLabel("<b>🔄 Nhận dạng 1:N</b>"))
        header_nhan_dang.addStretch()
        header_nhan_dang.addWidget(QLabel("Giá trị: "))
        self.label_nhan_dang = QLabel(f"{self.threshold_nhan_dang:.2f}")
        self.label_nhan_dang.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: bold;
                color: #2563EB;
                font-family: monospace;
            }
        """)
        header_nhan_dang.addWidget(self.label_nhan_dang)
        layout_nhan_dang.addLayout(header_nhan_dang)

        self.slider_nhan_dang = QSlider(Qt.Horizontal)
        self.slider_nhan_dang.setMinimum(int(MIN_COSINE_THRESHOLD * 100))
        self.slider_nhan_dang.setMaximum(int(MAX_COSINE_THRESHOLD * 100))
        self.slider_nhan_dang.setValue(int(self.threshold_nhan_dang * 100))
        self.slider_nhan_dang.setTickInterval(5)
        self.slider_nhan_dang.setTickPosition(QSlider.TicksBelow)
        self.slider_nhan_dang.valueChanged.connect(self.on_nhan_dang_changed)

        layout_nhan_dang.addWidget(self.slider_nhan_dang)

        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel(f"Chặt ({MIN_COSINE_THRESHOLD:.2f})"))
        range_layout.addStretch()
        range_layout.addWidget(QLabel(f"Nới ({MAX_COSINE_THRESHOLD:.2f})"))
        layout_nhan_dang.addLayout(range_layout)

        layout_nhan_dang.addWidget(QLabel("💡 Khuyến nghị: 0.30 - 0.40"))

        bo_cuc.addWidget(khung_nhan_dang)

        # ============================================================
        # THRESHOLD XÁC MINH (GIỮ NGUYÊN)
        # ============================================================
        khung_xac_minh = QFrame()
        khung_xac_minh.setStyleSheet("""
            QFrame {
                background: white;
                border: 1px solid #E2E8F0;
                border-radius: 16px;
            }
        """)

        layout_xac_minh = QVBoxLayout(khung_xac_minh)
        layout_xac_minh.setContentsMargins(30, 20, 30, 20)

        header_xac_minh = QHBoxLayout()
        header_xac_minh.addWidget(QLabel("<b>✅ Xác minh 1:1</b>"))
        header_xac_minh.addStretch()
        header_xac_minh.addWidget(QLabel("Giá trị: "))
        self.label_xac_minh = QLabel(f"{self.threshold_xac_minh:.2f}")
        self.label_xac_minh.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: bold;
                color: #16A34A;
                font-family: monospace;
            }
        """)
        header_xac_minh.addWidget(self.label_xac_minh)
        layout_xac_minh.addLayout(header_xac_minh)

        self.slider_xac_minh = QSlider(Qt.Horizontal)
        self.slider_xac_minh.setMinimum(int(MIN_COSINE_THRESHOLD * 100))
        self.slider_xac_minh.setMaximum(int(MAX_COSINE_THRESHOLD * 100))
        self.slider_xac_minh.setValue(int(self.threshold_xac_minh * 100))
        self.slider_xac_minh.setTickInterval(5)
        self.slider_xac_minh.setTickPosition(QSlider.TicksBelow)
        self.slider_xac_minh.valueChanged.connect(self.on_xac_minh_changed)

        layout_xac_minh.addWidget(self.slider_xac_minh)

        range_layout_2 = QHBoxLayout()
        range_layout_2.addWidget(QLabel(f"Chặt ({MIN_COSINE_THRESHOLD:.2f})"))
        range_layout_2.addStretch()
        range_layout_2.addWidget(QLabel(f"Nới ({MAX_COSINE_THRESHOLD:.2f})"))
        layout_xac_minh.addLayout(range_layout_2)

        layout_xac_minh.addWidget(QLabel("💡 Khuyến nghị: 0.25 - 0.35 (khắt khe hơn 1:N)"))

        bo_cuc.addWidget(khung_xac_minh)

        # ============================================================
        # ✅ THÊM: CÀI ĐẶT CAMERA (CHÈN VÀO GIỮA)
        # ============================================================
        khung_camera = QFrame()
        khung_camera.setStyleSheet("""
            QFrame {
                background: white;
                border: 1px solid #E2E8F0;
                border-radius: 16px;
            }
        """)

        layout_camera = QVBoxLayout(khung_camera)
        layout_camera.setContentsMargins(30, 20, 30, 20)

        header_camera = QHBoxLayout()
        header_camera.addWidget(QLabel("<b>📷 Chọn Camera</b>"))
        header_camera.addStretch()
        layout_camera.addLayout(header_camera)

        # Dòng chọn camera
        row_camera = QHBoxLayout()
        row_camera.addWidget(QLabel("Camera:"))

        self.cbx_camera = QComboBox()
        self.cbx_camera.setMinimumWidth(200)
        self.cbx_camera.setStyleSheet("""
            QComboBox {
                padding: 6px 12px;
                border: 1px solid #E2E8F0;
                border-radius: 6px;
                background: white;
            }
            QComboBox:hover {
                border-color: #94A3B8;
            }
        """)
        self.cbx_camera.currentIndexChanged.connect(self.doi_camera)
        row_camera.addWidget(self.cbx_camera)

        self.nut_quet = QPushButton("🔍 Quét")
        self.nut_quet.clicked.connect(self.quet_camera)
        self.nut_quet.setFixedWidth(80)
        self.nut_quet.setStyleSheet("""
            QPushButton {
                background-color: #64748B;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #475569;
            }
        """)
        row_camera.addWidget(self.nut_quet)

        row_camera.addStretch()
        layout_camera.addLayout(row_camera)

        # Trạng thái camera
        self.label_cam_status = QLabel("🔄 Đang kiểm tra camera...")
        self.label_cam_status.setStyleSheet("""
            QLabel {
                padding: 5px 10px;
                border-radius: 6px;
                font-weight: bold;
                color: #475569;
            }
        """)
        layout_camera.addWidget(self.label_cam_status)

        # ✅ CHÈN VÀO SAU THRESHOLD XÁC MINH, TRƯỚC NÚT LƯU
        bo_cuc.insertWidget(3, khung_camera)  # Index 3 = sau khung xác minh

        # ============================================================
        # NÚT LƯU (GIỮ NGUYÊN)
        # ============================================================
        khung_nut = QHBoxLayout()
        khung_nut.addStretch()

        self.nut_luu = QPushButton("💾 Lưu cài đặt")
        self.nut_luu.clicked.connect(self.save_settings)
        self.nut_luu.setFixedSize(200, 45)
        self.nut_luu.setStyleSheet("""
            QPushButton {
                background-color: #2563EB;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1D4ED8;
            }
        """)
        khung_nut.addWidget(self.nut_luu)

        khung_nut.addStretch()
        bo_cuc.addLayout(khung_nut)

        # ============================================================
        # THÔNG TIN HỆ THỐNG (GIỮ NGUYÊN)
        # ============================================================
        khung_thong_tin = QFrame()
        khung_thong_tin.setStyleSheet("""
            QFrame {
                background: #F8FAFC;
                border: 1px solid #E2E8F0;
                border-radius: 12px;
                padding: 15px;
            }
        """)

        layout_thong_tin = QVBoxLayout(khung_thong_tin)
        layout_thong_tin.addWidget(QLabel("<b>🔧 Thông tin hệ thống</b>"))

        model = getattr(settings, 'TEN_MODEL', 'vggface2')
        emb_dim = getattr(settings, 'CHIEU_EMBEDDING', 512)
        
        if CUDA_AVAILABLE:
            try:
                device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
                device = f"GPU - {device_name}"
            except:
                device = "GPU (CUDA)"
        else:
            device = "CPU"

        layout_thong_tin.addWidget(QLabel(f"• Model: {model} (InceptionResnetV1)"))
        layout_thong_tin.addWidget(QLabel(f"• Embedding: {emb_dim}D"))
        layout_thong_tin.addWidget(QLabel(f"• Device: {device}"))

        config_path = getattr(settings, 'TEP_THRESHOLD', 'data/threshold.json')
        layout_thong_tin.addWidget(QLabel(f"• Config: {config_path}"))

        bo_cuc.addWidget(khung_thong_tin)

        bo_cuc.addStretch()

    # ============================================================
    # HÀM XỬ LÝ THRESHOLD (GIỮ NGUYÊN)
    # ============================================================

    def on_nhan_dang_changed(self, value):
        self.threshold_nhan_dang = value / 100.0
        self.label_nhan_dang.setText(f"{self.threshold_nhan_dang:.2f}")

    def on_xac_minh_changed(self, value):
        self.threshold_xac_minh = value / 100.0
        self.label_xac_minh.setText(f"{self.threshold_xac_minh:.2f}")

    def save_settings(self):
        try:
            from app.config import settings
            settings.save_threshold(
                nhan_dang=self.threshold_nhan_dang,
                xac_minh=self.threshold_xac_minh
            )

            self.threshold_changed.emit(
                self.threshold_nhan_dang,
                self.threshold_xac_minh
            )

            logger.info(f"[Settings] Đã lưu: 1N={self.threshold_nhan_dang}, 11={self.threshold_xac_minh}")

            QMessageBox.information(
                self,
                "Thành công",
                f"✅ Đã lưu cài đặt:\n"
                f"• Nhận dạng 1:N: {self.threshold_nhan_dang:.2f}\n"
                f"• Xác minh 1:1: {self.threshold_xac_minh:.2f}"
            )

        except Exception as loi:
            logger.error(f"[Settings] Lỗi lưu: {loi}")
            QMessageBox.critical(self, "Lỗi", f"Không thể lưu cài đặt:\n{loi}")

    # ============================================================
    # ✅ THÊM: HÀM XỬ LÝ CAMERA
    # ============================================================

    def quet_camera(self):
        """Quét camera khả dụng"""
        self.label_cam_status.setText("🔄 Đang quét camera...")
        self.label_cam_status.setStyleSheet("""
            QLabel {
                padding: 5px 10px;
                border-radius: 6px;
                font-weight: bold;
                color: #D97706;
                background: #FEF3C7;
            }
        """)
        
        self.available_cameras = self.camera_manager.scan_cameras(max_cameras=10)
        
        self.cbx_camera.clear()
        if self.available_cameras:
            for cam_id in self.available_cameras:
                self.cbx_camera.addItem(f"Camera {cam_id}")
            
            # Chọn camera hiện tại
            current = self.camera_manager.camera_id
            if current in self.available_cameras:
                idx = self.available_cameras.index(current)
                self.cbx_camera.setCurrentIndex(idx)
            
            self.label_cam_status.setText(f"✅ Tìm thấy {len(self.available_cameras)} camera")
            self.label_cam_status.setStyleSheet("""
                QLabel {
                    padding: 5px 10px;
                    border-radius: 6px;
                    font-weight: bold;
                    color: #16A34A;
                    background: #F0FDF4;
                }
            """)
        else:
            self.cbx_camera.addItem("❌ Không tìm thấy camera")
            self.cbx_camera.setEnabled(False)
            self.label_cam_status.setText("❌ Không tìm thấy camera nào")
            self.label_cam_status.setStyleSheet("""
                QLabel {
                    padding: 5px 10px;
                    border-radius: 6px;
                    font-weight: bold;
                    color: #DC2626;
                    background: #FEF2F2;
                }
            """)

    def doi_camera(self, index):
        """Đổi sang camera khác"""
        if index < 0 or index >= len(self.available_cameras):
            return
        
        camera_id = self.available_cameras[index]
        
        # Chuyển camera
        if self.camera_manager.switch_to_camera(camera_id):
            self.label_cam_status.setText(f"✅ Đã chuyển sang Camera {camera_id}")
            self.label_cam_status.setStyleSheet("""
                QLabel {
                    padding: 5px 10px;
                    border-radius: 6px;
                    font-weight: bold;
                    color: #16A34A;
                    background: #F0FDF4;
                }
            """)
            self.camera_changed.emit(camera_id)
            logger.info(f"[Settings] Đã chuyển sang camera {camera_id}")
        else:
            self.label_cam_status.setText(f"❌ Không thể mở Camera {camera_id}")
            self.label_cam_status.setStyleSheet("""
                QLabel {
                    padding: 5px 10px;
                    border-radius: 6px;
                    font-weight: bold;
                    color: #DC2626;
                    background: #FEF2F2;
                }
            """)