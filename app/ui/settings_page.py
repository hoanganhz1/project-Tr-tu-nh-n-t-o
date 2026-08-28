# app/ui/settings_page.py
# ================================================================
# CÀI ĐẶT - DÙNG LOAD/SAVE TỪ CONFIG (ĐÃ SỬA LỖI)
# ================================================================

import os
import json
import torch  # ✅ THÊM IMPORT NÀY

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QFrame,
    QPushButton,
    QMessageBox,
    QGroupBox
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


class SettingsPage(QWidget):
    """Trang cài đặt - TỰ ĐỘNG LOAD/SAVE"""

    threshold_changed = pyqtSignal(float, float)

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

        self.tao_giao_dien()

    def tao_giao_dien(self):
        bo_cuc = QVBoxLayout(self)
        bo_cuc.setContentsMargins(40, 35, 40, 35)

        bo_cuc.addWidget(QLabel("<h1>⚙️ CÀI ĐẶT THUẬT TOÁN</h1>"))

        # Threshold nhận dạng
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

        # Threshold xác minh
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

        # Nút lưu
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

        # Thông tin hệ thống - ✅ SỬA LỖI
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
        
        # ✅ SỬA LỖI: Kiểm tra CUDA_AVAILABLE
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