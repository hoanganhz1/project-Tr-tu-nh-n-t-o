# app/ui/database_page.py
# ================================================================
# QUẢN LÝ CƠ SỞ DỮ LIỆU - HOÀN CHỈNH + REFRESH CACHE
# ================================================================

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QHeaderView,
    QMessageBox
)
from PyQt5.QtCore import Qt

from app.api.database_api import DatabaseAPI
from app.utils.logger import logger


class DatabasePage(QWidget):
    """Trang quản lý cơ sở dữ liệu"""

    def __init__(self, database_api: DatabaseAPI):
        super().__init__()

        self.database_api = database_api

        self.tao_giao_dien()
        
        logger.info("[Database] Đã khởi tạo")

    # ============================================================
    # GIAO DIỆN
    # ============================================================

    def tao_giao_dien(self):
        bo_cuc = QVBoxLayout(self)
        bo_cuc.setContentsMargins(40, 35, 40, 35)

        bo_cuc.addWidget(QLabel("<h1>🗄️ QUẢN LÝ CƠ SỞ DỮ LIỆU</h1>"))

        toolbar = QHBoxLayout()
        self.nut_lam_moi = QPushButton("🔄 Làm mới")
        self.nut_lam_moi.clicked.connect(self.tai_du_lieu)

        self.nut_xoa_tat_ca = QPushButton("🗑️ Xóa tất cả")
        self.nut_xoa_tat_ca.clicked.connect(self.xoa_tat_ca)

        toolbar.addWidget(self.nut_lam_moi)
        toolbar.addWidget(self.nut_xoa_tat_ca)
        toolbar.addStretch()

        self.nhan_so_luong = QLabel("Tổng: 0 người")
        toolbar.addWidget(self.nhan_so_luong)

        bo_cuc.addLayout(toolbar)

        self.bang_du_lieu = QTableWidget(0, 6)
        self.bang_du_lieu.setHorizontalHeaderLabels([
            "ID", "Họ tên", "Lớp", "Ngành", "Embedding", "Thao tác"
        ])
        self.bang_du_lieu.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.bang_du_lieu.verticalHeader().setVisible(False)

        bo_cuc.addWidget(self.bang_du_lieu)

        self.tai_du_lieu()

    # ============================================================
    # TẢI DỮ LIỆU
    # ============================================================

    def tai_du_lieu(self):
        try:
            danh_sach = self.database_api.get_all_users()

            self.bang_du_lieu.setRowCount(0)

            for nguoi in danh_sach:
                dong = self.bang_du_lieu.rowCount()
                self.bang_du_lieu.insertRow(dong)

                if hasattr(nguoi, 'to_dict'):
                    nguoi_dict = nguoi.to_dict()
                else:
                    nguoi_dict = nguoi

                self.bang_du_lieu.setItem(
                    dong, 0,
                    QTableWidgetItem(str(nguoi_dict.get("id", "")))
                )
                self.bang_du_lieu.setItem(
                    dong, 1,
                    QTableWidgetItem(nguoi_dict.get("name", ""))
                )
                self.bang_du_lieu.setItem(
                    dong, 2,
                    QTableWidgetItem(nguoi_dict.get("class_name", ""))
                )
                self.bang_du_lieu.setItem(
                    dong, 3,
                    QTableWidgetItem(nguoi_dict.get("major", ""))
                )

                dim = nguoi_dict.get("embedding_dimension", 512)
                self.bang_du_lieu.setItem(
                    dong, 4,
                    QTableWidgetItem(f"{dim}D")
                )

                nut_xoa = QPushButton("🗑️ Xóa")
                nut_xoa.clicked.connect(
                    lambda checked, uid=nguoi_dict.get("id"):
                    self.xoa_nguoi(uid)
                )
                self.bang_du_lieu.setCellWidget(dong, 5, nut_xoa)

            self.nhan_so_luong.setText(f"Tổng: {len(danh_sach)} người")
            logger.info(f"[Database] Đã tải {len(danh_sach)} người dùng")

        except Exception as loi:
            logger.error(f"[Database] Lỗi tải: {loi}")
            QMessageBox.critical(self, "Lỗi", f"Không thể tải dữ liệu:\n{loi}")

    # ============================================================
    # XÓA - CÓ REFRESH CACHE
    # ============================================================

    def xoa_nguoi(self, user_id):
        tra_loi = QMessageBox.question(
            self,
            "Xác nhận",
            f"Bạn có chắc muốn xóa ID {user_id}?",
            QMessageBox.Yes | QMessageBox.No
        )

        if tra_loi != QMessageBox.Yes:
            return

        try:
            self.database_api.delete_user(user_id)
            
            # ✅ Refresh cache trong recognizer
            self._refresh_recognizer_cache()
            
            self.tai_du_lieu()
            logger.info(f"[Database] Đã xóa ID {user_id}")

        except Exception as loi:
            logger.error(f"[Database] Lỗi xóa: {loi}")
            QMessageBox.critical(self, "Lỗi", f"Không thể xóa dữ liệu:\n{loi}")

    def xoa_tat_ca(self):
        tra_loi = QMessageBox.question(
            self,
            "Xác nhận",
            "Bạn có chắc muốn xóa TẤT CẢ dữ liệu?",
            QMessageBox.Yes | QMessageBox.No
        )

        if tra_loi != QMessageBox.Yes:
            return

        try:
            danh_sach = self.database_api.get_all_users()
            for nguoi in danh_sach:
                user_id = nguoi.id if hasattr(nguoi, 'id') else nguoi.get('id')
                self.database_api.delete_user(user_id)

            # ✅ Refresh cache trong recognizer
            self._refresh_recognizer_cache()
            
            self.tai_du_lieu()
            logger.info("[Database] Đã xóa tất cả")

        except Exception as loi:
            logger.error(f"[Database] Lỗi xóa tất cả: {loi}")
            QMessageBox.critical(self, "Lỗi", f"Không thể xóa dữ liệu:\n{loi}")

    def _refresh_recognizer_cache(self):
        """Refresh cache của recognizer"""
        try:
            # Gọi callback từ database
            if hasattr(self.database_api, 'database'):
                self.database_api.database._notify_change()
                logger.info("[Database] Đã gửi yêu cầu refresh cache")
        except Exception as e:
            logger.error(f"[Database] Lỗi refresh cache: {e}")