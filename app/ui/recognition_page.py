# app/ui/recognition_page.py
# ================================================================
# NHẬN DẠNG & XÁC MINH - ĐỒNG BỘ 30FPS, KHÔNG XUNG ĐỘT
# ================================================================

import cv2
import numpy as np
import time

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
from app.utils.camera_manager import CameraManager
from app.utils.worker import NhanDangWorker, XacMinhWorker
from app.utils.logger import logger


class RecognitionPage(QWidget):
    """Trang nhận dạng và xác minh - Đồng bộ 30FPS, không xung đột"""

    def __init__(self, face_api):
        super().__init__()

        self.face_api = face_api

        # Lấy detector và embedder từ face_api (dùng chung)
        self.detector = face_api.identification_service.recognizer.embedder.detector
        self.embedder = face_api.identification_service.recognizer.embedder
        self.matcher = face_api.identification_service.recognizer.matcher

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

        # Trạng thái
        self.anh_hien_tai = None
        self.dang_xu_ly = False
        self.tu_dong_quet = True
        self.che_do_hien_tai = "1:N"   # "1:N" hoặc "1:1"
        self.cach_khop = "embedding"

        # Tăng tốc: xử lý mỗi N frame
        self.frame_counter = 0
        self.PROCESS_EVERY_N_FRAMES = 2
        self.last_process_time = 0
        self.min_process_interval = 66  # ~15 lần/giây

        # Threshold
        self.threshold_nhan_dang = getattr(settings, 'NGUONG_NHAN_DANG', DEFAULT_NHAN_DANG_THRESHOLD)
        self.threshold_xac_minh = getattr(settings, 'NGUONG_XAC_MINH', DEFAULT_XAC_MINH_THRESHOLD)

        # ThreadPool
        self.threadpool = QThreadPool.globalInstance()

        # Lưu kết quả cuối cùng để hiển thị
        self.last_results = []
        self.last_boxes = []
        self.last_thong_tin = []

        # Tạo giao diện
        self.tao_giao_dien()

        logger.info("[Recognition] Đã khởi tạo (Đồng bộ 30FPS, không xung đột)")

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

        # Splitter
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
            logger.info("[Recognition] Active: BẬT (30fps)")
        else:
            logger.info("[Recognition] Active: TẮT")
            self.camera_manager.pause()
            self.hien_thi_camera.setText("⏸️ Camera tạm dừng")

    # ============================================================
    # XỬ LÝ FRAME TỪ CAMERA MANAGER
    # ============================================================

    def cap_nhat_camera(self, anh_bgr):
        if not self.active or anh_bgr is None:
            return

        with QMutexLocker(self.buffer_mutex):
            self.frame_buffer = anh_bgr.copy()

        # Chỉ xử lý nhận dạng tự động nếu đang ở chế độ 1:N và bật tự động quét
        if self.tu_dong_quet and self.che_do_hien_tai == "1:N" and not self.dang_xu_ly:
            self.frame_counter += 1
            if self.frame_counter % self.PROCESS_EVERY_N_FRAMES == 0:
                current_time = time.time() * 1000
                if current_time - self.last_process_time >= self.min_process_interval:
                    self.last_process_time = current_time
                    with QMutexLocker(self.buffer_mutex):
                        anh_xu_ly = self.frame_buffer.copy() if self.frame_buffer is not None else None
                    if anh_xu_ly is not None:
                        self.bat_dau_nhan_dang_tu_dong(anh_xu_ly)

    # ============================================================
    # HIỂN THỊ (GỌI BỞI TIMER)
    # ============================================================

    def _update_display(self):
        if not self.active:
            return

        with QMutexLocker(self.buffer_mutex):
            if self.frame_buffer is None:
                return
            anh = self.frame_buffer.copy()

        # Nếu đang ở chế độ 1:1, chỉ hiển thị khung cơ bản (không vẽ kết quả nhận dạng)
        if self.che_do_hien_tai == "1:1":
            # Vẽ khung MTCNN đơn giản
            box, _ = self.detector.phat_hien(anh)
            if box is not None:
                cv2.rectangle(anh, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                self.nhan_so_face.setText("👤 1 khuôn mặt")
            else:
                self.nhan_so_face.setText("👤 0 khuôn mặt")
        else:
            # Chế độ 1:N: vẽ kết quả nhận dạng nếu có
            if self.last_results and self.last_boxes:
                anh = self.detector.ve_khung_cho_nhieu_face(anh, self.last_boxes, self.last_thong_tin)
            else:
                # Nếu chưa có kết quả, vẽ khung MTCNN cơ bản
                box, _ = self.detector.phat_hien(anh)
                if box is not None:
                    cv2.rectangle(anh, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    self.nhan_so_face.setText("👤 1 khuôn mặt")
                else:
                    self.nhan_so_face.setText("👤 0 khuôn mặt")

        self._hien_thi_anh(anh)

    def _hien_thi_anh(self, anh_bgr):
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
    # NHẬN DẠNG TỰ ĐỘNG (DÙNG WORKER)
    # ============================================================

    def bat_dau_nhan_dang_tu_dong(self, anh_bgr):
        if anh_bgr is None or self.dang_xu_ly or self.che_do_hien_tai != "1:N":
            return

        self.dang_xu_ly = True
        worker = NhanDangWorker(
            face_api=self.face_api,
            anh_bgr=anh_bgr,
            threshold=self.threshold_nhan_dang
        )
        worker.signals.result.connect(self.nhan_ket_qua_nhan_dang)
        worker.signals.error.connect(self.xu_ly_loi_nhan_dang)
        worker.signals.finished.connect(self.ket_thuc_nhan_dang)
        self.threadpool.start(worker)

    def nhan_ket_qua_nhan_dang(self, ket_qua):
        self.dang_xu_ly = False

        process_time = ket_qua.get("processing_time_ms", 0)
        if process_time > 0:
            self.label_performance.setText(f"⚡ {process_time:.0f}ms")
            if process_time < 50:
                self.label_performance.setStyleSheet("color: #16A34A; font-size: 12px; font-weight: bold;")
            elif process_time < 150:
                self.label_performance.setStyleSheet("color: #F59E0B; font-size: 12px; font-weight: bold;")
            else:
                self.label_performance.setStyleSheet("color: #DC2626; font-size: 12px; font-weight: bold;")

        if ket_qua.get("success", False):
            self.nhan_trang_thai.setText(f"✅ {ket_qua.get('message', 'Thành công')}")
            self.nhan_trang_thai.setStyleSheet("""
                QLabel {
                    color: #16A34A;
                    font-size: 13px;
                    padding: 5px 10px;
                    background: rgba(22, 163, 74, 0.15);
                    border-radius: 6px;
                }
            """)
        else:
            self.nhan_trang_thai.setText(f"❌ {ket_qua.get('message', 'Thất bại')}")
            self.nhan_trang_thai.setStyleSheet("""
                QLabel {
                    color: #DC2626;
                    font-size: 13px;
                    padding: 5px 10px;
                    background: rgba(220, 38, 38, 0.15);
                    border-radius: 6px;
                }
            """)

        # Lưu kết quả để hiển thị (chỉ khi ở chế độ 1:N)
        if self.che_do_hien_tai == "1:N":
            self.last_results = ket_qua.get("results", [])
            with QMutexLocker(self.buffer_mutex):
                if self.frame_buffer is not None:
                    anh = self.frame_buffer.copy()
                else:
                    anh = None
            if anh is not None:
                self.last_boxes, _ = self.detector.phat_hien_tat_ca(anh)
                self.last_thong_tin = self.khop_nhieu_face(anh, self.last_boxes, self.last_results)

    def xu_ly_loi_nhan_dang(self, error):
        self.dang_xu_ly = False
        logger.error(f"[Recognition] Lỗi nhận dạng: {error}")

    def ket_thuc_nhan_dang(self):
        pass

    # ============================================================
    # KHỚP NHIỀU KHUÔN MẶT
    # ============================================================

    def khop_nhieu_face(self, anh_bgr, boxes, results):
        if not boxes or not results:
            return []

        # Xây dựng map user từ results
        user_map = {}
        for item in results:
            if isinstance(item, dict):
                user = item.get("user")
                dist = item.get("distance")
                if user:
                    if hasattr(user, 'to_dict'):
                        user_dict = user.to_dict()
                    else:
                        user_dict = user
                    user_id = user_dict.get("id")
                    user_map[user_id] = {
                        "name": user_dict.get("name", "Unknown"),
                        "distance": dist,
                        "status": "success" if dist is not None and dist <= self.threshold_nhan_dang else "fail",
                        "user": user
                    }

        # Cắt các ROI
        list_roi = []
        for box in boxes:
            x1, y1, x2, y2 = box
            face_roi = anh_bgr[y1:y2, x1:x2]
            if face_roi.size == 0:
                list_roi.append(None)
            else:
                list_roi.append(face_roi)

        # Lọc None
        valid_indices = [i for i, roi in enumerate(list_roi) if roi is not None]
        valid_rois = [list_roi[i] for i in valid_indices]

        if not valid_rois:
            return [{"name": "Không xác định", "distance": None, "status": "fail"} for _ in boxes]

        # Batch extract embeddings
        embeddings = [None] * len(list_roi)
        batch_embs = self.embedder.trich_xuat_batch(valid_rois, use_advanced=True)
        for idx, emb in zip(valid_indices, batch_embs):
            embeddings[idx] = emb

        # Xử lý từng face
        thong_tin_list = []
        from app.database.repository import CoSoDuLieu

        for i, (box, embedding) in enumerate(zip(boxes, embeddings)):
            info = {"name": "Không xác định", "distance": None, "status": "fail"}

            if embedding is None:
                thong_tin_list.append(info)
                continue

            if self.cach_khop == "position":
                if i < len(results):
                    item = results[i]
                    if isinstance(item, dict):
                        user = item.get("user")
                        dist = item.get("distance")
                        if user:
                            if hasattr(user, 'to_dict'):
                                user_dict = user.to_dict()
                            else:
                                user_dict = user
                            info["name"] = user_dict.get("name", "Unknown")
                            info["distance"] = dist
                            info["status"] = "success" if dist is not None and dist <= self.threshold_nhan_dang else "fail"

            elif self.cach_khop == "id":
                best_match = None
                best_distance = float('inf')
                for item in results:
                    if not isinstance(item, dict):
                        continue
                    user = item.get("user")
                    dist = item.get("distance")
                    if user is None or dist is None:
                        continue
                    if hasattr(user, 'embedding'):
                        user_embedding = user.embedding
                    elif isinstance(user, dict) and 'embedding' in user:
                        user_embedding = user['embedding']
                    else:
                        continue
                    actual_distance = self.matcher.tinh_cosine_distance(embedding, user_embedding)
                    if actual_distance < best_distance:
                        best_distance = actual_distance
                        best_match = item
                if best_match is not None:
                    user = best_match.get("user")
                    if user:
                        if hasattr(user, 'to_dict'):
                            user_dict = user.to_dict()
                        else:
                            user_dict = user
                        info["name"] = user_dict.get("name", "Unknown")
                        info["distance"] = best_distance
                        info["status"] = "success" if best_distance <= self.threshold_nhan_dang else "fail"

            else:  # embedding
                db = CoSoDuLieu()
                all_users = db.lay_tat_ca_nguoi()
                best_match = None
                best_distance = float('inf')
                for user in all_users:
                    if hasattr(user, 'embedding'):
                        user_embedding = user.embedding
                    elif isinstance(user, dict) and 'embedding' in user:
                        user_embedding = user['embedding']
                    else:
                        continue
                    actual_distance = self.matcher.tinh_cosine_distance(embedding, user_embedding)
                    if actual_distance < best_distance:
                        best_distance = actual_distance
                        best_match = user
                if best_match is not None:
                    if hasattr(best_match, 'to_dict'):
                        user_dict = best_match.to_dict()
                    else:
                        user_dict = best_match
                    info["name"] = user_dict.get("name", "Unknown")
                    info["distance"] = best_distance
                    info["status"] = "success" if best_distance <= self.threshold_nhan_dang else "fail"

            thong_tin_list.append(info)

        return thong_tin_list

    # ============================================================
    # XÁC MINH 1:1 (DÙNG THREADPOOL)
    # ============================================================

    def bat_dau_xac_minh_thu_cong(self):
        """Bắt đầu xác minh 1:1 thủ công - Tạm dừng nhận dạng tự động"""
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

        # Ngăn không cho nhận dạng tự động chạy trong khi xác minh
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
        worker.signals.result.connect(self.nhan_ket_qua_xac_minh)
        worker.signals.error.connect(self.xu_ly_loi_xac_minh)
        worker.signals.finished.connect(self.ket_thuc_xac_minh)
        self.threadpool.start(worker)

    def nhan_ket_qua_xac_minh(self, ket_qua):
        self.dang_xu_ly = False
        self.nut_bat_dau.setEnabled(True)

        # Cập nhật hiển thị
        with QMutexLocker(self.buffer_mutex):
            if self.frame_buffer is not None:
                anh = self.frame_buffer.copy()
            else:
                anh = None

        if anh is not None:
            box, _ = self.detector.phat_hien(anh)
            if box is not None:
                best_match = ket_qua.get("best_match", {})
                user_info = best_match.get("user") if best_match else None
                distance = best_match.get("distance") if best_match else None
                ten = user_info.to_dict().get("name") if user_info and hasattr(user_info, 'to_dict') else None
                anh = self.detector.ve_khung_va_thong_tin(
                    anh, box,
                    ten_nguoi=ten,
                    distance=distance,
                    threshold=self.threshold_xac_minh,
                    trang_thai="success" if ket_qua.get("success") else "fail"
                )
            self._hien_thi_anh(anh)

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

    def xu_ly_loi_xac_minh(self, error):
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

    def ket_thuc_xac_minh(self):
        self.nut_bat_dau.setEnabled(True)

    # ============================================================
    # CÁC HÀM HỖ TRỢ KHÁC
    # ============================================================

    def thay_doi_che_do(self):
        is_1_1 = self.nut_1_1.isChecked()
        self.che_do_hien_tai = "1:1" if is_1_1 else "1:N"

        # Khi chuyển sang 1:1, xóa kết quả nhận dạng cũ để tránh hiển thị nhầm
        if is_1_1:
            self.last_results = []
            self.last_boxes = []
            self.last_thong_tin = []
            # Tạm dừng tự động quét (nhận dạng) để tránh xung đột
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
            # Chuyển về 1:N: khôi phục tự động quét nếu checkbox đang bật
            if self.check_tu_dong.isChecked():
                self.tu_dong_quet = True
            self.panel_ket_qua.hide()
            self.o_id_xac_minh.clear()
            self.o_id_xac_minh.setEnabled(False)

        self.lam_moi()

    def hien_thi_khung_xac_minh(self):
        """Hiển thị khung xác minh (1:1 mode)"""
        with QMutexLocker(self.buffer_mutex):
            if self.frame_buffer is None:
                return
            anh = self.frame_buffer.copy()
        box, _ = self.detector.phat_hien(anh)
        if box is not None:
            cv2.rectangle(anh, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            self.nhan_so_face.setText("👤 1 khuôn mặt")
        else:
            self.nhan_so_face.setText("👤 0 khuôn mặt")
        self._hien_thi_anh(anh)

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
                anh = self.frame_buffer.copy() if self.frame_buffer is not None else None
            if anh is not None:
                self.bat_dau_nhan_dang_tu_dong(anh)
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
        self.last_results = []
        self.last_boxes = []
        self.last_thong_tin = []

    def closeEvent(self, su_kien):
        self.active = False
        su_kien.accept()