import sys
import os
import json
import cv2
import numpy as np
import torch

from PIL import Image

from facenet_pytorch import MTCNN, InceptionResnetV1

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QStackedWidget, QFrame, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QSlider, QRadioButton,
    QButtonGroup, QScrollArea
)

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap


# ================================================================
# CẤU HÌNH
# ================================================================

THU_MUC_DU_LIEU = "dataset"
TEP_CSDL = os.path.join(
    THU_MUC_DU_LIEU,
    "face_database.json"
)

SO_ANH_DANG_KY = 50

# Có thể chỉnh trong giao diện
NGUONG_MAC_DINH = 0.45

# Số embedding tốt nhất được dùng để quyết định
TOP_K = 5


# ================================================================
# THIẾT BỊ
# ================================================================

try:
    THIET_BI = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
except Exception:
    THIET_BI = torch.device("cpu")


print("=" * 70)
print("FACE SECURE")
print("=" * 70)
print(f"Device: {THIET_BI}")

if THIET_BI.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("=" * 70)


# ================================================================
# STYLE
# ================================================================

STYLESHEET = """
QMainWindow {
    background-color: #F8FAFC;
    font-family: 'Segoe UI', Arial;
}

#Sidebar {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

#Sidebar QPushButton {
    text-align: left;
    padding: 14px 20px;
    font-size: 14px;
    font-weight: bold;
    color: #64748B;
    border: none;
    border-radius: 8px;
    margin: 4px 15px;
}

#Sidebar QPushButton:hover {
    background-color: #F1F5F9;
    color: #0F172A;
}

#Sidebar QPushButton:checked {
    background-color: #EFF6FF;
    color: #2563EB;
}

.Card {
    background-color: #FFFFFF;
    border-radius: 16px;
    border: 1px solid #E2E8F0;
}

.SuccessPanel {
    background-color: #ECFDF5;
    border: 1px solid #10B981;
    border-radius: 12px;
}

.ErrorPanel {
    background-color: #FEF2F2;
    border: 1px solid #EF4444;
    border-radius: 12px;
}

QLabel {
    font-size: 14px;
    color: #475569;
}

.LabelBold {
    font-weight: bold;
    color: #0F172A;
    font-size: 16px;
}

QLineEdit {
    padding: 12px;
    border: 1px solid #CBD5E1;
    border-radius: 8px;
    background-color: #FFFFFF;
    font-size: 14px;
    color: #0F172A;
}

QLineEdit:focus {
    border: 2px solid #3B82F6;
    background-color: #F8FAFC;
}

.PrimaryButton {
    background-color: #3B82F6;
    color: white;
    border-radius: 8px;
    padding: 14px 24px;
    font-weight: bold;
    font-size: 14px;
}

.PrimaryButton:hover {
    background-color: #2563EB;
}

.SecondaryButton {
    background-color: #E2E8F0;
    color: #475569;
    border-radius: 8px;
    padding: 14px 24px;
    font-weight: bold;
    font-size: 14px;
}

.SecondaryButton:hover {
    background-color: #CBD5E1;
}

.DeleteButton {
    background-color: #FEE2E2;
    color: #DC2626;
    border-radius: 6px;
    padding: 8px 15px;
    font-weight: bold;
    border: none;
}

.DeleteButton:hover {
    background-color: #FCA5A5;
}

QTableWidget {
    border: none;
    gridline-color: #F1F5F9;
    background-color: #FFFFFF;
    outline: none;
}

QHeaderView::section {
    background-color: #F8FAFC;
    padding: 12px;
    font-weight: bold;
    color: #475569;
    border: none;
    border-bottom: 2px solid #E2E8F0;
}

QTableWidget::item {
    padding: 5px 12px;
    border-bottom: 1px solid #F1F5F9;
    color: #0F172A;
}

QRadioButton {
    font-size: 14px;
    font-weight: bold;
    color: #0F172A;
}
"""


# ================================================================
# APPLICATION
# ================================================================

class FaceSecureApp(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "FaceSecure - Nhận diện khuôn mặt"
        )

        self.setGeometry(
            50, 50, 1280, 800
        )

        self.setStyleSheet(STYLESHEET)

        # --------------------------------------------------------
        # DATABASE
        # --------------------------------------------------------

        self.user_database = []

        self.user_id_counter = 1

        self.current_threshold = NGUONG_MAC_DINH

        # --------------------------------------------------------
        # REGISTRATION
        # --------------------------------------------------------

        self.is_register_scanning = False

        self.is_verify_scanning = False

        self.scan_frame_count = 0

        self.temp_user_info = {}

        self.danh_sach_embedding_tam = []

        self.danh_sach_anh_tam = []

        # --------------------------------------------------------
        # VERIFICATION
        # --------------------------------------------------------

        self.face_detected_in_verification = False

        self.embedding_camera_cuoi = None

        # --------------------------------------------------------
        # DATASET
        # --------------------------------------------------------

        os.makedirs(
            THU_MUC_DU_LIEU,
            exist_ok=True
        )

        # --------------------------------------------------------
        # CAMERA
        # --------------------------------------------------------

        self.cap = None

        self.timer = QTimer()

        self.timer.timeout.connect(
            self.update_frame
        )

        # --------------------------------------------------------
        # MODEL
        # --------------------------------------------------------

        self.load_models()

        # --------------------------------------------------------
        # UI
        # --------------------------------------------------------

        self.initUI()

        # --------------------------------------------------------
        # DATABASE
        # --------------------------------------------------------

        self.tai_co_so_du_lieu()

        print(
            f"[Database] Loaded: "
            f"{len(self.user_database)} users"
        )


    # ============================================================
    # LOAD MODEL
    # ============================================================

    def load_models(self):

        print("[Model] Loading MTCNN...")

        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            keep_all=True,
            device=THIET_BI
        )

        print("[Model] Loading InceptionResnetV1...")

        self.face_model = InceptionResnetV1(
            pretrained="vggface2"
        ).eval().to(THIET_BI)

        print("[Model] Ready.")

    # ============================================================
    # COSINE DISTANCE
    # ============================================================

    def compute_cosine_distance(
        self,
        emb1,
        emb2
    ):

        vec1 = np.asarray(
            emb1,
            dtype=np.float32
        )

        vec2 = np.asarray(
            emb2,
            dtype=np.float32
        )

        norm_a = np.linalg.norm(vec1)

        norm_b = np.linalg.norm(vec2)

        if norm_a == 0 or norm_b == 0:
            return 2.0

        cosine = np.dot(
            vec1,
            vec2
        ) / (
            norm_a * norm_b
        )

        cosine = np.clip(
            cosine,
            -1.0,
            1.0
        )

        return float(
            1.0 - cosine
        )

    # ============================================================
    # EXTRACT EMBEDDING
    # ============================================================

    def trich_xuat_embedding(
        self,
        anh_bgr
    ):

        try:

            anh_rgb = cv2.cvtColor(
                anh_bgr,
                cv2.COLOR_BGR2RGB
            )

            anh_pil = Image.fromarray(
                anh_rgb
            )

            # MTCNN tìm tất cả khuôn mặt
            boxes, probs = self.mtcnn.detect(
                anh_pil
            )

            if boxes is None:
                return None

            if len(boxes) == 0:
                return None

            # Chọn khuôn mặt lớn nhất
            best_index = 0
            best_area = 0

            for i, box in enumerate(boxes):

                x1, y1, x2, y2 = box

                area = max(
                    0,
                    x2 - x1
                ) * max(
                    0,
                    y2 - y1
                )

                if area > best_area:

                    best_area = area

                    best_index = i

            box = boxes[best_index]

            x1, y1, x2, y2 = box

            x1 = max(
                0,
                int(x1)
            )

            y1 = max(
                0,
                int(y1)
            )

            x2 = min(
                anh_rgb.shape[1],
                int(x2)
            )

            y2 = min(
                anh_rgb.shape[0],
                int(y2)
            )

            if x2 <= x1 or y2 <= y1:
                return None

            # Crop khuôn mặt
            face_crop = anh_rgb[
                y1:y2,
                x1:x2
            ]

            if face_crop.size == 0:
                return None

            face_pil = Image.fromarray(
                face_crop
            )

            # MTCNN alignment
            aligned = self.mtcnn(
                face_pil
            )

            # Trường hợp crop vẫn không detect
            if aligned is None:
                return None

            # Nếu keep_all=True thì output có thể là nhiều mặt
            if aligned.ndim == 4:

                aligned = aligned[0]

            with torch.no_grad():

                embedding = self.face_model(
                    aligned.unsqueeze(0).to(
                        THIET_BI
                    )
                )

            # L2 normalize
            embedding = torch.nn.functional.normalize(
                embedding,
                p=2,
                dim=1
            )

            embedding = (
                embedding
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            # Kiểm tra đúng 512D
            if embedding.shape[0] != 512:

                print(
                    "[Embedding] ERROR:",
                    embedding.shape
                )

                return None

            return embedding

        except Exception as e:

            print(
                f"[Embedding] ERROR: {e}"
            )

            return None

    # ============================================================
    # DETECT FACE
    # ============================================================

    def phat_hien_khuon_mat(
        self,
        anh_bgr
    ):

        try:

            rgb = cv2.cvtColor(
                anh_bgr,
                cv2.COLOR_BGR2RGB
            )

            pil = Image.fromarray(
                rgb
            )

            boxes, probabilities = (
                self.mtcnn.detect(pil)
            )

            if boxes is None:
                return None, 0.0

            if len(boxes) == 0:
                return None, 0.0

            best_box = None
            best_probability = 0
            best_area = 0

            for i, box in enumerate(boxes):

                x1, y1, x2, y2 = box

                area = max(
                    0,
                    x2 - x1
                ) * max(
                    0,
                    y2 - y1
                )

                probability = 0.0

                if probabilities is not None:
                    probability = float(
                        probabilities[i]
                    )

                if area > best_area:

                    best_area = area

                    best_box = tuple(
                        map(
                            int,
                            box
                        )
                    )

                    best_probability = (
                        probability
                    )

            return (
                best_box,
                best_probability
            )

        except Exception as e:

            print(
                "[Detection]",
                e
            )

            return None, 0.0

    # ============================================================
    # AVERAGE EMBEDDING
    # ============================================================

    def tinh_embedding_trung_binh(
        self,
        embeddings
    ):

        if not embeddings:
            return None

        matrix = np.vstack(
            embeddings
        )

        mean_vector = np.mean(
            matrix,
            axis=0
        )

        norm = np.linalg.norm(
            mean_vector
        )

        if norm == 0:
            return None

        mean_vector = (
            mean_vector / norm
        )

        return mean_vector.astype(
            np.float32
        )

    # ============================================================
    # DATABASE SAVE
    # ============================================================

    def luu_co_so_du_lieu(self):

        try:

            data = []

            for user in self.user_database:

                record = {
                    "id": int(user["id"]),
                    "name": user["name"],
                    "age": user.get("age", ""),
                    "home": user.get("home", ""),
                    "class": user.get("class", ""),
                    "major": user.get("major", ""),
                    "embedding_dimension": 512,
                    "image_count": len(
                        user.get(
                            "embeddings",
                            []
                        )
                    ),
                    "embeddings": [
                        np.asarray(
                            emb,
                            dtype=np.float32
                        ).tolist()
                        for emb in user.get(
                            "embeddings",
                            []
                        )
                    ],
                    "embedding": np.asarray(
                        user["embedding"],
                        dtype=np.float32
                    ).tolist()
                }

                data.append(record)

            with open(
                TEP_CSDL,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    data,
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            print(
                "[Database] Saved."
            )

        except Exception as e:

            print(
                "[Database] Save error:",
                e
            )

    # ============================================================
    # DATABASE LOAD
    # ============================================================

    def tai_co_so_du_lieu(self):

        if not os.path.exists(
            TEP_CSDL
        ):
            return

        try:

            with open(
                TEP_CSDL,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            self.user_database = []

            for user in data:

                # ------------------------------------------------
                # EMBEDDING 512D
                # ------------------------------------------------

                user["embedding"] = np.asarray(
                    user["embedding"],
                    dtype=np.float32
                )

                # ------------------------------------------------
                # EMBEDDINGS 50 ẢNH
                # ------------------------------------------------

                embeddings = []

                for emb in user.get(
                    "embeddings",
                    []
                ):

                    arr = np.asarray(
                        emb,
                        dtype=np.float32
                    )

                    if arr.shape[0] == 512:
                        embeddings.append(
                            arr
                        )

                user["embeddings"] = embeddings

                self.user_database.append(
                    user
                )

            if self.user_database:

                self.user_id_counter = (
                    max(
                        int(u["id"])
                        for u in self.user_database
                    ) + 1
                )

            print(
                f"[Database] Loaded "
                f"{len(self.user_database)} users"
            )

        except Exception as e:

            print(
                "[Database] Load error:",
                e
            )

            self.user_database = []

    # ============================================================
    # VERIFY 1:1
    # ============================================================

    def verify_1_1(
        self,
        current_embedding,
        claimed_user_id
    ):

        target_user = next(
            (
                u
                for u in self.user_database
                if str(u["id"])
                == str(claimed_user_id)
            ),
            None
        )

        if target_user is None:

            return (
                False,
                2.0,
                None,
                "ID không tồn tại."
            )

        # --------------------------------------------------------
        # Nếu có 50 embedding -> so với toàn bộ
        # --------------------------------------------------------

        embeddings = target_user.get(
            "embeddings",
            []
        )

        if embeddings:

            distances = [
                self.compute_cosine_distance(
                    current_embedding,
                    emb
                )
                for emb in embeddings
            ]

            distances.sort()

            top_k = distances[
                :min(
                    TOP_K,
                    len(distances)
                )
            ]

            distance = float(
                np.mean(top_k)
            )

        else:

            distance = (
                self.compute_cosine_distance(
                    current_embedding,
                    target_user["embedding"]
                )
            )

        print("=" * 60)
        print("1:1 VERIFICATION")
        print(
            f"ID: {target_user['id']}"
        )
        print(
            f"Name: {target_user['name']}"
        )
        print(
            f"Distance: {distance:.6f}"
        )
        print(
            f"Threshold: "
            f"{self.current_threshold:.6f}"
        )
        print("=" * 60)

        is_match = (
            distance
            <= self.current_threshold
        )

        if is_match:

            message = (
                "Xác minh thành công."
            )

        else:

            message = (
                "Sai khuôn mặt."
            )

        return (
            is_match,
            distance,
            target_user,
            message
        )

    # ============================================================
    # IDENTIFY 1:N
    # ============================================================

    def identify_1_N(
        self,
        current_embedding
    ):

        if not self.user_database:

            return (
                False,
                2.0,
                None,
                "CSDL trống.",
                []
            )

        results = []

        # --------------------------------------------------------
        # So sánh với từng người
        # --------------------------------------------------------

        for user in self.user_database:

            embeddings = user.get(
                "embeddings",
                []
            )

            # Database cũ chỉ có 1 embedding
            if not embeddings:

                embeddings = [
                    user["embedding"]
                ]

            user_distances = []

            for emb in embeddings:

                dist = (
                    self.compute_cosine_distance(
                        current_embedding,
                        emb
                    )
                )

                user_distances.append(
                    dist
                )

            user_distances.sort()

            # Top-K gần nhất
            top_k = user_distances[
                :min(
                    TOP_K,
                    len(user_distances)
                )
            ]

            # Median thường ổn định hơn 1 frame
            person_distance = float(
                np.median(top_k)
            )

            results.append({
                "user": user,
                "distance": person_distance,
                "all_distances":
                    user_distances
            })

        # --------------------------------------------------------
        # Sắp xếp
        # --------------------------------------------------------

        results.sort(
            key=lambda x: x["distance"]
        )

        best = results[0]

        best_user = best["user"]

        min_distance = (
            best["distance"]
        )

        is_match = (
            min_distance
            <= self.current_threshold
        )

        print("\n")
        print("=" * 70)
        print("1:N IDENTIFICATION")
        print("=" * 70)

        for item in results:

            user = item["user"]

            print(
                f"ID_{user['id']} "
                f"{user['name']:<25} "
                f"Distance = "
                f"{item['distance']:.6f}"
            )

        print("-" * 70)

        print(
            f"BEST = "
            f"ID_{best_user['id']} "
            f"{best_user['name']}"
        )

        print(
            f"Distance = "
            f"{min_distance:.6f}"
        )

        print(
            f"Threshold = "
            f"{self.current_threshold:.6f}"
        )

        print("=" * 70)

        if is_match:

            message = (
                "Nhận dạng thành công."
            )

        else:

            best_user = None

            message = (
                "Người lạ."
            )

        return (
            is_match,
            min_distance,
            best_user,
            message,
            results
        )

    # ============================================================
    # UI
    # ============================================================

    def initUI(self):

        main_widget = QWidget()

        self.setCentralWidget(
            main_widget
        )

        main_layout = QHBoxLayout(
            main_widget
        )

        main_layout.setContentsMargins(
            0, 0, 0, 0
        )

        main_layout.setSpacing(0)

        # --------------------------------------------------------
        # SIDEBAR
        # --------------------------------------------------------

        sidebar = QFrame()

        sidebar.setObjectName(
            "Sidebar"
        )

        sidebar.setFixedWidth(
            280
        )

        sidebar_layout = QVBoxLayout(
            sidebar
        )

        sidebar_layout.setContentsMargins(
            0, 40, 0, 40
        )

        logo = QLabel(
            " FACE ID\n"
            " Windows Application"
        )

        logo.setStyleSheet(
            """
            font-weight: 900;
            color: #2563EB;
            font-size: 18px;
            margin-left: 20px;
            """
        )

        sidebar_layout.addWidget(
            logo
        )

        sidebar_layout.addSpacing(
            40
        )

        self.btn_nav_nhandang = (
            QPushButton(
                "👤 Thu thập Dataset"
            )
        )

        self.btn_nav_xacminh = (
            QPushButton(
                "✔️ Nhận dạng & Xác minh"
            )
        )

        self.btn_nav_quanly = (
            QPushButton(
                "🗄️ Quản lý dữ liệu"
            )
        )

        self.btn_nav_caidat = (
            QPushButton(
                "⚙️ Cài đặt ngưỡng"
            )
        )

        buttons = [
            self.btn_nav_nhandang,
            self.btn_nav_xacminh,
            self.btn_nav_quanly,
            self.btn_nav_caidat
        ]

        for button in buttons:

            button.setCheckable(True)

            button.setCursor(
                Qt.PointingHandCursor
            )

            sidebar_layout.addWidget(
                button
            )

        sidebar_layout.addStretch()

        # --------------------------------------------------------
        # CONTENT
        # --------------------------------------------------------

        self.content_area = (
            QStackedWidget()
        )

        main_layout.addWidget(
            sidebar
        )

        main_layout.addWidget(
            self.content_area
        )

        # --------------------------------------------------------
        # PAGES
        # --------------------------------------------------------

        self.page_nhandang = (
            self.create_page_nhandang()
        )

        self.page_xacminh = (
            self.create_page_xacminh()
        )

        self.page_quanly = (
            self.create_page_quanly()
        )

        self.page_caidat = (
            self.create_page_caidat()
        )

        self.content_area.addWidget(
            self.page_nhandang
        )

        self.content_area.addWidget(
            self.page_xacminh
        )

        self.content_area.addWidget(
            self.page_quanly
        )

        self.content_area.addWidget(
            self.page_caidat
        )

        # --------------------------------------------------------
        # EVENTS
        # --------------------------------------------------------

        self.btn_nav_nhandang.clicked.connect(
            lambda:
            self.switch_page(
                0,
                self.btn_nav_nhandang
            )
        )

        self.btn_nav_xacminh.clicked.connect(
            lambda:
            self.switch_page(
                1,
                self.btn_nav_xacminh
            )
        )

        self.btn_nav_quanly.clicked.connect(
            lambda:
            self.switch_page(
                2,
                self.btn_nav_quanly
            )
        )

        self.btn_nav_caidat.clicked.connect(
            lambda:
            self.switch_page(
                3,
                self.btn_nav_caidat
            )
        )

        self.switch_page(
            0,
            self.btn_nav_nhandang
        )

    # ============================================================
    # SWITCH PAGE
    # ============================================================

    def switch_page(
        self,
        index,
        active_btn
    ):

        self.content_area.setCurrentIndex(
            index
        )

        for btn in [
            self.btn_nav_nhandang,
            self.btn_nav_xacminh,
            self.btn_nav_quanly,
            self.btn_nav_caidat
        ]:

            btn.setChecked(False)

        active_btn.setChecked(True)

        if index in [0, 1]:

            self.start_camera()

        else:

            self.stop_camera()

        if index == 2:

            self.refresh_data_table()

        self.result_panel.setVisible(
            False
        )

        self.is_register_scanning = False

        self.is_verify_scanning = False

    # ============================================================
    # PAGE REGISTER
    # ============================================================

    def create_page_nhandang(
        self
    ):

        page = QWidget()

        layout = QVBoxLayout(
            page
        )

        layout.setContentsMargins(
            50, 50, 50, 50
        )

        layout.addWidget(
            QLabel(
                "<h1>THU THẬP DỮ LIỆU</h1>"
            )
        )

        content = QHBoxLayout()

        # --------------------------------------------------------
        # FORM
        # --------------------------------------------------------

        form_card = QFrame()

        form_card.setProperty(
            "class",
            "Card"
        )

        form_layout = QVBoxLayout(
            form_card
        )

        form_layout.setContentsMargins(
            30, 30, 30, 30
        )

        self.inp_name = QLineEdit()
        self.inp_name.setPlaceholderText(
            "Họ và tên"
        )

        form_layout.addWidget(
            QLabel("Họ và tên *")
        )

        form_layout.addWidget(
            self.inp_name
        )

        self.inp_age = QLineEdit()
        self.inp_age.setPlaceholderText(
            "Tuổi"
        )

        form_layout.addWidget(
            QLabel("Tuổi")
        )

        form_layout.addWidget(
            self.inp_age
        )

        self.inp_home = QLineEdit()
        self.inp_home.setPlaceholderText(
            "Quê quán"
        )

        form_layout.addWidget(
            QLabel("Quê quán")
        )

        form_layout.addWidget(
            self.inp_home
        )

        self.inp_class = QLineEdit()
        self.inp_class.setPlaceholderText(
            "Lớp học"
        )

        form_layout.addWidget(
            QLabel("Lớp học")
        )

        form_layout.addWidget(
            self.inp_class
        )

        self.inp_major = QLineEdit()
        self.inp_major.setPlaceholderText(
            "Ngành học"
        )

        form_layout.addWidget(
            QLabel("Ngành học")
        )

        form_layout.addWidget(
            self.inp_major
        )

        form_layout.addStretch()

        # --------------------------------------------------------
        # CAMERA
        # --------------------------------------------------------

        cam_card = QFrame()

        cam_card.setProperty(
            "class",
            "Card"
        )

        cam_layout = QVBoxLayout(
            cam_card
        )

        cam_layout.setContentsMargins(
            30, 30, 30, 30
        )

        self.cam_label_nhandang = QLabel()

        self.cam_label_nhandang.setAlignment(
            Qt.AlignCenter
        )

        self.cam_label_nhandang.setStyleSheet(
            """
            background-color: #0F172A;
            border-radius: 12px;
            """
        )

        self.cam_label_nhandang.setMinimumSize(
            500,
            450
        )

        cam_layout.addWidget(
            self.cam_label_nhandang
        )

        content.addWidget(
            form_card,
            1
        )

        content.addWidget(
            cam_card,
            1
        )

        layout.addLayout(
            content
        )

        # --------------------------------------------------------
        # BUTTONS
        # --------------------------------------------------------

        buttons = QHBoxLayout()

        buttons.addStretch()

        self.btn_clear = QPushButton(
            "🧹 Làm mới Form"
        )

        self.btn_clear.setProperty(
            "class",
            "SecondaryButton"
        )

        self.btn_clear.clicked.connect(
            self.clear_registration_form
        )

        self.btn_submit = QPushButton(
            "💾 Bắt đầu đăng ký 50 ảnh"
        )

        self.btn_submit.setProperty(
            "class",
            "PrimaryButton"
        )

        self.btn_submit.clicked.connect(
            self.start_registration_scan
        )

        buttons.addWidget(
            self.btn_clear
        )

        buttons.addWidget(
            self.btn_submit
        )

        layout.addLayout(
            buttons
        )

        return page

    # ============================================================
    # PAGE VERIFY
    # ============================================================

    def create_page_xacminh(
        self
    ):

        page = QWidget()

        layout = QVBoxLayout(
            page
        )

        layout.setContentsMargins(
            30, 30, 30, 30
        )

        layout.addWidget(
            QLabel(
                "<h1>NHẬN DẠNG 1:1 / 1:N</h1>"
            )
        )

        # --------------------------------------------------------
        # MODE
        # --------------------------------------------------------

        mode_card = QFrame()

        mode_card.setProperty(
            "class",
            "Card"
        )

        mode_layout = QHBoxLayout(
            mode_card
        )

        self.radio_group = (
            QButtonGroup(self)
        )

        self.radio_1_N = (
            QRadioButton(
                "Nhận dạng 1:N"
            )
        )

        self.radio_1_1 = (
            QRadioButton(
                "Xác minh 1:1"
            )
        )

        self.radio_1_N.setChecked(
            True
        )

        self.radio_group.addButton(
            self.radio_1_N
        )

        self.radio_group.addButton(
            self.radio_1_1
        )

        self.inp_verify_id = QLineEdit()

        self.inp_verify_id.setPlaceholderText(
            "ID cần xác minh"
        )

        self.inp_verify_id.setFixedWidth(
            180
        )

        self.inp_verify_id.setEnabled(
            False
        )

        self.radio_1_1.toggled.connect(
            lambda:
            self.inp_verify_id.setEnabled(
                self.radio_1_1.isChecked()
            )
        )

        mode_layout.addWidget(
            self.radio_1_N
        )

        mode_layout.addWidget(
            self.radio_1_1
        )

        mode_layout.addWidget(
            self.inp_verify_id
        )

        mode_layout.addStretch()

        layout.addWidget(
            mode_card
        )

        # --------------------------------------------------------
        # CONTENT
        # --------------------------------------------------------

        verify_layout = QHBoxLayout()

        self.cam_label_xacminh = QLabel()

        self.cam_label_xacminh.setFixedSize(
            640,
            480
        )

        self.cam_label_xacminh.setAlignment(
            Qt.AlignCenter
        )

        self.cam_label_xacminh.setStyleSheet(
            """
            background-color: #0F172A;
            border-radius: 16px;
            """
        )

        verify_layout.addWidget(
            self.cam_label_xacminh
        )

        # --------------------------------------------------------
        # RESULT
        # --------------------------------------------------------

        self.result_panel = QFrame()

        self.result_panel.setProperty(
            "class",
            "SuccessPanel"
        )

        self.result_panel.setFixedWidth(
            430
        )

        res_layout = QVBoxLayout(
            self.result_panel
        )

        res_layout.setContentsMargins(
            25, 25, 25, 25
        )

        self.lbl_status = QLabel(
            "-"
        )

        self.lbl_desc = QLabel(
            "-"
        )

        self.lbl_desc.setWordWrap(
            True
        )

        self.res_dist = QLabel(
            "Distance: -"
        )

        self.res_dist.setStyleSheet(
            """
            color: #2563EB;
            font-weight: bold;
            font-size: 17px;
            """
        )

        self.res_name = QLabel(
            "Họ tên: -"
        )

        self.res_class = QLabel(
            "Lớp: -"
        )

        self.lbl_all_distances = QLabel()

        self.lbl_all_distances.setWordWrap(
            True
        )

        self.lbl_all_distances.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )

        self.lbl_all_distances.setStyleSheet(
            """
            QLabel {
                font-size: 14px;
                color: #0F172A;
                background-color: #F8FAFC;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                padding: 12px;
            }
            """
        )

        res_layout.addWidget(
            self.lbl_status
        )

        res_layout.addWidget(
            self.lbl_desc
        )

        res_layout.addSpacing(
            10
        )

        res_layout.addWidget(
            self.res_dist
        )

        res_layout.addWidget(
            self.res_name
        )

        res_layout.addWidget(
            self.res_class
        )

        res_layout.addSpacing(
            10
        )

        res_layout.addWidget(
            self.lbl_all_distances
        )

        res_layout.addStretch()

        verify_layout.addWidget(
            self.result_panel
        )

        self.result_panel.setVisible(
            False
        )

        layout.addLayout(
            verify_layout
        )

        # --------------------------------------------------------
        # BUTTON
        # --------------------------------------------------------

        btn_verify = QPushButton(
            "🔍 Bắt đầu đối sánh"
        )

        btn_verify.setProperty(
            "class",
            "PrimaryButton"
        )

        btn_verify.clicked.connect(
            self.start_verification_scan
        )

        layout.addWidget(
            btn_verify,
            alignment=Qt.AlignCenter
        )

        return page

    # ============================================================
    # PAGE DATABASE
    # ============================================================

    def create_page_quanly(
        self
    ):

        page = QWidget()

        layout = QVBoxLayout(
            page
        )

        layout.setContentsMargins(
            50, 50, 50, 50
        )

        layout.addWidget(
            QLabel(
                "<h1>QUẢN LÝ DATABASE</h1>"
            )
        )

        card = QFrame()

        card.setProperty(
            "class",
            "Card"
        )

        card_layout = QVBoxLayout(
            card
        )

        self.table = QTableWidget(
            0,
            6
        )

        self.table.setHorizontalHeaderLabels(
            [
                "ID",
                "HỌ TÊN",
                "LỚP",
                "NGÀNH",
                "SỐ EMBEDDING",
                "THAO TÁC"
            ]
        )

        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )

        self.table.verticalHeader().setVisible(
            False
        )

        self.table.setShowGrid(
            False
        )

        card_layout.addWidget(
            self.table
        )

        layout.addWidget(
            card
        )

        return page

    # ============================================================
    # PAGE SETTINGS
    # ============================================================

    def create_page_caidat(
        self
    ):

        page = QWidget()

        layout = QVBoxLayout(
            page
        )

        layout.setContentsMargins(
            50, 50, 50, 50
        )

        layout.addWidget(
            QLabel(
                "<h1>CÀI ĐẶT NGƯỠNG</h1>"
            )
        )

        card = QFrame()

        card.setProperty(
            "class",
            "Card"
        )

        card_layout = QVBoxLayout(
            card
        )

        card_layout.addWidget(
            QLabel(
                "Cosine Distance càng nhỏ "
                "thì khuôn mặt càng giống."
            )
        )

        self.lbl_threshold_val = QLabel(
            f"Threshold: "
            f"{self.current_threshold:.2f}"
        )

        self.lbl_threshold_val.setProperty(
            "class",
            "LabelBold"
        )

        card_layout.addWidget(
            self.lbl_threshold_val
        )

        self.slider = QSlider(
            Qt.Horizontal
        )

        self.slider.setMinimum(
            10
        )

        self.slider.setMaximum(
            100
        )

        self.slider.setValue(
            int(
                self.current_threshold * 100
            )
        )

        self.slider.valueChanged.connect(
            self.update_threshold
        )

        card_layout.addWidget(
            self.slider
        )

        layout.addWidget(
            card
        )

        layout.addStretch()

        return page

    # ============================================================
    # THRESHOLD
    # ============================================================

    def update_threshold(
        self,
        value
    ):

        self.current_threshold = (
            value / 100.0
        )

        self.lbl_threshold_val.setText(
            f"Threshold: "
            f"{self.current_threshold:.2f}"
        )

    # ============================================================
    # CLEAR FORM
    # ============================================================

    def clear_registration_form(
        self
    ):

        self.inp_name.clear()

        self.inp_age.clear()

        self.inp_home.clear()

        self.inp_class.clear()

        self.inp_major.clear()

    # ============================================================
    # START REGISTRATION
    # ============================================================

    def start_registration_scan(
        self
    ):

        name = self.inp_name.text().strip()

        if not name:

            QMessageBox.warning(
                self,
                "Lỗi",
                "Vui lòng nhập họ tên."
            )

            return

        # Reset
        self.danh_sach_embedding_tam = []

        self.danh_sach_anh_tam = []

        self.scan_frame_count = 0

        self.temp_user_info = {
            "id":
                self.user_id_counter,

            "name":
                name,

            "age":
                self.inp_age.text().strip(),

            "home":
                self.inp_home.text().strip(),

            "class":
                self.inp_class.text().strip(),

            "major":
                self.inp_major.text().strip()
        }

        self.is_register_scanning = True

        self.btn_submit.setText(
            "Đang thu thập 0/50..."
        )

        self.btn_submit.setEnabled(
            False
        )

        print(
            f"[Register] "
            f"Start ID={self.user_id_counter}"
        )

    # ============================================================
    # FINISH REGISTRATION
    # ============================================================

    def finish_registration(
        self
    ):

        self.is_register_scanning = False

        embeddings = (
            self.danh_sach_embedding_tam
        )

        if len(embeddings) < 10:

            QMessageBox.warning(
                self,
                "Lỗi",
                "Không đủ embedding hợp lệ."
            )

            self.btn_submit.setText(
                "💾 Bắt đầu đăng ký 50 ảnh"
            )

            self.btn_submit.setEnabled(
                True
            )

            return

        mean_embedding = (
            self.tinh_embedding_trung_binh(
                embeddings
            )
        )

        if mean_embedding is None:

            QMessageBox.warning(
                self,
                "Lỗi",
                "Không thể tạo embedding trung bình."
            )

            self.btn_submit.setEnabled(
                True
            )

            return

        # --------------------------------------------------------
        # DATABASE RECORD
        # --------------------------------------------------------

        user = dict(
            self.temp_user_info
        )

        user["embedding"] = (
            mean_embedding
        )

        user["embeddings"] = [
            np.asarray(
                emb,
                dtype=np.float32
            )
            for emb in embeddings
        ]

        user["embedding_dimension"] = 512

        user["image_count"] = len(
            embeddings
        )

        self.user_database.append(
            user
        )

        self.luu_co_so_du_lieu()

        self.user_id_counter += 1

        count = len(
            embeddings
        )

        self.danh_sach_embedding_tam = []

        self.danh_sach_anh_tam = []

        self.btn_submit.setText(
            "💾 Bắt đầu đăng ký 50 ảnh"
        )

        self.btn_submit.setEnabled(
            True
        )

        QMessageBox.information(
            self,
            "Thành công",
            f"Đã đăng ký:\n\n"
            f"ID: {user['id']}\n"
            f"Họ tên: {user['name']}\n"
            f"Embedding: 512D\n"
            f"Số embedding: {count}"
        )

        print(
            f"[Register] "
            f"ID={user['id']} "
            f"{count} embeddings"
        )

    # ============================================================
    # START VERIFICATION
    # ============================================================

    def start_verification_scan(
        self
    ):

        if not self.user_database:

            QMessageBox.warning(
                self,
                "Cảnh báo",
                "Database đang trống."
            )

            return

        if (
            self.radio_1_1.isChecked()
            and
            not self.inp_verify_id.text().strip()
        ):

            QMessageBox.warning(
                self,
                "Cảnh báo",
                "Vui lòng nhập ID."
            )

            return

        self.result_panel.setVisible(
            False
        )

        self.face_detected_in_verification = (
            False
        )

        self.embedding_camera_cuoi = None

        self.scan_frame_count = 0

        self.is_verify_scanning = True

    # ============================================================
    # FINISH VERIFICATION
    # ============================================================

    def finish_verification(
        self
    ):

        self.is_verify_scanning = False

        self.result_panel.setVisible(
            True
        )

        if (
            not self.face_detected_in_verification
            or
            self.embedding_camera_cuoi is None
        ):

            self.set_result_ui(
                False,
                2.0,
                "❌ Không nhận được khuôn mặt",
                "Không thể tạo embedding.",
                None
            )

            self.lbl_all_distances.setText(
                "Không có embedding camera."
            )

            return

        camera_embedding = (
            self.embedding_camera_cuoi
        )

        # ========================================================
        # 1:1
        # ========================================================

        if self.radio_1_1.isChecked():

            claimed_id = (
                self.inp_verify_id
                .text()
                .strip()
            )

            (
                success,
                distance,
                user,
                message
            ) = self.verify_1_1(
                camera_embedding,
                claimed_id
            )

            title = (
                "✅ Xác minh 1:1 Thành công"
                if success
                else
                "❌ Xác minh 1:1 Thất bại"
            )

            self.set_result_ui(
                success,
                distance,
                title,
                message,
                user
            )

            self.lbl_all_distances.setText(
                f"""
                <b>1:1 DISTANCE</b><br><br>
                ID_{claimed_id}: 
                <b>{distance:.6f}</b><br><br>
                Threshold:
                <b>{self.current_threshold:.6f}</b>
                """
            )

        # ========================================================
        # 1:N
        # ========================================================

        else:

            (
                success,
                distance,
                user,
                message,
                results
            ) = self.identify_1_N(
                camera_embedding
            )

            title = (
                "✅ Nhận dạng 1:N Thành công"
                if success
                else
                "❌ Không thuộc hệ thống"
            )

            self.set_result_ui(
                success,
                distance,
                title,
                message,
                user
            )

            # ----------------------------------------------------
            # HIỂN THỊ DISTANCE TỪNG NGƯỜI
            # ----------------------------------------------------

            html = (
                "<b>📊 DISTANCE TỪNG NGƯỜI</b>"
                "<br><br>"
            )

            for index, item in enumerate(
                results
            ):

                u = item["user"]

                dist = item["distance"]

                if index == 0:

                    html += (
                        "🏆 "
                        "<b>"
                        f"ID_{u['id']} - "
                        f"{u['name']}"
                        "</b>"
                    )

                else:

                    html += (
                        "👤 "
                        f"ID_{u['id']} - "
                        f"{u['name']}"
                    )

                html += (
                    "<br>"
                    f"&nbsp;&nbsp;&nbsp;"
                    f"Distance: "
                    f"<b>{dist:.6f}</b>"
                    "<br><br>"
                )

            html += (
                "<hr>"
                f"Threshold: "
                f"<b>{self.current_threshold:.6f}</b>"
            )

            self.lbl_all_distances.setText(
                html
            )

    # ============================================================
    # RESULT UI
    # ============================================================

    def set_result_ui(
        self,
        success,
        distance,
        title,
        desc,
        user
    ):

        if success:

            self.result_panel.setProperty(
                "class",
                "SuccessPanel"
            )

            self.lbl_status.setStyleSheet(
                """
                font-size: 20px;
                font-weight: bold;
                color: #047857;
                """
            )

        else:

            self.result_panel.setProperty(
                "class",
                "ErrorPanel"
            )

            self.lbl_status.setStyleSheet(
                """
                font-size: 20px;
                font-weight: bold;
                color: #DC2626;
                """
            )

        self.result_panel.style().unpolish(
            self.result_panel
        )

        self.result_panel.style().polish(
            self.result_panel
        )

        self.lbl_status.setText(
            title
        )

        self.lbl_desc.setText(
            desc
        )

        self.res_dist.setText(
            f"Cosine Distance: "
            f"{distance:.6f}"
        )

        if user:

            self.res_name.setText(
                f"ID_{user['id']} - "
                f"{user['name']}"
            )

            self.res_class.setText(
                f"Lớp: "
                f"{user.get('class', '-')}"
            )

        else:

            self.res_name.setText(
                "Họ tên: Không xác định"
            )

            self.res_class.setText(
                "Lớp: -"
            )

    # ============================================================
    # REFRESH TABLE
    # ============================================================

    def refresh_data_table(
        self
    ):

        self.table.setRowCount(0)

        for row, user in enumerate(
            self.user_database
        ):

            self.table.insertRow(
                row
            )

            self.table.setRowHeight(
                row,
                60
            )

            def create_item(text):

                item = QTableWidgetItem(
                    str(text)
                )

                item.setTextAlignment(
                    Qt.AlignVCenter
                    |
                    Qt.AlignLeft
                )

                return item

            self.table.setItem(
                row,
                0,
                create_item(
                    f"ID_{user['id']}"
                )
            )

            self.table.setItem(
                row,
                1,
                create_item(
                    user["name"]
                )
            )

            self.table.setItem(
                row,
                2,
                create_item(
                    user.get(
                        "class",
                        ""
                    )
                )
            )

            self.table.setItem(
                row,
                3,
                create_item(
                    user.get(
                        "major",
                        ""
                    )
                )
            )

            self.table.setItem(
                row,
                4,
                create_item(
                    len(
                        user.get(
                            "embeddings",
                            []
                        )
                    )
                )
            )

            btn_delete = QPushButton(
                "🗑️ Xóa"
            )

            btn_delete.setProperty(
                "class",
                "DeleteButton"
            )

            btn_delete.clicked.connect(
                lambda checked,
                uid=user["id"]:
                self.delete_user(uid)
            )

            widget = QWidget()

            btn_layout = QHBoxLayout(
                widget
            )

            btn_layout.setContentsMargins(
                10, 5, 10, 5
            )

            btn_layout.addWidget(
                btn_delete
            )

            btn_layout.setAlignment(
                Qt.AlignCenter
            )

            self.table.setCellWidget(
                row,
                5,
                widget
            )

    # ============================================================
    # DELETE USER
    # ============================================================

    def delete_user(
        self,
        user_id
    ):

        reply = QMessageBox.question(
            self,
            "Xác nhận",
            f"Xóa ID_{user_id}?",
            QMessageBox.Yes
            |
            QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        self.user_database = [
            u
            for u in self.user_database
            if int(u["id"])
            != int(user_id)
        ]

        self.luu_co_so_du_lieu()

        # Xóa ảnh
        for i in range(
            1,
            SO_ANH_DANG_KY + 1
        ):

            path = (
                f"dataset/"
                f"user.{user_id}.{i}.jpg"
            )

            if os.path.exists(path):

                try:
                    os.remove(path)

                except Exception:
                    pass

        self.refresh_data_table()

    # ============================================================
    # CAMERA START
    # ============================================================

    def start_camera(
        self
    ):

        if (
            self.cap is not None
            and
            self.cap.isOpened()
        ):

            return

        self.cap = cv2.VideoCapture(
            0,
            cv2.CAP_DSHOW
        )

        if not self.cap.isOpened():

            self.cap = cv2.VideoCapture(
                0
            )

        if not self.cap.isOpened():

            self.cap = None

            QMessageBox.critical(
                self,
                "Camera",
                "Không thể mở webcam."
            )

            return

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            640
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            480
        )

        self.timer.start(
            30
        )

    # ============================================================
    # CAMERA STOP
    # ============================================================

    def stop_camera(
        self
    ):

        self.timer.stop()

        if self.cap is not None:

            self.cap.release()

            self.cap = None

    # ============================================================
    # UPDATE FRAME
    # ============================================================

    def update_frame(
        self
    ):

        if (
            self.cap is None
            or
            not self.cap.isOpened()
        ):

            return

        ret, frame = (
            self.cap.read()
        )

        if not ret:
            return

        frame = cv2.flip(
            frame,
            1
        )

        h, w, _ = frame.shape

        # --------------------------------------------------------
        # FACE DETECTION
        # --------------------------------------------------------

        box, probability = (
            self.phat_hien_khuon_mat(
                frame
            )
        )

        if box is not None:

            x1, y1, x2, y2 = box

            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                w - 1,
                x2
            )

            y2 = min(
                h - 1,
                y2
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"MTCNN: {probability:.2f}",
                (
                    x1,
                    max(
                        25,
                        y1 - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

        # ========================================================
        # REGISTRATION
        # ========================================================

        if self.is_register_scanning:

            self.scan_frame_count += 1

            if (
                self.scan_frame_count % 3 == 0
                and
                box is not None
            ):

                embedding = (
                    self.trich_xuat_embedding(
                        frame
                    )
                )

                if embedding is not None:

                    count = len(
                        self.danh_sach_embedding_tam
                    )

                    if count < SO_ANH_DANG_KY:

                        self.danh_sach_embedding_tam.append(
                            embedding
                        )

                        # ----------------------------------------
                        # SAVE FACE IMAGE
                        # ----------------------------------------

                        rgb = cv2.cvtColor(
                            frame,
                            cv2.COLOR_BGR2RGB
                        )

                        x1, y1, x2, y2 = box

                        x1 = max(
                            0,
                            x1
                        )

                        y1 = max(
                            0,
                            y1
                        )

                        x2 = min(
                            w,
                            x2
                        )

                        y2 = min(
                            h,
                            y2
                        )

                        face = rgb[
                            y1:y2,
                            x1:x2
                        ]

                        if face.size > 0:

                            face = cv2.resize(
                                face,
                                (160, 160)
                            )

                            filename = (
                                f"dataset/"
                                f"user."
                                f"{self.temp_user_info['id']}."
                                f"{count + 1}.jpg"
                            )

                            cv2.imwrite(
                                filename,
                                cv2.cvtColor(
                                    face,
                                    cv2.COLOR_RGB2BGR
                                )
                            )

                            self.danh_sach_anh_tam.append(
                                filename
                            )

            count = len(
                self.danh_sach_embedding_tam
            )

            # ----------------------------------------------------
            # INSTRUCTION
            # ----------------------------------------------------

            if count < 10:

                instruction = (
                    "NHIN THANG"
                )

            elif count < 20:

                instruction = (
                    "QUAY MAT SANG TRAI"
                )

            elif count < 30:

                instruction = (
                    "QUAY MAT SANG PHAI"
                )

            elif count < 40:

                instruction = (
                    "NGUA MAT LEN"
                )

            else:

                instruction = (
                    "CUI MAT XUONG"
                )

            cv2.putText(
                frame,
                instruction,
                (30, 40),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Embedding: "
                f"{count}/{SO_ANH_DANG_KY}",
                (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            if box is None:

                cv2.putText(
                    frame,
                    "Khong tim thay khuon mat",
                    (30, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            self.btn_submit.setText(
                f"Dang thu thap "
                f"{count}/{SO_ANH_DANG_KY}..."
            )

            if count >= SO_ANH_DANG_KY:

                self.finish_registration()

        # ========================================================
        # VERIFICATION
        # ========================================================

        elif self.is_verify_scanning:

            self.scan_frame_count += 1

            if (
                self.scan_frame_count % 2 == 0
                and
                box is not None
            ):

                embedding = (
                    self.trich_xuat_embedding(
                        frame
                    )
                )

                if embedding is not None:

                    self.embedding_camera_cuoi = (
                        embedding
                    )

                    self.face_detected_in_verification = (
                        True
                    )

            cv2.putText(
                frame,
                (
                    "VERIFY 1:1"
                    if self.radio_1_1.isChecked()
                    else
                    "IDENTIFY 1:N"
                ),
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 200, 0),
                2
            )

            cv2.putText(
                frame,
                f"Scan: "
                f"{self.scan_frame_count}/30",
                (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            if (
                self.scan_frame_count >= 30
            ):

                self.finish_verification()

        # ========================================================
        # DISPLAY CAMERA
        # ========================================================

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        qt_img = QImage(
            frame_rgb.data,
            w,
            h,
            3 * w,
            QImage.Format_RGB888
        )

        if (
            self.content_area.currentIndex()
            == 0
        ):

            self.cam_label_nhandang.setPixmap(
                QPixmap.fromImage(
                    qt_img
                ).scaled(
                    self.cam_label_nhandang.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

        elif (
            self.content_area.currentIndex()
            == 1
        ):

            self.cam_label_xacminh.setPixmap(
                QPixmap.fromImage(
                    qt_img
                ).scaled(
                    self.cam_label_xacminh.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

    # ============================================================
    # CLOSE
    # ============================================================

    def closeEvent(
        self,
        event
    ):

        self.stop_camera()

        event.accept()


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )

    window = FaceSecureApp()

    window.showMaximized()

    sys.exit(
        app.exec_()
    )