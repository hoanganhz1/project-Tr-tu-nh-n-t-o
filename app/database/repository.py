import json
import os
import numpy as np

from app.config.settings import TEP_CSDL
from app.database.models import NguoiDung


class CoSoDuLieu:

    def __init__(self, tep_csdL=TEP_CSDL):

        self.tep_csdL = tep_csdL

        self.danh_sach_nguoi = []

        self.tai_du_lieu()


    # ========================================================
    # LOAD
    # ========================================================

    def tai_du_lieu(self):

        if not os.path.exists(self.tep_csdL):

            self.danh_sach_nguoi = []

            return

        try:

            with open(
                self.tep_csdL,
                "r",
                encoding="utf-8"
            ) as tep:

                du_lieu = json.load(tep)

            # JSON phiên bản mới
            if isinstance(du_lieu, dict):

                danh_sach = du_lieu.get(
                    "users",
                    []
                )

            # Hỗ trợ JSON cũ
            else:

                danh_sach = du_lieu

            self.danh_sach_nguoi = []

            for du_lieu_nguoi in danh_sach:

                try:

                    nguoi = NguoiDung.from_dict(
                        du_lieu_nguoi
                    )

                    # Kiểm tra embedding
                    embedding = np.asarray(
                        nguoi.embedding,
                        dtype=np.float32
                    )

                    if embedding.shape != (512,):

                        print(
                            f"[WARNING] "
                            f"ID {nguoi.id} không phải embedding 512D."
                        )

                        continue

                    nguoi.embedding = embedding.tolist()

                    self.danh_sach_nguoi.append(
                        nguoi
                    )

                except Exception as loi:

                    print(
                        f"[DATABASE] "
                        f"Bỏ qua bản ghi lỗi: {loi}"
                    )

        except Exception as loi:

            print(
                f"[DATABASE] Không thể đọc JSON: {loi}"
            )

            self.danh_sach_nguoi = []


    # ========================================================
    # SAVE
    # ========================================================

    def luu_du_lieu(self):

        du_lieu = {

            "version": 1,

            "model": {
                "name": "InceptionResnetV1",
                "pretrained": "vggface2",
                "embedding_dimension": 512,
                "distance": "cosine"
            },

            "users": [
                nguoi.to_dict()
                for nguoi in self.danh_sach_nguoi
            ]
        }

        with open(
            self.tep_csdL,
            "w",
            encoding="utf-8"
        ) as tep:

            json.dump(
                du_lieu,
                tep,
                ensure_ascii=False,
                indent=2
            )


    # ========================================================
    # GET ALL
    # ========================================================

    def lay_tat_ca_nguoi(self):

        return self.danh_sach_nguoi.copy()


    # ========================================================
    # GET USER
    # ========================================================

    def lay_nguoi_theo_id(self, user_id):

        for nguoi in self.danh_sach_nguoi:

            if int(nguoi.id) == int(user_id):

                return nguoi

        return None


    # ========================================================
    # CREATE
    # ========================================================

    def them_nguoi(self, nguoi):

        self.danh_sach_nguoi.append(
            nguoi
        )

        self.luu_du_lieu()


    # ========================================================
    # DELETE
    # ========================================================

    def xoa_nguoi(self, user_id):

        self.danh_sach_nguoi = [

            nguoi

            for nguoi in self.danh_sach_nguoi

            if int(nguoi.id) != int(user_id)
        ]

        self.luu_du_lieu()


    # ========================================================
    # NEXT ID
    # ========================================================

    def lay_id_tiep_theo(self):

        if not self.danh_sach_nguoi:

            return 1

        return max(
            nguoi.id
            for nguoi in self.danh_sach_nguoi
        ) + 1