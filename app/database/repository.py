# app/database/repository.py
# ================================================================
# CƠ SỞ DỮ LIỆU - HỖ TRỢ CALLBACK KHI THAY ĐỔI
# ================================================================

import json
import os
import numpy as np

from app.config.settings import TEP_CSDL
from app.database.models import NguoiDung
from app.utils.logger import logger


class CoSoDuLieu:

    def __init__(self, tep_csdL=TEP_CSDL, on_change_callback=None):
        self.tep_csdL = tep_csdL
        self.danh_sach_nguoi = []
        self.on_change_callback = on_change_callback
        self.tai_du_lieu()

    def tai_du_lieu(self):
        if not os.path.exists(self.tep_csdL):
            self.danh_sach_nguoi = []
            return

        try:
            with open(self.tep_csdL, "r", encoding="utf-8") as tep:
                du_lieu = json.load(tep)

            if isinstance(du_lieu, dict):
                danh_sach = du_lieu.get("users", [])
            else:
                danh_sach = du_lieu

            self.danh_sach_nguoi = []
            for du_lieu_nguoi in danh_sach:
                try:
                    nguoi = NguoiDung.from_dict(du_lieu_nguoi)
                    embedding = np.asarray(nguoi.embedding, dtype=np.float32)
                    if embedding.shape != (512,):
                        print(f"[WARNING] ID {nguoi.id} không phải embedding 512D.")
                        continue
                    nguoi.embedding = embedding.tolist()
                    self.danh_sach_nguoi.append(nguoi)
                except Exception as loi:
                    print(f"[DATABASE] Bỏ qua bản ghi lỗi: {loi}")

        except Exception as loi:
            print(f"[DATABASE] Không thể đọc JSON: {loi}")
            self.danh_sach_nguoi = []

    # Trong method luu_du_lieu(), sửa thành:
    def luu_du_lieu(self):
        """Lưu dữ liệu với cơ chế backup"""
        du_lieu = {
            "version": 1,
            "model": {
                "name": "InceptionResnetV1",
                "pretrained": "vggface2",
                "embedding_dimension": 512,
                "distance": "cosine"
            },
            "users": [nguoi.to_dict() for nguoi in self.danh_sach_nguoi]
        }

        # ✅ SỬA: Ghi vào file tạm trước
        temp_file = self.tep_csdL + ".tmp"
        try:
            with open(temp_file, "w", encoding="utf-8") as tep:
                json.dump(du_lieu, tep, ensure_ascii=False, indent=2)
            
            # Sau khi ghi thành công, rename file
            if os.path.exists(self.tep_csdL):
                # Tạo backup
                backup_file = self.tep_csdL + ".backup"
                try:
                    os.replace(self.tep_csdL, backup_file)
                except:
                    pass
            
            os.replace(temp_file, self.tep_csdL)
            
        except Exception as e:
            logger.error(f"[Database] Lỗi lưu: {e}")
            # Xóa file tạm nếu có lỗi
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
            raise
        
        # Gọi callback khi có thay đổi
        self._notify_change()

    def _notify_change(self):
        """Thông báo thay đổi cho callback"""
        if self.on_change_callback:
            try:
                self.on_change_callback()
            except Exception as e:
                print(f"[DATABASE] Lỗi callback: {e}")

    def lay_tat_ca_nguoi(self):
        return self.danh_sach_nguoi.copy()

    def lay_nguoi_theo_id(self, user_id):
        for nguoi in self.danh_sach_nguoi:
            if int(nguoi.id) == int(user_id):
                return nguoi
        return None

    def them_nguoi(self, nguoi):
        self.danh_sach_nguoi.append(nguoi)
        self.luu_du_lieu()

    def xoa_nguoi(self, user_id):
        self.danh_sach_nguoi = [nguoi for nguoi in self.danh_sach_nguoi if int(nguoi.id) != int(user_id)]
        self.luu_du_lieu()

    def lay_id_tiep_theo(self):
        if not self.danh_sach_nguoi:
            return 1
        return max(nguoi.id for nguoi in self.danh_sach_nguoi) + 1