# app/services/registration.py
# ================================================================
# ĐĂNG KÝ - LỌC CHẤT LƯỢNG EMBEDDING
# ================================================================

import numpy as np
from app.database.models import NguoiDung
from app.config import settings


class DichVuDangKy:
    def __init__(self, embedder, database):
        self.embedder = embedder
        self.database = database
        
        # ✅ Số embedding tối thiểu từ settings
        self.so_embedding_toi_thieu = getattr(
            settings, 
            'SO_EMBEDDING_TOI_THIEU', 
            5
        )

    def tao_nguoi_dung(self, thong_tin, danh_sach_embedding):
        """
        Tạo người dùng với embedding đại diện
        Có lọc chất lượng embedding
        """
        if not danh_sach_embedding:
            raise ValueError("Không có embedding hợp lệ.")

        # ========================================================
        # ✅ LỌC EMBEDDING CHẤT LƯỢNG
        # ========================================================
        
        embeddings_filtered = []
        
        for emb in danh_sach_embedding:
            if emb is None:
                continue
            
            # Kiểm tra NaN/Inf
            if np.any(np.isnan(emb)) or np.any(np.isinf(emb)):
                continue
            
            # Kiểm tra norm (phải ≈ 1 sau khi normalize)
            norm = np.linalg.norm(emb)
            if norm < 0.5 or norm > 1.5:
                continue
            
            embeddings_filtered.append(emb)
        
        # Kiểm tra số lượng
        if len(embeddings_filtered) < self.so_embedding_toi_thieu:
            raise ValueError(
                f"Không đủ embedding chất lượng. "
                f"Cần ít nhất {self.so_embedding_toi_thieu}, "
                f"có {len(embeddings_filtered)}"
            )
        
        # ========================================================
        # ✅ TẠO EMBEDDING ĐẠI DIỆN
        # ========================================================
        
        # Sử dụng hàm tao_embedding_dai_dien của embedder
        embedding_dai_dien = self.embedder.tao_embedding_dai_dien(
            embeddings_filtered
        )
        
        if embedding_dai_dien is None:
            raise ValueError("Không tạo được embedding đại diện.")
        
        # ========================================================
        # ✅ LƯU VÀO DATABASE
        # ========================================================
        
        user_id = self.database.lay_id_tiep_theo()
        
        nguoi = NguoiDung(
            id=user_id,
            name=thong_tin["name"],
            age=thong_tin.get("age", ""),
            home=thong_tin.get("home", ""),
            class_name=thong_tin.get("class_name", ""),
            major=thong_tin.get("major", ""),
            embedding=embedding_dai_dien.tolist(),
            embedding_dimension=512,
            image_count=len(embeddings_filtered)  # ✅ Lưu số ảnh chất lượng
        )
        
        self.database.them_nguoi(nguoi)
        
        return nguoi