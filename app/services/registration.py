# app/services/registration.py
# ================================================================
# ĐĂNG KÝ - NHẬN EMBEDDING TRỰC TIẾP (PHƯƠNG ÁN CŨ)
# ================================================================

import numpy as np
from app.database.models import NguoiDung
from app.config import settings
from app.utils.logger import logger


class DichVuDangKy:
    def __init__(self, embedder, database):
        self.embedder = embedder
        self.database = database
        self.so_embedding_toi_thieu = getattr(settings, 'SO_EMBEDDING_TOI_THIEU', 5)

    def tao_nguoi_dung(self, thong_tin, danh_sach_embedding):
        """
        Tạo người dùng từ danh sách EMBEDDING (phương án cũ)
        
        Args:
            thong_tin: Dict chứa thông tin người dùng
            danh_sach_embedding: List embedding vectors (512D)
            
        Returns:
            NguoiDung: Đối tượng người dùng đã tạo
        """
        if not danh_sach_embedding:
            raise ValueError("Không có embedding hợp lệ.")
        
        logger.info(f"[Registration] Bắt đầu đăng ký với {len(danh_sach_embedding)} embedding")
        
        # Lọc embedding chất lượng
        embeddings_filtered = []
        for emb in danh_sach_embedding:
            if emb is None:
                continue
            
            if np.any(np.isnan(emb)) or np.any(np.isinf(emb)):
                continue
            
            norm = np.linalg.norm(emb)
            if norm < 0.5 or norm > 1.5:
                continue
            
            embeddings_filtered.append(emb)
        
        if len(embeddings_filtered) < self.so_embedding_toi_thieu:
            raise ValueError(
                f"Không đủ embedding chất lượng. "
                f"Cần ít nhất {self.so_embedding_toi_thieu}, "
                f"có {len(embeddings_filtered)}"
            )
        
        # Tạo embedding đại diện
        embedding_dai_dien = self.embedder.tao_embedding_dai_dien(embeddings_filtered)
        
        if embedding_dai_dien is None:
            raise ValueError("Không tạo được embedding đại diện.")
        
        # Lưu vào database
        user_id = self.database.lay_id_tiep_theo()
        
        nguoi = NguoiDung(
            id=user_id,
            name=thong_tin["name"],
            age=thong_tin.get("age", ""),
            home=thong_tin.get("home", ""),
            class_name=thong_tin.get("class_name", thong_tin.get("class", "")),
            major=thong_tin.get("major", ""),
            embedding=embedding_dai_dien.tolist(),
            embedding_dimension=512,
            image_count=len(embeddings_filtered)
        )
        
        self.database.them_nguoi(nguoi)
        
        logger.info(f"[Registration] Đăng ký thành công cho {thong_tin['name']} (ID: {user_id})")
        logger.info(f"  - Số embedding: {len(embeddings_filtered)}")
        logger.info(f"  - Norm: {np.linalg.norm(embedding_dai_dien):.4f}")
        
        return nguoi