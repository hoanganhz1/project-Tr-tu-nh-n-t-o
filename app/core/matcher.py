# app/core/matcher.py
# ================================================================
# MATCHER - NHIỀU PHƯƠNG PHÁP SO SÁNH
# ================================================================

import numpy as np
from app.config import settings


class FaceMatcher:
    def __init__(self):
        # ✅ Phương pháp so sánh từ settings
        self.phuong_phap = getattr(settings, 'PHUONG_PHAP_SO_SANH', 'cosine')

    # ============================================================
    # COSINE DISTANCE
    # ============================================================

    def tinh_cosine_distance(self, embedding_1, embedding_2):
        """Cosine distance - Độ tương tự cosine"""
        vec1 = np.asarray(embedding_1, dtype=np.float32)
        vec2 = np.asarray(embedding_2, dtype=np.float32)
        
        # Xử lý khác chiều
        if len(vec1) != len(vec2):
            if len(vec1) > len(vec2):
                vec1 = vec1[:len(vec2)]
            else:
                vec2 = vec2[:len(vec1)]
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 2.0
        
        cosine_similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
        
        return float(1.0 - cosine_similarity)

    # ============================================================
    # ✅ EUCLIDEAN DISTANCE
    # ============================================================

    def tinh_euclidean_distance(self, embedding_1, embedding_2):
        """Khoảng cách Euclidean - Khoảng cách thực tế"""
        vec1 = np.asarray(embedding_1, dtype=np.float32)
        vec2 = np.asarray(embedding_2, dtype=np.float32)
        
        if len(vec1) != len(vec2):
            if len(vec1) > len(vec2):
                vec1 = vec1[:len(vec2)]
            else:
                vec2 = vec2[:len(vec1)]
        
        return float(np.linalg.norm(vec1 - vec2))

    # ============================================================
    # ✅ MANHATTAN DISTANCE
    # ============================================================

    def tinh_manhattan_distance(self, embedding_1, embedding_2):
        """Khoảng cách Manhattan - Tổng trị tuyệt đối"""
        vec1 = np.asarray(embedding_1, dtype=np.float32)
        vec2 = np.asarray(embedding_2, dtype=np.float32)
        
        if len(vec1) != len(vec2):
            if len(vec1) > len(vec2):
                vec1 = vec1[:len(vec2)]
            else:
                vec2 = vec2[:len(vec1)]
        
        return float(np.sum(np.abs(vec1 - vec2)))

    # ============================================================
    # ✅ SO SÁNH ĐA PHƯƠNG PHÁP
    # ============================================================

    def tinh_khoang_cach(self, embedding_1, embedding_2, phuong_phap=None):
        """
        Tính khoảng cách theo phương pháp được chọn
        
        Args:
            embedding_1, embedding_2: Hai vector
            phuong_phap: 'cosine', 'euclidean', 'manhattan'
                         Nếu None, dùng phương pháp mặc định
        """
        if phuong_phap is None:
            phuong_phap = self.phuong_phap
        
        if phuong_phap == "cosine":
            return self.tinh_cosine_distance(embedding_1, embedding_2)
        elif phuong_phap == "euclidean":
            return self.tinh_euclidean_distance(embedding_1, embedding_2)
        elif phuong_phap == "manhattan":
            return self.tinh_manhattan_distance(embedding_1, embedding_2)
        else:
            return self.tinh_cosine_distance(embedding_1, embedding_2)

    # ============================================================
    # HÀM HIỆN CÓ (GIỮ NGUYÊN)
    # ============================================================

    def tinh_do_giong(self, cosine_distance):
        similarity = 1.0 - cosine_distance
        return float(np.clip(similarity, -1.0, 1.0))

    def kiem_tra_nguong(self, cosine_distance, threshold):
        return cosine_distance <= threshold