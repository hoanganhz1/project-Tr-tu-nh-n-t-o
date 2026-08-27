# app/core/recognizer.py
# ================================================================
# FACE RECOGNIZER - NHẬN DẠNG KHUÔN MẶT
# ================================================================

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from app.config import settings
from app.utils.logger import logger


class FaceRecognizer:
    """
    Bộ nhận dạng khuôn mặt
    Hỗ trợ: 1:N (Identify) và 1:1 (Verify)
    """

    def __init__(
        self,
        embedder,
        matcher,
        database
    ):
        self.embedder = embedder
        self.matcher = matcher
        self.database = database

        # Cache database để tăng tốc
        self._cache_users = None
        self._cache_timestamp = 0
        self._cache_ttl = 5  # Cache 5 giây

        # ✅ Ma trận embedding cho vector hóa
        self.embedding_matrix = None
        self.user_list = None
        self._build_cache()  # Xây dựng ngay

        # Thống kê
        self.stats = {
            "identify_calls": 0,
            "verify_calls": 0,
            "total_time_ms": 0,
            "avg_time_ms": 0
        }

        # Chọn phương pháp so sánh từ settings
        self.phuong_phap_so_sanh = getattr(
            settings,
            'PHUONG_PHAP_SO_SANH',
            'cosine'
        )

        logger.info(f"[Recognizer] Khởi tạo với phương pháp: {self.phuong_phap_so_sanh}")

    # ============================================================
    # ✅ CACHE MATRIX
    # ============================================================

    def _build_cache(self):
        """Xây dựng cache ma trận embedding"""
        users = self.database.lay_tat_ca_nguoi()
        if users:
            embeddings = []
            valid_users = []
            for u in users:
                emb = np.asarray(u.embedding, dtype=np.float32)
                if emb.shape == (512,):
                    embeddings.append(emb)
                    valid_users.append(u)
            if embeddings:
                self.embedding_matrix = np.vstack(embeddings)
                self.user_list = valid_users
            else:
                self.embedding_matrix = None
                self.user_list = []
        else:
            self.embedding_matrix = None
            self.user_list = []

    def refresh_cache(self):
        """Làm mới cache (gọi khi database thay đổi)"""
        self._build_cache()
        self._cache_users = None
        self._cache_timestamp = 0
        logger.info("[Recognizer] Đã làm mới cache matrix")

    # ============================================================
    # ✅ TÍNH TOÁN VECTOR HÓA
    # ============================================================

    def tinh_toan_tat_ca_distance_vectorized(self, query_emb: np.ndarray):
        """
        Tính distance với tất cả người dùng (vector hóa)
        Args:
            query_emb: Embedding cần so sánh (512D)
        Returns:
            List dict với user, distance, similarity
        """
        if self.embedding_matrix is None or len(self.user_list) == 0:
            return []

        # Chuẩn hóa query
        query = np.asarray(query_emb, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm == 0:
            return []
        query = query / norm

        # Tính cosine similarity
        similarities = np.dot(self.embedding_matrix, query)  # (N,)
        distances = 1.0 - similarities
        # Sắp xếp
        sorted_idx = np.argsort(distances)

        results = []
        for idx in sorted_idx:
            results.append({
                "user": self.user_list[idx],
                "distance": float(distances[idx]),
                "similarity": float(similarities[idx])
            })
        return results

    # ============================================================
    # CÁC PHƯƠNG THỨC HIỆN CÓ (SỬA ĐỂ DÙNG VECTOR HÓA)
    # ============================================================

    def _get_all_users(self):
        """Lấy tất cả người dùng với cache (giữ nguyên)"""
        import time as time_module
        current_time = time_module.time()
        if (self._cache_users is not None and
            current_time - self._cache_timestamp < self._cache_ttl):
            return self._cache_users
        self._cache_users = self.database.lay_tat_ca_nguoi()
        self._cache_timestamp = current_time
        return self._cache_users

    def _invalidate_cache(self):
        self._cache_users = None
        self._cache_timestamp = 0
        self.refresh_cache()

    def tinh_toan_tat_ca_distance(self, embedding_camera, phuong_phap=None):
        """
        Tính distance với tất cả - ưu tiên vector hóa
        """
        if phuong_phap is None:
            phuong_phap = self.phuong_phap_so_sanh

        # Chỉ hỗ trợ cosine cho vector hóa
        if phuong_phap == "cosine" and self.embedding_matrix is not None:
            return self.tinh_toan_tat_ca_distance_vectorized(embedding_camera)
        else:
            # Fallback cũ (nếu không dùng cosine hoặc matrix chưa có)
            all_users = self._get_all_users()
            if not all_users:
                return []
            ket_qua = []
            embedding_camera = np.asarray(embedding_camera, dtype=np.float32)
            for nguoi in all_users:
                user_embedding = np.asarray(nguoi.embedding, dtype=np.float32)
                if phuong_phap == "cosine":
                    distance = self.matcher.tinh_cosine_distance(embedding_camera, user_embedding)
                elif phuong_phap == "euclidean":
                    distance = self.matcher.tinh_euclidean_distance(embedding_camera, user_embedding)
                elif phuong_phap == "manhattan":
                    distance = self.matcher.tinh_manhattan_distance(embedding_camera, user_embedding)
                else:
                    distance = self.matcher.tinh_cosine_distance(embedding_camera, user_embedding)
                similarity = self.matcher.tinh_do_giong(distance)
                ket_qua.append({
                    "user": nguoi,
                    "distance": float(distance),
                    "similarity": float(similarity)
                })
            ket_qua.sort(key=lambda x: x["distance"])
            return ket_qua

    # ============================================================
    # 1:N - NHẬN DẠNG
    # ============================================================

    def identify(
        self,
        embedding_camera: np.ndarray,
        threshold: float,
        phuong_phap: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Nhận dạng 1:N

        Args:
            embedding_camera: Embedding cần nhận dạng
            threshold: Ngưỡng quyết định
            phuong_phap: Phương pháp so sánh

        Returns:
            Dict với success, message, best_match, results
        """
        start_time = time.time()

        # ✅ Thống kê
        self.stats["identify_calls"] += 1

        ket_qua = self.tinh_toan_tat_ca_distance(
            embedding_camera,
            phuong_phap
        )

        if not ket_qua:
            return {
                "success": False,
                "message": "CSDL trống.",
                "best_match": None,
                "results": [],
                "processing_time_ms": 0
            }

        tot_nhat = ket_qua[0]
        thanh_cong = tot_nhat["distance"] <= threshold

        # Tính thời gian
        elapsed_ms = (time.time() - start_time) * 1000

        # ✅ Cập nhật thống kê
        self.stats["total_time_ms"] += elapsed_ms
        self.stats["avg_time_ms"] = (
            self.stats["total_time_ms"] / self.stats["identify_calls"]
        )

        return {
            "success": thanh_cong,
            "message": "Nhận dạng thành công." if thanh_cong else "Người lạ.",
            "best_match": tot_nhat,
            "results": ket_qua,
            "processing_time_ms": round(elapsed_ms, 2)
        }

    # ============================================================
    # 1:1 - XÁC MINH
    # ============================================================

    def verify(
        self,
        embedding_camera: np.ndarray,
        user_id: Any,
        threshold: float,
        phuong_phap: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Xác minh 1:1

        Args:
            embedding_camera: Embedding cần xác minh
            user_id: ID của người cần xác minh
            threshold: Ngưỡng quyết định
            phuong_phap: Phương pháp so sánh

        Returns:
            Dict với success, message, best_match, results
        """
        start_time = time.time()

        # ✅ Thống kê
        self.stats["verify_calls"] += 1

        if phuong_phap is None:
            phuong_phap = self.phuong_phap_so_sanh

        # Lấy người dùng theo ID
        nguoi = self.database.lay_nguoi_theo_id(user_id)

        if nguoi is None:
            return {
                "success": False,
                "message": "ID không tồn tại.",
                "best_match": None,
                "results": [],
                "processing_time_ms": 0
            }

        # Lấy embedding của user
        if hasattr(nguoi, 'embedding'):
            user_embedding = np.asarray(nguoi.embedding, dtype=np.float32)
        else:
            return {
                "success": False,
                "message": "User không có embedding.",
                "best_match": None,
                "results": [],
                "processing_time_ms": 0
            }

        # Tính distance theo phương pháp
        if phuong_phap == "cosine":
            distance = self.matcher.tinh_cosine_distance(
                embedding_camera,
                user_embedding
            )
        elif phuong_phap == "euclidean":
            distance = self.matcher.tinh_euclidean_distance(
                embedding_camera,
                user_embedding
            )
        elif phuong_phap == "manhattan":
            distance = self.matcher.tinh_manhattan_distance(
                embedding_camera,
                user_embedding
            )
        else:
            distance = self.matcher.tinh_cosine_distance(
                embedding_camera,
                user_embedding
            )

        similarity = self.matcher.tinh_do_giong(distance)
        thanh_cong = distance <= threshold

        ket_qua = {
            "user": nguoi,
            "distance": float(distance),
            "similarity": float(similarity)
        }

        # Tính thời gian
        elapsed_ms = (time.time() - start_time) * 1000

        # ✅ Cập nhật thống kê
        self.stats["total_time_ms"] += elapsed_ms
        self.stats["avg_time_ms"] = (
            self.stats["total_time_ms"] / max(1, self.stats["verify_calls"])
        )

        return {
            "success": thanh_cong,
            "message": "Xác minh thành công." if thanh_cong else "Sai khuôn mặt.",
            "best_match": ket_qua,
            "results": [ket_qua],
            "processing_time_ms": round(elapsed_ms, 2)
        }

    # ============================================================
    # ✅ SO KHỚP NHIỀU KHUÔN MẶT
    # ============================================================

    def khop_nhieu_face(
        self,
        danh_sach_embedding: List[np.ndarray],
        threshold: float,
        phuong_phap: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        So khớp nhiều khuôn mặt với database

        Args:
            danh_sach_embedding: List các embedding (numpy array 512D)
            threshold: Ngưỡng quyết định
            phuong_phap: Phương pháp so sánh

        Returns:
            List các dict với thông tin từng khuôn mặt
        """
        if phuong_phap is None:
            phuong_phap = self.phuong_phap_so_sanh

        if not danh_sach_embedding:
            return []

        # Lấy tất cả người trong database (có cache)
        tat_ca_nguoi = self._get_all_users()

        if not tat_ca_nguoi:
            return [{
                "success": False,
                "best_match": None,
                "results": []
            } for _ in danh_sach_embedding]

        ket_qua = []

        for embedding in danh_sach_embedding:
            if embedding is None:
                ket_qua.append({
                    "success": False,
                    "best_match": None,
                    "results": []
                })
                continue

            # Tính distance với tất cả người
            danh_sach_distance = []

            for nguoi in tat_ca_nguoi:
                if hasattr(nguoi, 'embedding'):
                    user_embedding = np.asarray(nguoi.embedding, dtype=np.float32)
                else:
                    continue

                if phuong_phap == "cosine":
                    distance = self.matcher.tinh_cosine_distance(
                        embedding,
                        user_embedding
                    )
                elif phuong_phap == "euclidean":
                    distance = self.matcher.tinh_euclidean_distance(
                        embedding,
                        user_embedding
                    )
                elif phuong_phap == "manhattan":
                    distance = self.matcher.tinh_manhattan_distance(
                        embedding,
                        user_embedding
                    )
                else:
                    distance = self.matcher.tinh_cosine_distance(
                        embedding,
                        user_embedding
                    )

                similarity = self.matcher.tinh_do_giong(distance)

                danh_sach_distance.append({
                    "user": nguoi,
                    "distance": float(distance),
                    "similarity": float(similarity)
                })

            # Sắp xếp theo distance
            danh_sach_distance.sort(key=lambda x: x["distance"])

            # Lấy kết quả tốt nhất
            if danh_sach_distance:
                best = danh_sach_distance[0]
                thanh_cong = best["distance"] <= threshold

                ket_qua.append({
                    "success": thanh_cong,
                    "best_match": best,
                    "results": danh_sach_distance
                })
            else:
                ket_qua.append({
                    "success": False,
                    "best_match": None,
                    "results": []
                })

        return ket_qua

    # ============================================================
    # ✅ LẤY THỐNG KÊ
    # ============================================================

    def get_stats(self) -> Dict[str, Any]:
        """Lấy thống kê hiệu suất"""
        return {
            "identify_calls": self.stats["identify_calls"],
            "verify_calls": self.stats["verify_calls"],
            "total_time_ms": round(self.stats["total_time_ms"], 2),
            "avg_time_ms": round(self.stats["avg_time_ms"], 2),
            "phuong_phap": self.phuong_phap_so_sanh
        }

    def reset_stats(self):
        """Reset thống kê"""
        self.stats = {
            "identify_calls": 0,
            "verify_calls": 0,
            "total_time_ms": 0,
            "avg_time_ms": 0
        }

    # ============================================================
    # ✅ CẬP NHẬT DATABASE (GỌI KHI CÓ THAY ĐỔI)
    # ============================================================

    def refresh_database(self):
        """Làm mới cache database"""
        self._invalidate_cache()
        self._get_all_users()
        logger.info("[Recognizer] Đã làm mới cache database")