# app/core/recognizer.py
# ================================================================
# FACE RECOGNIZER - VECTORIZED MATCHING + AUTO REFRESH CACHE
# ================================================================

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from app.config import settings
from app.utils.logger import logger


class FaceRecognizer:
    def __init__(self, embedder, matcher, database):
        self.embedder = embedder
        self.matcher = matcher
        self.database = database

        # Cache database với TTL ngắn để tự động refresh
        self._cache_users = None
        self._cache_timestamp = 0
        self._cache_ttl = 1  # 1 giây - tự động refresh liên tục

        # Vectorized matching cache
        self.embedding_matrix = None
        self.user_list = []
        self._build_cache()

        self.stats = {
            "identify_calls": 0,
            "verify_calls": 0,
            "total_time_ms": 0,
            "avg_time_ms": 0
        }

        self.phuong_phap_so_sanh = getattr(settings, 'PHUONG_PHAP_SO_SANH', 'cosine')

        logger.info(f"[Recognizer] Khởi tạo với phương pháp: {self.phuong_phap_so_sanh}")

    # ============================================================
    # CACHE VECTORIZED - TỰ ĐỘNG REFRESH
    # ============================================================

    def _build_cache(self):
        """Xây dựng ma trận embedding để so khớp nhanh"""
        users = self.database.lay_tat_ca_nguoi()
        if users:
            embeddings = []
            valid_users = []
            for u in users:
                emb = u.embedding if hasattr(u, 'embedding') else u.get('embedding')
                if emb is not None:
                    try:
                        embeddings.append(np.array(emb, dtype=np.float32))
                        valid_users.append(u)
                    except:
                        continue
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
        """Làm mới cache khi database thay đổi"""
        self._build_cache()
        self._cache_users = None
        self._cache_timestamp = 0
        logger.info("[Recognizer] Đã làm mới cache")

    def _get_all_users(self):
        """Lấy danh sách người dùng với cache TTL - TỰ ĐỘNG REFRESH"""
        current_time = time.time()
        # Nếu cache hết hạn hoặc chưa có, load lại
        if (self._cache_users is None or
            current_time - self._cache_timestamp > self._cache_ttl):
            self._cache_users = self.database.lay_tat_ca_nguoi()
            self._cache_timestamp = current_time
            # Đồng thời cập nhật vectorized cache
            self._build_cache()
            logger.debug(f"[Recognizer] Cache refreshed: {len(self._cache_users)} users")
        return self._cache_users

    def _invalidate_cache(self):
        """Xóa cache buộc load lại lần sau"""
        self._cache_users = None
        self._cache_timestamp = 0
        self.embedding_matrix = None
        self.user_list = []

    # ============================================================
    # VECTORIZED MATCHING
    # ============================================================

    def tinh_toan_tat_ca_distance_vectorized(self, query_emb):
        """So khớp vector hóa - Nhanh hơn nhiều (< 1ms)"""
        # Kiểm tra cache và refresh nếu cần
        self._get_all_users()  # Tự động refresh nếu hết hạn
        
        if self.embedding_matrix is None or not self.user_list:
            return []

        if not isinstance(query_emb, np.ndarray):
            query_emb = np.array(query_emb, dtype=np.float32)

        # Cosine similarity vector hóa
        similarities = np.dot(self.embedding_matrix, query_emb)
        distances = 1 - similarities

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

    def tinh_toan_tat_ca_distance(self, embedding_camera: np.ndarray, phuong_phap: Optional[str] = None):
        """Fallback: tính tuần tự nếu chưa có cache"""
        if phuong_phap is None:
            phuong_phap = self.phuong_phap_so_sanh

        all_users = self._get_all_users()
        if not all_users:
            return []

        if not isinstance(embedding_camera, np.ndarray):
            embedding_camera = np.asarray(embedding_camera, dtype=np.float32)

        ket_qua = []
        for nguoi in all_users:
            if hasattr(nguoi, 'embedding'):
                user_embedding = np.asarray(nguoi.embedding, dtype=np.float32)
            else:
                continue

            distance = self.matcher.tinh_khoang_cach(embedding_camera, user_embedding, phuong_phap)
            similarity = self.matcher.tinh_do_giong(distance)

            ket_qua.append({
                "user": nguoi,
                "distance": float(distance),
                "similarity": float(similarity)
            })

        ket_qua.sort(key=lambda x: x["distance"])
        return ket_qua

    # ============================================================
    # NHẬN DẠNG 1:N
    # ============================================================

    def identify(self, embedding_camera: np.ndarray, threshold: float, phuong_phap: Optional[str] = None):
        start_time = time.time()
        self.stats["identify_calls"] += 1

        # Refresh cache trước khi nhận dạng
        self._get_all_users()

        # Luôn dùng vectorized nếu có cache
        if self.embedding_matrix is not None and len(self.user_list) > 0:
            ket_qua = self.tinh_toan_tat_ca_distance_vectorized(embedding_camera)
        else:
            ket_qua = self.tinh_toan_tat_ca_distance(embedding_camera, phuong_phap)

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

        elapsed_ms = (time.time() - start_time) * 1000
        self.stats["total_time_ms"] += elapsed_ms
        self.stats["avg_time_ms"] = self.stats["total_time_ms"] / self.stats["identify_calls"]

        return {
            "success": thanh_cong,
            "message": "Nhận dạng thành công." if thanh_cong else "Người lạ.",
            "best_match": tot_nhat,
            "results": ket_qua,
            "processing_time_ms": round(elapsed_ms, 2)
        }

    # ============================================================
    # XÁC MINH 1:1
    # ============================================================

    def verify(self, embedding_camera: np.ndarray, user_id: Any, threshold: float, phuong_phap: Optional[str] = None):
        start_time = time.time()
        self.stats["verify_calls"] += 1

        if phuong_phap is None:
            phuong_phap = self.phuong_phap_so_sanh

        # Load trực tiếp từ database (không cache) để đảm bảo dữ liệu mới nhất
        nguoi = self.database.lay_nguoi_theo_id(user_id)
        if nguoi is None:
            return {
                "success": False,
                "message": "ID không tồn tại.",
                "best_match": None,
                "results": [],
                "processing_time_ms": 0
            }

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

        distance = self.matcher.tinh_khoang_cach(embedding_camera, user_embedding, phuong_phap)
        similarity = self.matcher.tinh_do_giong(distance)
        thanh_cong = distance <= threshold

        ket_qua = {
            "user": nguoi,
            "distance": float(distance),
            "similarity": float(similarity)
        }

        elapsed_ms = (time.time() - start_time) * 1000
        self.stats["total_time_ms"] += elapsed_ms
        self.stats["avg_time_ms"] = self.stats["total_time_ms"] / max(1, self.stats["verify_calls"])

        return {
            "success": thanh_cong,
            "message": "Xác minh thành công." if thanh_cong else "Sai khuôn mặt.",
            "best_match": ket_qua,
            "results": [ket_qua],
            "processing_time_ms": round(elapsed_ms, 2)
        }

    def khop_nhieu_face(self, danh_sach_embedding: List[np.ndarray], threshold: float, phuong_phap: Optional[str] = None):
        if phuong_phap is None:
            phuong_phap = self.phuong_phap_so_sanh

        if not danh_sach_embedding:
            return []

        # Refresh cache trước khi khớp
        tat_ca_nguoi = self._get_all_users()
        if not tat_ca_nguoi:
            return [{"success": False, "best_match": None, "results": []} for _ in danh_sach_embedding]

        ket_qua = []
        for embedding in danh_sach_embedding:
            if embedding is None:
                ket_qua.append({"success": False, "best_match": None, "results": []})
                continue

            danh_sach_distance = []
            for nguoi in tat_ca_nguoi:
                if hasattr(nguoi, 'embedding'):
                    user_embedding = np.asarray(nguoi.embedding, dtype=np.float32)
                else:
                    continue

                distance = self.matcher.tinh_khoang_cach(embedding, user_embedding, phuong_phap)
                similarity = self.matcher.tinh_do_giong(distance)

                danh_sach_distance.append({
                    "user": nguoi,
                    "distance": float(distance),
                    "similarity": float(similarity)
                })

            danh_sach_distance.sort(key=lambda x: x["distance"])
            if danh_sach_distance:
                best = danh_sach_distance[0]
                thanh_cong = best["distance"] <= threshold
                ket_qua.append({
                    "success": thanh_cong,
                    "best_match": best,
                    "results": danh_sach_distance
                })
            else:
                ket_qua.append({"success": False, "best_match": None, "results": []})

        return ket_qua

    def get_stats(self) -> Dict[str, Any]:
        return {
            "identify_calls": self.stats["identify_calls"],
            "verify_calls": self.stats["verify_calls"],
            "total_time_ms": round(self.stats["total_time_ms"], 2),
            "avg_time_ms": round(self.stats["avg_time_ms"], 2),
            "phuong_phap": self.phuong_phap_so_sanh
        }

    def reset_stats(self):
        self.stats = {
            "identify_calls": 0,
            "verify_calls": 0,
            "total_time_ms": 0,
            "avg_time_ms": 0
        }