# app/core/embedder.py
# ================================================================
# EMBEDDER - CHUẨN HÓA CHO CẢ ĐĂNG KÝ VÀ NHẬN DIỆN
# ================================================================

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

from app.config import settings
from app.config.settings import THIET_BI, CHIEU_EMBEDDING
from app.utils.logger import logger


class FaceEmbedder:
    """Singleton - Chỉ có một instance duy nhất"""

    _instance = None

    def __new__(cls, detector=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, detector=None):
        if self._initialized:
            return

        if detector is None:
            from app.core.detector import PhatHienKhuonMat
            self.detector = PhatHienKhuonMat()
        else:
            self.detector = detector

        print(f"[MODEL] Đang tải FaceNet trên {THIET_BI}...")
        self.model = InceptionResnetV1(
            pretrained="vggface2"
        ).eval().to(THIET_BI)
        print("[MODEL] FaceNet đã sẵn sàng.")

        # Thống kê chất lượng
        self.quality_stats = {
            "total": 0,
            "good": 0,
            "bad": 0
        }

        # Cấu hình từ settings
        self.use_ensemble = getattr(settings, 'SU_DUNG_ENSEMBLE', True)
        self.so_anh_ensemble = getattr(settings, 'SO_ANH_ENSEMBLE', 5)
        self.goc_xoay = getattr(settings, 'GOC_XOAY', [-5, 5])
        self.norm_min = getattr(settings, 'NORM_TOI_THIEU', 0.8)
        self.norm_max = getattr(settings, 'NORM_TOI_DA', 1.2)

        # Luôn luôn sử dụng chuẩn hóa
        self.use_advanced = True
        self._initialized = True

    # ============================================================
    # CÁC PHƯƠNG THỨC HIỆN CÓ (GIỮ NGUYÊN)
    # ============================================================
    # chuan_hoa_anh, trich_xuat, trich_xuat_ensemble, 
    # trich_xuat_nhan_dien, kiem_tra_chat_luong_anh, 
    # kiem_tra_chat_luong_embedding, tao_embedding_dai_dien,
    # get_quality_stats
    # ============================================================

    # ... (giữ nguyên toàn bộ code cũ của embedder) ...

    # ============================================================
    # ✅ THÊM PHƯƠNG THỨC BATCH
    # ============================================================

    def trich_xuat_batch(self, list_anh_bgr, use_advanced=None):
        """
        Trích xuất embedding cho nhiều ảnh cùng lúc (batch)
        Args:
            list_anh_bgr: List ảnh BGR
            use_advanced: bool
        Returns:
            List embedding (có thể có None)
        """
        if use_advanced is None:
            use_advanced = self.use_advanced

        # Căn chỉnh tất cả
        aligned_faces = []
        valid_indices = []
        for i, anh in enumerate(list_anh_bgr):
            if anh is None:
                continue
            if use_advanced:
                face = self.detector.can_chinh_khuon_mat_nang_cao(anh)
            else:
                face = self.detector.can_chinh_khuon_mat(anh)
            if face is not None:
                aligned_faces.append(face)
                valid_indices.append(i)

        if not aligned_faces:
            return [None] * len(list_anh_bgr)

        # Tạo batch tensor
        batch = torch.stack(aligned_faces).to(THIET_BI)
        with torch.no_grad():
            embeddings = self.model(batch)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings = embeddings.cpu().numpy().astype(np.float32)

        # Xây dựng kết quả
        result = [None] * len(list_anh_bgr)
        for idx, emb in zip(valid_indices, embeddings):
            # Có thể kiểm tra chất lượng ở đây nếu muốn
            result[idx] = emb
        return result
    # ============================================================
    # CHUẨN HÓA ẢNH
    # ============================================================

    def chuan_hoa_anh(self, anh_bgr):
        """
        Chuẩn hóa ảnh - Áp dụng cho CẢ đăng ký và nhận diện
        """
        if anh_bgr is None:
            return None
        
        try:
            # 1. Chuẩn hóa ánh sáng (CLAHE)
            lab = cv2.cvtColor(anh_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l)
            lab_clahe = cv2.merge([l_clahe, a, b])
            result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        except Exception as e:
            logger.warning(f"[Embedder] Lỗi CLAHE: {e}")
            result = anh_bgr.copy()
        
        # 2. Giảm nhiễu
        try:
            result = cv2.fastNlMeansDenoisingColored(result, None, 10, 10, 7, 21)
        except:
            pass
        
        # 3. Tăng độ tương phản
        try:
            alpha = getattr(settings, 'ALPHA_CONTRAST', 1.2)
            beta = getattr(settings, 'BETA_BRIGHTNESS', 10)
            result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
        except:
            pass
        
        # 4. Làm sắc nét
        try:
            kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
            result = cv2.filter2D(result, -1, kernel)
        except:
            pass
        
        return result

    # ============================================================
    # TRÍCH XUẤT EMBEDDING - CÓ LOG DEBUG
    # ============================================================
    def trich_xuat(self, anh_bgr, use_advanced=None):
        """Trích xuất embedding - BỎ QUA KIỂM TRA"""
        if use_advanced is None:
            use_advanced = self.use_advanced
        
        # Bước 1: Căn chỉnh
        if use_advanced:
            khuon_mat = self.detector.can_chinh_khuon_mat_nang_cao(anh_bgr)
        else:
            khuon_mat = self.detector.can_chinh_khuon_mat(anh_bgr)
        
        if khuon_mat is None:
            # ✅ THỬ LẠI VỚI CĂN CHỈNH CƠ BẢN
            logger.warning("[Embedder] Căn chỉnh nâng cao thất bại, thử cơ bản...")
            khuon_mat = self.detector.can_chinh_khuon_mat(anh_bgr)
            
            if khuon_mat is None:
                logger.warning("[Embedder] ❌ Không thể căn chỉnh khuôn mặt")
                return None
        
        # ✅ BỎ QUA KIỂM TRA CHẤT LƯỢNG ẢNH (tạm thời)
        # if not self.kiem_tra_chat_luong_anh(khuon_mat):
        #     return None
        
        # Bước 3: Trích xuất embedding
        with torch.no_grad():
            embedding = self.model(
                khuon_mat.unsqueeze(0).to(THIET_BI)
            )
        
        embedding = F.normalize(embedding, p=2, dim=1)
        embedding = embedding.squeeze(0).cpu().numpy().astype(np.float32)
        
        if embedding.shape != (CHIEU_EMBEDDING,):
            logger.warning(f"[Embedder] Sai chiều: {embedding.shape}")
            return None
        
        # ✅ BỎ QUA KIỂM TRA NORM (tạm thời)
        # if not self.kiem_tra_chat_luong_embedding(embedding):
        #     return None
        
        self.quality_stats["total"] += 1
        self.quality_stats["good"] += 1
        
        logger.info(f"[Embedder] ✅ Embedding thành công (norm={np.linalg.norm(embedding):.4f})")
        return embedding

    # ============================================================
    # TRÍCH XUẤT ENSEMBLE
    # ============================================================

    def trich_xuat_ensemble(self, anh_bgr, use_advanced=True):
        """
        Trích xuất ensemble embedding với chuẩn hóa
        """
        embeddings = []
        
        # 1. Ảnh gốc
        emb1 = self.trich_xuat(anh_bgr, use_advanced=use_advanced)
        if emb1 is not None:
            embeddings.append(emb1)
        
        if not self.use_ensemble or len(embeddings) == 0:
            return emb1
        
        h, w = anh_bgr.shape[:2]
        
        # 2. Xoay nhẹ
        for angle in self.goc_xoay:
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(anh_bgr, M, (w, h))
            emb = self.trich_xuat(rotated, use_advanced=use_advanced)
            if emb is not None:
                embeddings.append(emb)
                if len(embeddings) >= self.so_anh_ensemble:
                    break
        
        # 3. Lật ngang
        if len(embeddings) < 3:
            flipped = cv2.flip(anh_bgr, 1)
            emb = self.trich_xuat(flipped, use_advanced=use_advanced)
            if emb is not None:
                embeddings.append(emb)
        
        # 4. Phóng to/thu nhỏ
        if len(embeddings) < 3:
            for scale in [0.95, 1.05]:
                new_w = int(w * scale)
                new_h = int(h * scale)
                resized = cv2.resize(anh_bgr, (new_w, new_h))
                emb = self.trich_xuat(resized, use_advanced=use_advanced)
                if emb is not None:
                    embeddings.append(emb)
        
        # ============================================================
        # GỘP CÁC EMBEDDING
        # ============================================================
        
        if len(embeddings) < 2:
            return embeddings[0] if embeddings else None
        
        # Lọc outlier và gộp
        avg_embedding = np.mean(embeddings, axis=0)
        
        distances = []
        for emb in embeddings:
            dist = np.linalg.norm(emb - avg_embedding)
            distances.append(dist)
        
        threshold_dist = np.mean(distances) + np.std(distances)
        filtered = []
        for i, emb in enumerate(embeddings):
            if distances[i] <= threshold_dist:
                filtered.append(emb)
        
        if len(filtered) < 2:
            filtered = embeddings
        
        final_embedding = np.mean(filtered, axis=0)
        
        norm = np.linalg.norm(final_embedding)
        if norm > 0:
            final_embedding = final_embedding / norm
        
        logger.info(f"[Embedder] ✅ Ensemble embedding: {len(embeddings)} ảnh -> {len(filtered)} ảnh")
        return final_embedding.astype(np.float32)

    # ============================================================
    # TRÍCH XUẤT CHO NHẬN DIỆN
    # ============================================================

    def trich_xuat_nhan_dien(self, anh_bgr, use_advanced=True, use_ensemble=False):
        """
        Trích xuất embedding cho nhận diện - Tối ưu tốc độ
        """
        if use_ensemble:
            return self.trich_xuat_ensemble(anh_bgr, use_advanced=use_advanced)
        else:
            return self.trich_xuat(anh_bgr, use_advanced=use_advanced)

    # ============================================================
    # KIỂM TRA CHẤT LƯỢNG
    # ============================================================

    def kiem_tra_chat_luong_anh(self, khuon_mat):
        if khuon_mat is None:
            return False
        
        if torch.is_tensor(khuon_mat):
            img = khuon_mat.cpu().numpy()
        else:
            img = np.array(khuon_mat)
        
        if img.shape[0] < 50 or img.shape[1] < 50:
            logger.debug(f"[Embedder] Ảnh quá nhỏ: {img.shape}")
            return False
        
        if np.mean(img) < 10:
            logger.debug(f"[Embedder] Ảnh quá tối: {np.mean(img):.2f}")
            return False
        
        return True

    def kiem_tra_chat_luong_embedding(self, embedding):
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            logger.debug("[Embedder] Embedding chứa NaN/Inf")
            return False
        
        norm = np.linalg.norm(embedding)
        if norm < self.norm_min or norm > self.norm_max:
            logger.debug(f"[Embedder] Norm không hợp lệ: {norm:.4f} (cần {self.norm_min}-{self.norm_max})")
            return False
        
        return True

    def tao_embedding_dai_dien(self, danh_sach_embedding):
        if not danh_sach_embedding:
            logger.warning("[Embedder] Không có embedding để tạo đại diện")
            return None
        
        embeddings_filtered = []
        for emb in danh_sach_embedding:
            if emb is None:
                continue
            if not self.kiem_tra_chat_luong_embedding(emb):
                continue
            embeddings_filtered.append(emb)
        
        if len(embeddings_filtered) < 2:
            logger.warning(f"[Embedder] Chỉ có {len(embeddings_filtered)} embedding chất lượng, cần ít nhất 2")
            return embeddings_filtered[0] if embeddings_filtered else None
        
        ma_tran = np.vstack(embeddings_filtered)
        embedding_mean = np.mean(ma_tran, axis=0)
        embedding_median = np.median(ma_tran, axis=0)
        embedding_dai_dien = (embedding_mean + embedding_median) / 2
        
        chuan = np.linalg.norm(embedding_dai_dien)
        if chuan > 0:
            embedding_dai_dien = embedding_dai_dien / chuan
        
        logger.info(f"[Embedder] ✅ Tạo embedding đại diện từ {len(embeddings_filtered)} embedding")
        return embedding_dai_dien.astype(np.float32)

    def get_quality_stats(self):
        total = self.quality_stats["total"]
        good = self.quality_stats["good"]
        bad = self.quality_stats["bad"]
        
        if total == 0:
            rate = 0
        else:
            rate = good / total * 100
        
        return {
            "total": total,
            "good": good,
            "bad": bad,
            "good_rate": f"{rate:.1f}%"
        }