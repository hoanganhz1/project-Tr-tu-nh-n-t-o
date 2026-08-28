# app/core/embedder.py
# ================================================================
# EMBEDDER - ĐÃ SỬA LỖI CHẤT LƯỢNG ẢNH
# ================================================================

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

from app.config import settings
from app.config.settings import THIET_BI, CHIEU_EMBEDDING, USE_ADVANCED_ALIGNMENT, USE_ENSEMBLE
from app.utils.logger import logger


class FaceEmbedder:
    def __init__(self, detector):
        self.detector = detector
        
        print(f"[MODEL] Đang tải FaceNet trên {THIET_BI}...")
        self.model = InceptionResnetV1(
            pretrained="vggface2"
        ).eval().to(THIET_BI)
        print("[MODEL] FaceNet đã sẵn sàng.")
        
        self.quality_stats = {
            "total": 0,
            "good": 0,
            "bad": 0
        }
        
        self.use_ensemble = USE_ENSEMBLE
        self.so_anh_ensemble = getattr(settings, 'SO_ANH_ENSEMBLE', 3)
        self.goc_xoay = getattr(settings, 'GOC_XOAY', [-5, 5])
        self.norm_min = getattr(settings, 'NORM_TOI_THIEU', 0.8)
        self.norm_max = getattr(settings, 'NORM_TOI_DA', 1.2)
        self.use_advanced = USE_ADVANCED_ALIGNMENT

    def chuan_hoa_anh(self, anh_bgr):
        """Chuẩn hóa ảnh - Áp dụng cho cả đăng ký và nhận dạng"""
        if anh_bgr is None:
            return None
        try:
            lab = cv2.cvtColor(anh_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l)
            lab_clahe = cv2.merge([l_clahe, a, b])
            result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        except Exception as e:
            logger.debug(f"[Embedder] Lỗi CLAHE: {e}")
            result = anh_bgr.copy()
        
        try:
            result = cv2.fastNlMeansDenoisingColored(result, None, 10, 10, 7, 21)
        except:
            pass
        
        try:
            alpha = getattr(settings, 'ALPHA_CONTRAST', 1.2)
            beta = getattr(settings, 'BETA_BRIGHTNESS', 10)
            result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
        except:
            pass
        
        try:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            result = cv2.filter2D(result, -1, kernel)
        except:
            pass
        
        return result

    def _kiem_tra_anh_hop_le(self, anh_bgr):
        """Kiểm tra ảnh có hợp lệ không - NỚI LỎNG ĐIỀU KIỆN"""
        if anh_bgr is None:
            return False
        if not isinstance(anh_bgr, np.ndarray):
            return False
        if len(anh_bgr.shape) != 3:
            return False
        # ✅ SỬA: Giảm ngưỡng kích thước tối thiểu
        if anh_bgr.shape[0] < 20 or anh_bgr.shape[1] < 20:
            return False
        return True

    def trich_xuat(self, anh_bgr, use_advanced=None):
        """
        Trích xuất embedding - CHUẨN HÓA ĐỒNG BỘ
        
        Args:
            anh_bgr: Ảnh BGR (đã được cắt ROI hoặc ảnh gốc)
            use_advanced: True = có chuẩn hóa nâng cao, False = không
        
        Returns:
            Embedding vector (512D) hoặc None nếu thất bại
        """
        # Kiểm tra ảnh hợp lệ
        if not self._kiem_tra_anh_hop_le(anh_bgr):
            logger.warning("[Embedder] ❌ Ảnh không hợp lệ")
            return None
        
        if use_advanced is None:
            use_advanced = self.use_advanced
        
        # Thử căn chỉnh nâng cao
        khuon_mat = None
        
        if use_advanced:
            try:
                # Chuẩn hóa ảnh trước
                anh_normalized = self.chuan_hoa_anh(anh_bgr)
                # Căn chỉnh nâng cao trên ảnh đã chuẩn hóa
                khuon_mat = self.detector.can_chinh_khuon_mat_nang_cao(anh_normalized)
            except Exception as e:
                logger.debug(f"[Embedder] Lỗi căn chỉnh nâng cao: {e}")
        
        # Nếu căn chỉnh nâng cao thất bại, thử căn chỉnh cơ bản
        if khuon_mat is None:
            try:
                khuon_mat = self.detector.can_chinh_khuon_mat(anh_bgr)
            except Exception as e:
                logger.debug(f"[Embedder] Lỗi căn chỉnh cơ bản: {e}")
        
        # ✅ SỬA: Nếu cả hai đều thất bại, thử xử lý thủ công
        if khuon_mat is None:
            logger.debug("[Embedder] Căn chỉnh thất bại, thử xử lý thủ công...")
            try:
                # Resize ảnh về 160x160 và chuyển sang tensor
                anh_resized = cv2.resize(anh_bgr, (160, 160))
                anh_rgb = cv2.cvtColor(anh_resized, cv2.COLOR_BGR2RGB)
                from PIL import Image
                anh_pil = Image.fromarray(anh_rgb)
                # Chuyển sang tensor và chuẩn hóa
                import torchvision.transforms as transforms
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                khuon_mat = transform(anh_pil).unsqueeze(0).to(THIET_BI)
                logger.debug("[Embedder] ✅ Xử lý thủ công thành công")
            except Exception as e:
                logger.warning(f"[Embedder] ❌ Xử lý thủ công thất bại: {e}")
                return None
        
        # ✅ SỬA: Nới lỏng kiểm tra chất lượng ảnh
        if khuon_mat is not None:
            # Chỉ kiểm tra kích thước cơ bản
            if torch.is_tensor(khuon_mat):
                if len(khuon_mat.shape) >= 3:
                    h, w = khuon_mat.shape[-2], khuon_mat.shape[-1]
                    if h < 20 or w < 20:
                        logger.warning("[Embedder] ❌ Khuôn mặt quá nhỏ")
                        return None
            else:
                try:
                    img = np.array(khuon_mat)
                    if img.shape[0] < 20 or img.shape[1] < 20:
                        logger.warning("[Embedder] ❌ Khuôn mặt quá nhỏ")
                        return None
                except:
                    pass
        
        # Trích xuất embedding
        try:
            # Đảm bảo khuon_mat đúng định dạng
            if not torch.is_tensor(khuon_mat):
                # Nếu là PIL Image, chuyển sang tensor
                from PIL import Image
                if isinstance(khuon_mat, Image.Image):
                    import torchvision.transforms as transforms
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])
                    khuon_mat = transform(khuon_mat).unsqueeze(0).to(THIET_BI)
                else:
                    logger.warning("[Embedder] Không xác định định dạng khuôn mặt")
                    return None
            
            # Đảm bảo đúng shape
            if len(khuon_mat.shape) == 3:
                khuon_mat = khuon_mat.unsqueeze(0)
            
            # Đưa về đúng device
            if khuon_mat.device != THIET_BI:
                khuon_mat = khuon_mat.to(THIET_BI)
            
            with torch.no_grad():
                embedding = self.model(khuon_mat)
            
            embedding = F.normalize(embedding, p=2, dim=1)
            embedding = embedding.squeeze(0).cpu().numpy().astype(np.float32)
            
            if embedding.shape != (CHIEU_EMBEDDING,):
                logger.warning(f"[Embedder] Sai chiều: {embedding.shape}")
                return None
            
            self.quality_stats["total"] += 1
            self.quality_stats["good"] += 1
            
            return embedding
            
        except Exception as e:
            logger.error(f"[Embedder] Lỗi trích xuất: {e}")
            return None

    def trich_xuat_nhan_dien(self, anh_bgr, use_advanced=None, use_ensemble=False):
        """
        Trích xuất embedding cho nhận dạng - ĐỒNG BỘ VỚI ĐĂNG KÝ
        """
        if use_advanced is None:
            use_advanced = self.use_advanced
        
        return self.trich_xuat(anh_bgr, use_advanced=use_advanced)

    # ============================================================
    # TRÍCH XUẤT BATCH - TỐI ƯU GPU
    # ============================================================

    def trich_xuat_batch(self, danh_sach_anh_bgr, use_advanced=True):
        """Trích xuất batch embedding - CÓ XỬ LÝ LỖI"""
        if not danh_sach_anh_bgr:
            return []

        aligned_faces = []
        for anh in danh_sach_anh_bgr:
            if not self._kiem_tra_anh_hop_le(anh):
                aligned_faces.append(None)
                continue
                
            if use_advanced:
                face = self.detector.can_chinh_khuon_mat_nang_cao(anh)
            else:
                face = self.detector.can_chinh_khuon_mat(anh)
            aligned_faces.append(face)

        valid_indices = [i for i, f in enumerate(aligned_faces) if f is not None]
        
        if not valid_indices:
            return [None] * len(danh_sach_anh_bgr)

        batch_tensors = [aligned_faces[i] for i in valid_indices]
        
        try:
            batch = torch.stack(batch_tensors).to(THIET_BI)

            with torch.no_grad():
                embeddings = self.model(batch)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            embeddings = embeddings.cpu().numpy().astype(np.float32)

            result = [None] * len(danh_sach_anh_bgr)
            for idx, emb in zip(valid_indices, embeddings):
                result[idx] = emb

            return result
        except Exception as e:
            logger.error(f"[Embedder] Lỗi batch: {e}")
            return [None] * len(danh_sach_anh_bgr)

    def trich_xuat_ensemble(self, anh_bgr, use_advanced=True):
        """Trích xuất ensemble embedding"""
        if not self._kiem_tra_anh_hop_le(anh_bgr):
            return None
            
        embeddings = []
        
        emb1 = self.trich_xuat(anh_bgr, use_advanced=use_advanced)
        if emb1 is not None:
            embeddings.append(emb1)
        
        if not self.use_ensemble or len(embeddings) == 0:
            return emb1
        
        h, w = anh_bgr.shape[:2]
        
        for angle in self.goc_xoay:
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(anh_bgr, M, (w, h))
            emb = self.trich_xuat(rotated, use_advanced=use_advanced)
            if emb is not None:
                embeddings.append(emb)
                if len(embeddings) >= self.so_anh_ensemble:
                    break
        
        if len(embeddings) < 3:
            flipped = cv2.flip(anh_bgr, 1)
            emb = self.trich_xuat(flipped, use_advanced=use_advanced)
            if emb is not None:
                embeddings.append(emb)
        
        if len(embeddings) < 3:
            for scale in [0.95, 1.05]:
                new_w = int(w * scale)
                new_h = int(h * scale)
                resized = cv2.resize(anh_bgr, (new_w, new_h))
                emb = self.trich_xuat(resized, use_advanced=use_advanced)
                if emb is not None:
                    embeddings.append(emb)
        
        if len(embeddings) < 2:
            return embeddings[0] if embeddings else None
        
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
        
        return final_embedding.astype(np.float32)

    # ✅ SỬA: Nới lỏng kiểm tra chất lượng ảnh
    def kiem_tra_chat_luong_anh(self, khuon_mat):
        """Kiểm tra chất lượng ảnh khuôn mặt - NỚI LỎNG"""
        if khuon_mat is None:
            return False
        
        try:
            if torch.is_tensor(khuon_mat):
                img = khuon_mat.cpu().numpy()
            else:
                img = np.array(khuon_mat)
            
            # ✅ SỬA: Giảm ngưỡng kích thước
            if img.shape[0] < 20 or img.shape[1] < 20:
                return False
            
            # ✅ SỬA: Giảm ngưỡng độ sáng
            if np.mean(img) < 5:
                return False
            
            return True
        except:
            return False

    def kiem_tra_chat_luong_embedding(self, embedding):
        """Kiểm tra chất lượng embedding"""
        if embedding is None:
            return False
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            return False
        
        norm = np.linalg.norm(embedding)
        if norm < self.norm_min or norm > self.norm_max:
            return False
        
        return True

    def tao_embedding_dai_dien(self, danh_sach_embedding):
        """Tạo embedding đại diện từ danh sách"""
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
        """Lấy thống kê chất lượng"""
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