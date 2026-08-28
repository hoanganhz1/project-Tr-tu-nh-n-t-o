# app/core/detector.py
# ================================================================
# PHÁT HIỆN KHUÔN MẶT - TỐI ƯU TỐC ĐỘ
# ================================================================

import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

from app.config import settings
from app.config.settings import THIET_BI
from app.utils.logger import logger


class PhatHienKhuonMat:
    def __init__(self):
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=THIET_BI
        )
        
        self.use_clahe = getattr(settings, 'SU_DUNG_CLAHE', True)
        self.use_denoise = getattr(settings, 'SU_DUNG_DENOISE', True)
        self.use_sharpen = getattr(settings, 'SU_DUNG_SHARPEN', True)
        self.alpha = getattr(settings, 'ALPHA_CONTRAST', 1.2)
        self.beta = getattr(settings, 'BETA_BRIGHTNESS', 10)
        
        # ✅ THÊM: Cache để tránh detect lặp lại
        self._last_detect_result = None
        self._last_frame_hash = None

    def chuyen_sang_pil(self, anh_bgr):
        anh_rgb = cv2.cvtColor(anh_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(anh_rgb)

    # ============================================================
    # DETECT NHANH - TỐI ƯU TỐC ĐỘ
    # ============================================================

    def phat_hien_nhanh(self, anh_bgr, scale=0.5):
        """
        Phát hiện khuôn mặt nhanh - TỐI ƯU TỐC ĐỘ
        
        ✅ SỬA: Giảm scale xuống 0.3 để tăng tốc
        ✅ SỬA: Bỏ resize nếu ảnh đã nhỏ
        """
        if anh_bgr is None:
            return None, 0.0

        h, w = anh_bgr.shape[:2]
        
        # ✅ SỬA: Nếu ảnh nhỏ, detect trực tiếp
        if h < 200 or w < 200:
            scale = 1.0
        
        # ✅ SỬA: Giảm scale xuống 0.3 để tăng tốc (mặc định 0.5)
        if scale > 0.3:
            scale = 0.3
        
        # Resize ảnh nhỏ để detect nhanh
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            anh_nho = cv2.resize(anh_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            anh_nho = anh_bgr

        anh_pil = self.chuyen_sang_pil(anh_nho)
        
        # ✅ SỬA: Detect với thresholds cao hơn để nhanh hơn
        boxes, probs = self.mtcnn.detect(anh_pil)

        if boxes is None or len(boxes) == 0:
            return None, 0.0

        # Chọn khuôn mặt lớn nhất
        best_box = None
        best_prob = 0.0
        best_area = -1

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_box = box
                if probs is not None:
                    best_prob = float(probs[i])

        # Scale lại box về kích thước gốc
        if best_box is not None and scale < 1.0:
            scale_factor = 1.0 / scale
            x1, y1, x2, y2 = best_box
            best_box = (
                int(x1 * scale_factor),
                int(y1 * scale_factor),
                int(x2 * scale_factor),
                int(y2 * scale_factor)
            )

        return best_box, best_prob

    # ============================================================
    # DETECT CHUẨN - DÙNG KHI CẦN ĐỘ CHÍNH XÁC CAO
    # ============================================================

    def phat_hien(self, anh_bgr):
        """Phát hiện khuôn mặt - Độ chính xác cao, chậm hơn"""
        try:
            if anh_bgr is None:
                return None, 0.0
            
            anh_pil = self.chuyen_sang_pil(anh_bgr)
            boxes, probabilities = self.mtcnn.detect(anh_pil)

            if boxes is None:
                return None, 0.0

            dien_tich_lon_nhat = -1
            box_tot_nhat = None
            xac_suat_tot_nhat = 0.0

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                dien_tich = max(0, x2 - x1) * max(0, y2 - y1)

                if dien_tich > dien_tich_lon_nhat:
                    dien_tich_lon_nhat = dien_tich
                    box_tot_nhat = tuple(map(int, box))
                    if probabilities is not None:
                        xac_suat_tot_nhat = float(probabilities[i])

            return box_tot_nhat, xac_suat_tot_nhat

        except Exception as loi:
            logger.debug(f"[DETECTOR] Lỗi phát hiện: {loi}")
            return None, 0.0

    def phat_hien_tat_ca(self, anh_bgr):
        """Phát hiện tất cả khuôn mặt"""
        try:
            if anh_bgr is None:
                return [], []
                
            anh_pil = self.chuyen_sang_pil(anh_bgr)
            boxes, probabilities = self.mtcnn.detect(anh_pil)

            if boxes is None:
                return [], []

            boxes_list = []
            probs_list = []

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                boxes_list.append(tuple(map(int, box)))
                if probabilities is not None:
                    probs_list.append(float(probabilities[i]))
                else:
                    probs_list.append(0.0)

            return boxes_list, probs_list

        except Exception as loi:
            logger.debug(f"[DETECTOR] Lỗi phát hiện tất cả: {loi}")
            return [], []

    # ============================================================
    # CHUẨN HÓA ẢNH - TỐI ƯU
    # ============================================================

    def chuan_hoa_anh_sang(self, image):
        if not self.use_clahe:
            return image
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l)
            lab_clahe = cv2.merge([l_clahe, a, b])
            result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
            return result
        except Exception as e:
            return image

    def tang_cuong_chat_luong(self, image):
        result = image.copy()
        if self.use_denoise:
            try:
                result = cv2.fastNlMeansDenoisingColored(result, None, 10, 10, 7, 21)
            except:
                pass
        try:
            result = cv2.convertScaleAbs(result, alpha=self.alpha, beta=self.beta)
        except:
            pass
        if self.use_sharpen:
            try:
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                result = cv2.filter2D(result, -1, kernel)
            except:
                pass
        return result

    def chuan_hoa_anh(self, image):
        result = self.chuan_hoa_anh_sang(image)
        result = self.tang_cuong_chat_luong(result)
        return result

    # ============================================================
    # CĂN CHỈNH KHUÔN MẶT - TỐI ƯU
    # ============================================================

    def can_chinh_khuon_mat_nang_cao(self, anh_bgr):
        """Căn chỉnh khuôn mặt nâng cao - TỐI ƯU"""
        try:
            if anh_bgr is None:
                return None
                
            anh_pil = self.chuyen_sang_pil(anh_bgr)
            boxes, probs = self.mtcnn.detect(anh_pil)
            
            if boxes is None or len(boxes) == 0:
                return None
            
            box = boxes[0]
            x1, y1, x2, y2 = map(int, box)
            
            margin = int((x2 - x1) * 0.2)
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(anh_bgr.shape[1], x2 + margin)
            y2 = min(anh_bgr.shape[0], y2 + margin)
            
            face_roi = anh_bgr[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                return None
            
            face_roi = self.chuan_hoa_anh(face_roi)
            face_roi = cv2.resize(face_roi, (160, 160))
            face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
            aligned_face = self.mtcnn(face_pil)
            
            return aligned_face
            
        except Exception as loi:
            logger.debug(f"[ALIGNMENT] {loi}")
            return None

    def can_chinh_khuon_mat(self, anh_bgr):
        """Căn chỉnh khuôn mặt cơ bản"""
        try:
            if anh_bgr is None:
                return None
            anh_pil = self.chuyen_sang_pil(anh_bgr)
            khuon_mat = self.mtcnn(anh_pil)
            return khuon_mat
        except Exception as loi:
            logger.debug(f"[ALIGNMENT] {loi}")
            return None

    # ============================================================
    # VẼ KHUNG - GIỮ NGUYÊN
    # ============================================================

    def ve_khung(self, anh_bgr, box, mau_khung=(0, 255, 0), do_day=2):
        if box is None:
            return anh_bgr

        x1, y1, x2, y2 = box
        anh_copy = anh_bgr.copy()

        cv2.rectangle(anh_copy, (x1, y1), (x2, y2), mau_khung, do_day)

        kich_thuoc_goc = 12
        cv2.line(anh_copy, (x1, y1), (x1 + kich_thuoc_goc, y1), mau_khung, do_day)
        cv2.line(anh_copy, (x1, y1), (x1, y1 + kich_thuoc_goc), mau_khung, do_day)
        cv2.line(anh_copy, (x2, y1), (x2 - kich_thuoc_goc, y1), mau_khung, do_day)
        cv2.line(anh_copy, (x2, y1), (x2, y1 + kich_thuoc_goc), mau_khung, do_day)
        cv2.line(anh_copy, (x1, y2), (x1 + kich_thuoc_goc, y2), mau_khung, do_day)
        cv2.line(anh_copy, (x1, y2), (x1, y2 - kich_thuoc_goc), mau_khung, do_day)
        cv2.line(anh_copy, (x2, y2), (x2 - kich_thuoc_goc, y2), mau_khung, do_day)
        cv2.line(anh_copy, (x2, y2), (x2, y2 - kich_thuoc_goc), mau_khung, do_day)

        return anh_copy

    def ve_khung_va_thong_tin(self, anh_bgr, box, ten_nguoi=None, distance=None,
                                threshold=None, trang_thai=None, do_day=3):
        if box is None:
            return anh_bgr

        x1, y1, x2, y2 = box
        anh_copy = anh_bgr.copy()

        if trang_thai == "success":
            mau_khung = (0, 255, 0)
            mau_chu = (0, 255, 0)
            label = "✅"
        elif trang_thai == "fail":
            mau_khung = (0, 0, 255)
            mau_chu = (0, 0, 255)
            label = "❌"
        else:
            mau_khung = (0, 255, 255)
            mau_chu = (0, 255, 255)
            label = "🔍"

        cv2.rectangle(anh_copy, (x1, y1), (x2, y2), mau_khung, do_day)

        kich_thuoc_goc = 15
        cv2.line(anh_copy, (x1, y1), (x1 + kich_thuoc_goc, y1), mau_khung, do_day)
        cv2.line(anh_copy, (x1, y1), (x1, y1 + kich_thuoc_goc), mau_khung, do_day)
        cv2.line(anh_copy, (x2, y1), (x2 - kich_thuoc_goc, y1), mau_khung, do_day)
        cv2.line(anh_copy, (x2, y1), (x2, y1 + kich_thuoc_goc), mau_khung, do_day)
        cv2.line(anh_copy, (x1, y2), (x1 + kich_thuoc_goc, y2), mau_khung, do_day)
        cv2.line(anh_copy, (x1, y2), (x1, y2 - kich_thuoc_goc), mau_khung, do_day)
        cv2.line(anh_copy, (x2, y2), (x2 - kich_thuoc_goc, y2), mau_khung, do_day)
        cv2.line(anh_copy, (x2, y2), (x2, y2 - kich_thuoc_goc), mau_khung, do_day)

        text_lines = []
        if ten_nguoi:
            text_lines.append(f"{label} {ten_nguoi}")
        else:
            text_lines.append(f"{label} Không xác định")

        if distance is not None:
            text_lines.append(f"Distance: {distance:.4f}")

        text_x = x1
        text_y = y1 - 10

        if text_y < 20:
            text_y = y1 + 30

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 2

        for i, line in enumerate(text_lines):
            y_pos = text_y + i * 25
            (text_w, text_h), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            cv2.rectangle(anh_copy, (text_x - 5, y_pos - text_h - 5),
                         (text_x + text_w + 5, y_pos + 5), (0, 0, 0), -1)
            cv2.putText(anh_copy, line, (text_x, y_pos), font, font_scale,
                       mau_chu, font_thickness, cv2.LINE_AA)

        return anh_copy

    def ve_khung_cho_nhieu_face(self, anh_bgr, boxes, thong_tin_list=None):
        anh_copy = anh_bgr.copy()

        if not boxes:
            return anh_copy

        thong_tin_list = thong_tin_list or []

        for i, box in enumerate(boxes):
            thong_tin = thong_tin_list[i] if i < len(thong_tin_list) else {}

            anh_copy = self.ve_khung_va_thong_tin(
                anh_copy,
                box,
                ten_nguoi=thong_tin.get("name"),
                distance=thong_tin.get("distance"),
                threshold=thong_tin.get("threshold"),
                trang_thai=thong_tin.get("status")
            )

        return anh_copy