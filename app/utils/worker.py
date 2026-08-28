# app/utils/worker.py
# ================================================================
# WORKER - ĐẦY ĐỦ CÁC CLASS
# ================================================================

import traceback
import time
from PyQt5.QtCore import QRunnable, QObject, pyqtSignal, QThreadPool


class WorkerSignals(QObject):
    """Signal cho worker"""
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal()


class BaseWorker(QRunnable):
    """Worker cơ bản"""

    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            if not self._is_cancelled:
                self._run()
        except RuntimeError as e:
            if "wrapped C/C++ object" in str(e):
                pass
            else:
                traceback.print_exc()
                try:
                    self.signals.error.emit(str(e))
                except:
                    pass
        except Exception as e:
            traceback.print_exc()
            try:
                self.signals.error.emit(str(e))
            except:
                pass
        finally:
            try:
                self.signals.finished.emit()
            except:
                pass

    def _run(self):
        raise NotImplementedError


class TrichXuatWorker(BaseWorker):
    """Worker trích xuất embedding"""

    def __init__(self, embedder, anh_bgr, use_advanced=True):
        super().__init__()
        self.embedder = embedder
        self.anh_bgr = anh_bgr
        self.use_advanced = use_advanced

    def _run(self):
        if self._is_cancelled:
            return
        emb = self.embedder.trich_xuat(self.anh_bgr, use_advanced=self.use_advanced)
        if not self._is_cancelled:
            try:
                self.signals.result.emit(emb)
            except:
                pass


class NhanDangWorker(BaseWorker):
    """Worker nhận dạng 1:N - CHỈ 1 KHUÔN MẶT"""

    def __init__(self, face_api, anh_bgr, threshold):
        super().__init__()
        self.face_api = face_api
        self.anh_bgr = anh_bgr
        self.threshold = threshold

    def _run(self):
        if self._is_cancelled:
            return
        
        t0 = time.time()
        ket_qua = self.face_api.identify(self.anh_bgr, threshold=self.threshold)
        elapsed = (time.time() - t0) * 1000
        ket_qua["processing_time_ms"] = elapsed
        
        print(f"[TIMING] NhanDangWorker: {elapsed:.0f}ms")
        
        if not self._is_cancelled:
            try:
                self.signals.result.emit(ket_qua)
            except:
                pass


class NhanDangSingleWorker(NhanDangWorker):
    """Worker nhận dạng 1:N - Chỉ 1 khuôn mặt (alias)"""
    pass


class XacMinhWorker(BaseWorker):
    """Worker xác minh 1:1"""

    def __init__(self, face_api, anh_bgr, user_id, threshold, use_normalize=True):
        super().__init__()
        self.face_api = face_api
        self.anh_bgr = anh_bgr
        self.user_id = user_id
        self.threshold = threshold
        self.use_normalize = use_normalize

    def _run(self):
        if self._is_cancelled:
            return
        
        try:
            if self.use_normalize:
                ket_qua = self.face_api.verify_normalized(
                    self.anh_bgr, self.user_id,
                    threshold=self.threshold,
                    use_advanced=True
                )
            else:
                ket_qua = self.face_api.verify(
                    self.anh_bgr, self.user_id,
                    threshold=self.threshold
                )
            
            if not self._is_cancelled:
                try:
                    self.signals.result.emit(ket_qua)
                except:
                    pass
        except Exception as e:
            if not self._is_cancelled:
                try:
                    self.signals.error.emit(str(e))
                except:
                    pass


class TrichXuatBatchWorker(BaseWorker):
    """Worker trích xuất batch embedding"""

    def __init__(self, embedder, list_anh_bgr, use_advanced=True):
        super().__init__()
        self.embedder = embedder
        self.list_anh_bgr = list_anh_bgr
        self.use_advanced = use_advanced

    def _run(self):
        if self._is_cancelled:
            return
        embeddings = self.embedder.trich_xuat_batch(self.list_anh_bgr, use_advanced=self.use_advanced)
        if not self._is_cancelled:
            try:
                self.signals.result.emit(embeddings)
            except:
                pass


# ================================================================
# WORKER VẼ KHUNG
# ================================================================

class VeKhungRunnable(QRunnable):
    """Runnable cho việc vẽ khung lên ảnh"""

    def __init__(self, detector, anh_bgr, results, threshold):
        super().__init__()
        self.detector = detector
        self.anh_bgr = anh_bgr
        self.results = results
        self.threshold = threshold
        self.signals = WorkerSignals()
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            if self._is_cancelled or self.anh_bgr is None:
                try:
                    self.signals.result.emit((None, [], []))
                except:
                    pass
                return

            anh = self.anh_bgr.copy()
            boxes, _ = self.detector.phat_hien_tat_ca(anh)

            if boxes and not self._is_cancelled:
                thong_tin_list = []
                for i, box in enumerate(boxes):
                    info = {
                        "name": "Không xác định",
                        "distance": None,
                        "status": "fail"
                    }

                    if i < len(self.results):
                        item = self.results[i]
                        if isinstance(item, dict):
                            user = item.get("user")
                            if user:
                                if hasattr(user, 'to_dict'):
                                    user_dict = user.to_dict()
                                else:
                                    user_dict = user
                                info["name"] = user_dict.get("name", "Unknown")
                                info["distance"] = item.get("distance")
                                info["status"] = "success" if (
                                    info["distance"] is not None and
                                    info["distance"] <= self.threshold
                                ) else "fail"
                    thong_tin_list.append(info)

                anh = self.detector.ve_khung_cho_nhieu_face(anh, boxes, thong_tin_list)
                try:
                    self.signals.result.emit((anh, boxes, thong_tin_list))
                except:
                    pass
            else:
                try:
                    self.signals.result.emit((anh, [], []))
                except:
                    pass

        except Exception as e:
            traceback.print_exc()
            try:
                self.signals.error.emit(str(e))
            except:
                pass
        finally:
            try:
                self.signals.finished.emit()
            except:
                pass


class VeKhungXacMinhRunnable(QRunnable):
    """Runnable cho việc vẽ khung xác minh 1:1"""

    def __init__(self, detector, anh_bgr, ket_qua, threshold):
        super().__init__()
        self.detector = detector
        self.anh_bgr = anh_bgr
        self.ket_qua = ket_qua
        self.threshold = threshold
        self.signals = WorkerSignals()
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            if self._is_cancelled or self.anh_bgr is None:
                try:
                    self.signals.result.emit(None)
                except:
                    pass
                return

            anh = self.anh_bgr.copy()
            box, _ = self.detector.phat_hien(anh)

            if box is not None and not self._is_cancelled:
                best_match = self.ket_qua.get("best_match", {})
                user_info = best_match.get("user") if best_match else None
                distance = best_match.get("distance") if best_match else None

                ten = None
                if user_info:
                    if hasattr(user_info, 'to_dict'):
                        user_dict = user_info.to_dict()
                    else:
                        user_dict = user_info
                    ten = user_dict.get("name")

                anh = self.detector.ve_khung_va_thong_tin(
                    anh, box,
                    ten_nguoi=ten,
                    distance=distance,
                    threshold=self.threshold,
                    trang_thai="success" if self.ket_qua.get("success") else "fail"
                )

            try:
                self.signals.result.emit(anh)
            except:
                pass

        except Exception as e:
            traceback.print_exc()
            try:
                self.signals.error.emit(str(e))
            except:
                pass
        finally:
            try:
                self.signals.finished.emit()
            except:
                pass