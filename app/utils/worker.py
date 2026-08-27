# app/utils/worker.py
# ================================================================
# QRUNNABLE CHO CÁC TÁC VỤ NẶNG (DÙNG QTHREADPOOL)
# ================================================================

import traceback
from PyQt5.QtCore import QRunnable, QObject, pyqtSignal, QThreadPool


class WorkerSignals(QObject):
    """Signal cho worker"""
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)  # tùy chọn
    finished = pyqtSignal()


class BaseWorker(QRunnable):
    """Worker cơ bản"""

    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()

    def run(self):
        try:
            self._run()
        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()

    def _run(self):
        """Ghi đè ở subclass"""
        raise NotImplementedError


class TrichXuatWorker(BaseWorker):
    """Worker trích xuất embedding"""

    def __init__(self, embedder, anh_bgr, use_advanced=True):
        super().__init__()
        self.embedder = embedder
        self.anh_bgr = anh_bgr
        self.use_advanced = use_advanced

    def _run(self):
        emb = self.embedder.trich_xuat(self.anh_bgr, use_advanced=self.use_advanced)
        self.signals.result.emit(emb)


class NhanDangWorker(BaseWorker):
    """Worker nhận dạng 1:N"""

    def __init__(self, face_api, anh_bgr, threshold):
        super().__init__()
        self.face_api = face_api
        self.anh_bgr = anh_bgr
        self.threshold = threshold

    def _run(self):
        import time
        start = time.time()
        ket_qua = self.face_api.identify(self.anh_bgr, threshold=self.threshold)
        ket_qua["processing_time_ms"] = (time.time() - start) * 1000
        self.signals.result.emit(ket_qua)


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
        self.signals.result.emit(ket_qua)


class TrichXuatBatchWorker(BaseWorker):
    """Worker trích xuất batch embedding"""

    def __init__(self, embedder, list_anh_bgr, use_advanced=True):
        super().__init__()
        self.embedder = embedder
        self.list_anh_bgr = list_anh_bgr
        self.use_advanced = use_advanced

    def _run(self):
        embeddings = self.embedder.trich_xuat_batch(self.list_anh_bgr, use_advanced=self.use_advanced)
        self.signals.result.emit(embeddings)