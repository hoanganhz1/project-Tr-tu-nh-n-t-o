# app/utils/camera_manager.py
# ================================================================
# QUẢN LÝ CAMERA - ĐÃ SỬA LỖI RELEASE + HỖ TRỢ GPU/60FPS
# ================================================================

import cv2
import numpy as np
from PyQt5.QtCore import QObject, QTimer, pyqtSignal, QMutex, QMutexLocker


class CameraManager(QObject):
    """Quản lý duy nhất một camera, phát broadcast frame"""

    frame_ready = pyqtSignal(np.ndarray)
    camera_error = pyqtSignal(str)

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        super().__init__()
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._read_frame)
        self.is_running = False
        self.paused = False
        self.camera_id = 0
        self.mutex = QMutex()
        self.fps = 30
        self._initialized = True

    def start(self, camera_id=0, fps=30):
        """Mở và bắt đầu đọc camera"""
        with QMutexLocker(self.mutex):
            self.fps = fps
            interval = int(1000 / self.fps)

            # Đã mở camera đúng id -> chỉ resume
            if self.is_running and self.cap is not None:
                if self.camera_id == camera_id:
                    self.paused = False
                    if not self.timer.isActive():
                        self.timer.start(interval)
                    return
                else:
                    self._stop_internal()

            # Đóng camera cũ nếu có (an toàn)
            self._safe_release()

            # Mở camera mới
            self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.camera_error.emit(f"Không thể mở camera {camera_id}")
                self.cap = None
                self.is_running = False
                return

            # Thiết lập thông số
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if fps >= 60:
                self.cap.set(cv2.CAP_PROP_FPS, 60)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.camera_id = camera_id
            self.is_running = True
            self.paused = False
            self.timer.start(interval)
            print(f"[CameraManager] Camera đã mở với FPS={self.fps}")

    def _safe_release(self):
        """Giải phóng camera an toàn, tránh lỗi"""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                print(f"[CameraManager] Lỗi khi release camera: {e}")
            self.cap = None

    def _stop_internal(self):
        """Dừng camera nội bộ (không lock)"""
        if self.timer.isActive():
            self.timer.stop()
        self._safe_release()
        self.is_running = False
        self.paused = False

    def stop(self):
        with QMutexLocker(self.mutex):
            self._stop_internal()

    def pause(self):
        with QMutexLocker(self.mutex):
            if self.is_running:
                self.paused = True
                if self.timer.isActive():
                    self.timer.stop()

    def resume(self):
        with QMutexLocker(self.mutex):
            if self.is_running and not self.timer.isActive():
                self.paused = False
                self.timer.start(int(1000 / self.fps))

    def set_fps(self, fps):
        with QMutexLocker(self.mutex):
            self.fps = fps
            if self.is_running and self.timer.isActive():
                self.timer.setInterval(int(1000 / self.fps))

    def is_opened(self):
        with QMutexLocker(self.mutex):
            return self.is_running and self.cap is not None and self.cap.isOpened()

    def _read_frame(self):
        """Đọc frame và phát signal"""
        with QMutexLocker(self.mutex):
            if self.paused or not self.is_running or self.cap is None:
                return
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    frame = cv2.flip(frame, 1)
                    self.frame_ready.emit(frame)
                else:
                    self.camera_error.emit("Mất kết nối camera, thử lại...")
                    # Lên lịch khởi động lại (không gọi trực tiếp vì đang giữ lock)
                    QTimer.singleShot(1000, self._restart_camera)
            except Exception as e:
                print(f"[CameraManager] Lỗi đọc frame: {e}")
                self.camera_error.emit(f"Lỗi camera: {e}")
                QTimer.singleShot(1000, self._restart_camera)

    def _restart_camera(self):
        """Khởi động lại camera (gọi từ luồng chính)"""
        with QMutexLocker(self.mutex):
            if self.is_running:
                cam_id = self.camera_id
                fps = self.fps
                self._stop_internal()
                self.start(cam_id, fps)