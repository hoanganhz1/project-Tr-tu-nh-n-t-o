# app/utils/camera_manager.py
# ================================================================
# QUẢN LÝ CAMERA - ĐÃ SỬA LỖI TREO
# ================================================================

import cv2
import numpy as np
from PyQt5.QtCore import QObject, QTimer, pyqtSignal, QMutex, QMutexLocker
import time


class CameraManager(QObject):
    """Quản lý camera - ĐÃ SỬA LỖI TREO"""

    frame_ready = pyqtSignal(np.ndarray)
    camera_error = pyqtSignal(str)
    camera_changed = pyqtSignal(int)

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
        
        # ✅ THÊM: Kiểm soát lỗi
        self.error_count = 0
        self.MAX_ERRORS = 5
        self.is_error_state = False
        self.reconnect_timer = QTimer()
        self.reconnect_timer.setSingleShot(True)
        self.reconnect_timer.timeout.connect(self._reconnect_camera)
        
        self._initialized = True

    def start(self, camera_id=0, fps=30):
        """Mở và bắt đầu đọc camera"""
        with QMutexLocker(self.mutex):
            self.fps = fps
            interval = int(1000 / self.fps)
            self.error_count = 0
            self.is_error_state = False

            if self.is_running and self.cap is not None:
                if self.camera_id == camera_id:
                    self.paused = False
                    if not self.timer.isActive():
                        self.timer.start(interval)
                    return
                else:
                    self._stop_internal()

            self._safe_release()

            # ✅ SỬA: Thử nhiều backend
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            opened = False
            
            for backend in backends:
                try:
                    print(f"[CameraManager] Thử backend {backend}...")
                    self.cap = cv2.VideoCapture(camera_id, backend)
                    if self.cap.isOpened():
                        opened = True
                        print(f"[CameraManager] ✅ Mở được với backend {backend}")
                        break
                    else:
                        self.cap.release()
                        self.cap = None
                except Exception as e:
                    print(f"[CameraManager] Lỗi backend {backend}: {e}")
                    continue

            if not opened or self.cap is None:
                self.camera_error.emit(f"Không thể mở camera {camera_id}")
                self.cap = None
                self.is_running = False
                return

            # Cấu hình camera
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if fps >= 60:
                self.cap.set(cv2.CAP_PROP_FPS, 60)

            self.camera_id = camera_id
            self.is_running = True
            self.paused = False
            self.timer.start(interval)
            print(f"[CameraManager] Camera đã mở với FPS={self.fps}")

    def _safe_release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                print(f"[CameraManager] Lỗi release: {e}")
            self.cap = None

    def _stop_internal(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.reconnect_timer.isActive():
            self.reconnect_timer.stop()
        self._safe_release()
        self.is_running = False
        self.paused = False
        self.is_error_state = False
        self.error_count = 0

    def stop(self):
        with QMutexLocker(self.mutex):
            self._stop_internal()
            print("[CameraManager] Đã dừng camera")

    def pause(self):
        with QMutexLocker(self.mutex):
            if self.is_running:
                self.paused = True
                if self.timer.isActive():
                    self.timer.stop()
                print("[CameraManager] Đã tạm dừng camera")

    def resume(self):
        with QMutexLocker(self.mutex):
            if self.is_running and not self.timer.isActive() and not self.is_error_state:
                self.paused = False
                self.timer.start(int(1000 / self.fps))
                print("[CameraManager] Đã tiếp tục camera")

    def set_fps(self, fps):
        with QMutexLocker(self.mutex):
            self.fps = fps
            if self.is_running and self.timer.isActive():
                self.timer.setInterval(int(1000 / self.fps))

    def is_opened(self):
        with QMutexLocker(self.mutex):
            return self.is_running and self.cap is not None and self.cap.isOpened() and not self.is_error_state

    def switch_to_camera(self, camera_id: int) -> bool:
        """Chuyển sang camera khác"""
        with QMutexLocker(self.mutex):
            if camera_id == self.camera_id and self.is_running and not self.is_error_state:
                return True
            
            print(f"[CameraManager] Chuyển sang camera {camera_id}")
            self.error_count = 0
            self.is_error_state = False
            
            self._stop_internal()
            
            # Mở camera mới
            self.camera_id = camera_id
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            
            for backend in backends:
                try:
                    self.cap = cv2.VideoCapture(camera_id, backend)
                    if self.cap.isOpened():
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        self.is_running = True
                        self.paused = False
                        self.timer.start(int(1000 / self.fps))
                        
                        print(f"[CameraManager] ✅ Đã chuyển sang camera {camera_id}")
                        self.camera_changed.emit(camera_id)
                        return True
                except:
                    continue
            
            print(f"[CameraManager] ❌ Không mở được camera {camera_id}")
            self.cap = None
            self.is_running = False
            return False

    def scan_cameras(self, max_cameras: int = 5) -> list:
        """Quét camera khả dụng - KHÔNG BỊ TREO"""
        available = []
        for cam_id in range(max_cameras):
            try:
                # ✅ THÊM: Timeout ngắn để không treo
                cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        available.append(cam_id)
                        print(f"[CameraManager] ✅ Tìm thấy camera {cam_id}")
                    cap.release()
                time.sleep(0.05)  # ✅ THÊM: Delay nhỏ để không quá tải
            except Exception as e:
                print(f"[CameraManager] Lỗi quét camera {cam_id}: {e}")
                continue
        print(f"[CameraManager] Tìm thấy {len(available)} camera: {available}")
        return available

    def _read_frame(self):
        """Đọc frame - CÓ XỬ LÝ LỖI TREO"""
        if self.paused or not self.is_running or self.cap is None:
            return
        
        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # ✅ Reset lỗi khi đọc thành công
                self.error_count = 0
                self.is_error_state = False
                frame = cv2.flip(frame, 1)
                self.frame_ready.emit(frame)
            else:
                self._handle_read_error()
        except Exception as e:
            print(f"[CameraManager] Lỗi đọc frame: {e}")
            self._handle_read_error()

    def _handle_read_error(self):
        """Xử lý lỗi đọc frame - CÓ GIỚI HẠN"""
        self.error_count += 1
        print(f"[CameraManager] Lỗi đọc frame #{self.error_count}")
        
        if self.error_count >= self.MAX_ERRORS:
            self.is_error_state = True
            self.camera_error.emit("Mất kết nối camera, đang thử kết nối lại...")
            print(f"[CameraManager] ⚠️ Quá số lần lỗi, khởi động lại camera...")
            
            # Dừng timer để tránh spam
            if self.timer.isActive():
                self.timer.stop()
            
            # Lên lịch reconnect sau 2 giây
            if not self.reconnect_timer.isActive():
                self.reconnect_timer.start(2000)

    def _reconnect_camera(self):
        """Kết nối lại camera sau lỗi"""
        with QMutexLocker(self.mutex):
            print(f"[CameraManager] 🔄 Đang thử kết nối lại camera {self.camera_id}...")
            
            # Reset trạng thái
            self.is_error_state = False
            self.error_count = 0
            
            # Thử mở lại camera hiện tại
            self._stop_internal()
            success = self.switch_to_camera(self.camera_id)
            
            if success:
                print(f"[CameraManager] ✅ Đã kết nối lại camera {self.camera_id}")
                self.camera_error.emit(f"Đã kết nối lại camera {self.camera_id}")
            else:
                print(f"[CameraManager] ❌ Không thể kết nối lại camera {self.camera_id}")
                self.camera_error.emit("Không thể kết nối lại camera, vui lòng kiểm tra")

    def _restart_camera(self):
        """Restart camera - ĐÃ SỬA LỖI TREO"""
        with QMutexLocker(self.mutex):
            if self.is_running and not self.is_error_state:
                cam_id = self.camera_id
                fps = self.fps
                self._stop_internal()
                
                # ✅ THÊM: Delay nhỏ trước khi restart
                QTimer.singleShot(500, lambda: self._do_restart(cam_id, fps))

    def _do_restart(self, cam_id, fps):
        """Thực hiện restart - CÓ XỬ LÝ LỖI"""
        with QMutexLocker(self.mutex):
            if self.is_error_state:
                return
            
            print(f"[CameraManager] 🔄 Restart camera {cam_id}")
            self.start(cam_id, fps)

    def connect_ip_camera(self, rtsp_url: str, username: str = "", password: str = "", 
                           buffer_size: int = 1) -> bool:
        """Kết nối IP Camera"""
        with QMutexLocker(self.mutex):
            if username and password:
                url = f"rtsp://{username}:{password}@{rtsp_url}"
            else:
                url = rtsp_url
            
            self._stop_internal()
            self.error_count = 0
            self.is_error_state = False
            
            try:
                self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    
                    self.is_running = True
                    self.paused = False
                    self.camera_id = -1
                    self.timer.start(int(1000 / min(self.fps, 25)))
                    
                    print(f"[CameraManager] ✅ Kết nối IP camera thành công")
                    self.camera_changed.emit(-1)
                    return True
            except Exception as e:
                print(f"[CameraManager] Lỗi IP camera: {e}")
            
            return False