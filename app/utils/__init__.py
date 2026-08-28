# app/utils/__init__.py
# ================================================================
# UTILS PACKAGE
# ================================================================

from app.utils.logger import logger
from app.utils.camera_manager import CameraManager
from app.utils.worker import (
    TrichXuatWorker,
    NhanDangWorker,
    NhanDangSingleWorker,
    XacMinhWorker,
    TrichXuatBatchWorker,
    VeKhungRunnable,
    VeKhungXacMinhRunnable
)

# ✅ Dùng tts_simple thay vì tts_gtts
from app.utils.tts_simple import speak

__all__ = [
    'logger',
    'CameraManager',
    'TrichXuatWorker',
    'NhanDangWorker',
    'NhanDangSingleWorker',
    'XacMinhWorker',
    'TrichXuatBatchWorker',
    'VeKhungRunnable',
    'VeKhungXacMinhRunnable',
    'speak'
]