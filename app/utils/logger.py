# app/utils/logger.py
# ================================================================
# LOGGER - GHI LOG CHI TIẾT
# ================================================================

import logging
import os
from datetime import datetime

# Tạo thư mục logs
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, f"facesecure_{datetime.now().strftime('%Y%m%d')}.log")

def setup_logger(name="FaceSecure", log_level=logging.INFO):
    """Cấu hình logger"""
    
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Tránh duplicate handlers
    if logger.handlers:
        return logger
    
    # File handler
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

# Logger mặc định
logger = setup_logger()