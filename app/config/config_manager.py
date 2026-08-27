# app/config/config_manager.py
# ================================================================
# QUẢN LÝ CẤU HÌNH TẬP TRUNG (SINGLETON)
# ================================================================

import json
import os
from pathlib import Path


class ConfigManager:
    """Singleton quản lý toàn bộ cấu hình hệ thống"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._config = {}
        self._config_path = self._get_config_path()
        self._load()
        self._initialized = True

    def _get_config_path(self):
        """Đường dẫn đến file config.json"""
        # Lấy thư mục gốc (app/../)
        base_dir = Path(__file__).parent.parent.parent
        data_dir = base_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir / "config.json"

    def _defaults(self):
        """Cấu hình mặc định"""
        return {
            "threshold": {
                "nhan_dang": 0.35,
                "xac_minh": 0.30
            },
            "model": {
                "embedding_dim": 512,
                "pretrained": "vggface2"
            },
            "camera": {
                "width": 640,
                "height": 480,
                "fps": 30
            },
            "registration": {
                "sample_count": 20,
                "min_sample": 5
            },
            "advanced": {
                "use_clahe": True,
                "use_denoise": True,
                "use_sharpen": True,
                "alpha_contrast": 1.2,
                "beta_brightness": 10,
                "use_ensemble": True,
                "ensemble_count": 5
            },
            "matching": {
                "method": "cosine"  # cosine, euclidean, manhattan
            }
        }

    def _load(self):
        """Tải config từ file"""
        if self._config_path.exists():
            try:
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
                print("[ConfigManager] Đã tải cấu hình từ", self._config_path)
            except Exception as e:
                print(f"[ConfigManager] Lỗi tải config: {e}, dùng mặc định")
                self._config = self._defaults()
        else:
            print("[ConfigManager] Chưa có config.json, tạo mới")
            self._config = self._defaults()
            self._save()

    def _save(self):
        """Lưu config vào file"""
        try:
            with open(self._config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            print("[ConfigManager] Đã lưu cấu hình")
        except Exception as e:
            print(f"[ConfigManager] Lỗi lưu config: {e}")

    def get(self, key, default=None):
        """Lấy giá trị theo key dạng 'a.b.c'"""
        parts = key.split('.')
        value = self._config
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
                if value is None:
                    return default
            else:
                return default
        return value if value is not None else default

    def set(self, key, value):
        """Gán giá trị theo key dạng 'a.b.c' và lưu"""
        parts = key.split('.')
        config = self._config
        for part in parts[:-1]:
            if part not in config or not isinstance(config[part], dict):
                config[part] = {}
            config = config[part]
        config[parts[-1]] = value
        self._save()

    def reload(self):
        """Tải lại config từ file (bỏ qua thay đổi chưa lưu)"""
        self._load()

    def get_all(self):
        """Trả về toàn bộ config"""
        return self._config.copy()