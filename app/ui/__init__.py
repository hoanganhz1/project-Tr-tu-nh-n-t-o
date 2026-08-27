# app/ui/__init__.py
# ================================================================
# UI PACKAGE - KHÔNG IMPORT VÒNG TRÒN
# ================================================================

# ❌ KHÔNG import trực tiếp từ main_window ở đây
# ✅ Chỉ import khi cần thiết ở file khác

# Các import này sẽ được sử dụng ở nơi khác
from app.ui.registration_page import RegistrationPage
from app.ui.recognition_page import RecognitionPage
from app.ui.database_page import DatabasePage
from app.ui.settings_page import SettingsPage

# FaceSecureApp sẽ được import trực tiếp từ main_window khi cần
# Không import ở đây để tránh vòng tròn