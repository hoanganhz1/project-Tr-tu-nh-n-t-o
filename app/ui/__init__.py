# app/ui/__init__.py
# ================================================================
# UI PACKAGE
# ================================================================

# Import các page để sử dụng ở nơi khác
from app.ui.registration_page import RegistrationPage
from app.ui.recognition_page import RecognitionPage
from app.ui.database_page import DatabasePage
from app.ui.settings_page import SettingsPage

# FaceSecureApp được import trực tiếp từ main_window khi cần
# from app.ui.main_window import FaceSecureApp