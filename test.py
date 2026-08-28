# test.py
from app.utils.tts_simple import speak, test_speak

# Test nhanh
speak("Xin chào", voice="vi-VN")  # Vẫn hoạt động dù có voice

# Hoặc test đầy đủ
test_speak()