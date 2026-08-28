# app/utils/tts_simple.py
# ================================================================
# TEXT-TO-SPEECH - ĐƠN GIẢN
# ================================================================

import os
import tempfile
import threading
from app.utils.logger import logger

# Kiểm tra thư viện
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    logger.warning("[TTS] gTTS chưa cài. Cài bằng: pip install gtts")

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logger.warning("[TTS] pygame chưa cài. Cài bằng: pip install pygame")


def speak(text, lang="vi", voice=None, rate=150, slow=False):
    """
    Phát âm thanh tiếng Việt
    
    Args:
        text: Nội dung cần nói
        lang: Ngôn ngữ (mặc định: vi)
        voice: Không dùng, giữ để tương thích
        rate: Không dùng, giữ để tương thích
        slow: Nói chậm (mặc định: False)
    """
    if not text:
        return
    
    # Chạy trong thread riêng
    thread = threading.Thread(target=_speak_worker, args=(text, lang, slow))
    thread.daemon = True
    thread.start()


def _speak_worker(text, lang="vi", slow=False):
    """Worker thực sự phát âm thanh"""
    try:
        if not GTTS_AVAILABLE or not PYGAME_AVAILABLE:
            logger.warning("[TTS] Thiếu thư viện")
            _speak_fallback(text)
            return
        
        logger.info(f"[TTS] Đang phát: {text[:30]}...")
        
        # Tạo âm thanh
        tts = gTTS(text=text, lang=lang, slow=slow)
        
        # Lưu vào file tạm
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
            tts.write_to_fp(f)
            temp_file = f.name
        
        # Phát âm thanh
        pygame.mixer.init()
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()
        
        # Chờ phát xong
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
        
        pygame.mixer.quit()
        
        # Xóa file tạm
        try:
            os.unlink(temp_file)
        except:
            pass
        
        logger.info(f"[TTS] ✅ Đã phát xong: {text[:30]}...")
        
    except Exception as e:
        logger.error(f"[TTS] Lỗi: {e}")
        _speak_fallback(text)


def _speak_fallback(text):
    """Fallback khi gTTS không hoạt động"""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 170)
        engine.setProperty('volume', 0.9)
        engine.say(text)
        engine.runAndWait()
        logger.info(f"[TTS] ✅ Đã phát (pyttsx3): {text[:30]}...")
        return
    except:
        pass
    
    try:
        import win32com.client
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        speaker.Speak(text)
        logger.info(f"[TTS] ✅ Đã phát (SAPI): {text[:30]}...")
        return
    except:
        pass
    
    logger.warning(f"[TTS] ❌ Không thể phát âm thanh: {text}")