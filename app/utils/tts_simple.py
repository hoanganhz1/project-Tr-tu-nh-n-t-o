# app/utils/tts_simple.py
# ================================================================
# TEXT-TO-SPEECH - CHỈ TIẾNG VIỆT (ĐÃ SỬA LỖI)
# ================================================================

import os
import tempfile
import threading
import time
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

# ================================================================
# QUẢN LÝ MIXER TOÀN CỤC - TRÁNH LỖI
# ================================================================

_mixer_lock = threading.Lock()
_pygame_initialized = False
_last_speak_time = 0
_MIN_INTERVAL = 0.5  # 500ms giữa các lần phát


def speak(text, lang="vi", voice=None, rate=150, slow=False):
    """
    Phát âm thanh tiếng Việt - CÓ XỬ LÝ LỖI
    
    Args:
        text: Nội dung cần nói (TIẾNG VIỆT)
        lang: Ngôn ngữ (mặc định: vi) - LUÔN LÀ TIẾNG VIỆT
        voice: Không dùng, giữ để tương thích
        rate: Không dùng, giữ để tương thích
        slow: Nói chậm (mặc định: False)
    """
    global _last_speak_time
    
    if not text:
        return
    
    # ✅ LUÔN DÙNG TIẾNG VIỆT
    lang = "vi"
    
    # Tránh phát quá nhiều trong thời gian ngắn
    current_time = time.time()
    if current_time - _last_speak_time < _MIN_INTERVAL:
        return
    _last_speak_time = current_time
    
    # Chạy trong thread riêng
    thread = threading.Thread(target=_speak_worker, args=(text, lang, slow))
    thread.daemon = True
    thread.start()


def _speak_worker(text, lang="vi", slow=False):
    """Worker thực sự phát âm thanh - CÓ XỬ LÝ LỖI"""
    global _pygame_initialized
    
    try:
        if not GTTS_AVAILABLE or not PYGAME_AVAILABLE:
            logger.warning("[TTS] Thiếu thư viện, không thể phát tiếng Việt")
            return
        
        logger.info(f"[TTS] Đang phát tiếng Việt: {text[:30]}...")
        
        # Tạo file âm thanh
        tts = gTTS(text=text, lang="vi", slow=slow)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
            tts.write_to_fp(f)
            temp_file = f.name
        
        # ✅ SỬA LỖI: Khởi tạo pygame.mixer an toàn
        with _mixer_lock:
            if not _pygame_initialized:
                try:
                    pygame.mixer.init()
                    _pygame_initialized = True
                    logger.info("[TTS] pygame.mixer đã khởi tạo")
                except Exception as e:
                    logger.error(f"[TTS] Không thể khởi tạo pygame.mixer: {e}")
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
                    return
            
            try:
                # Load và phát
                pygame.mixer.music.load(temp_file)
                pygame.mixer.music.play()
                
                # Chờ phát xong (có timeout để tránh treo)
                timeout = 10  # 10 giây timeout
                elapsed = 0
                while pygame.mixer.music.get_busy() and elapsed < timeout:
                    pygame.time.wait(100)
                    elapsed += 0.1
                
                # Dừng và unload để giải phóng tài nguyên
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
                
            except Exception as e:
                logger.error(f"[TTS] Lỗi phát âm thanh: {e}")
                try:
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
                except:
                    pass
        
        # Xóa file tạm
        try:
            os.unlink(temp_file)
        except Exception as e:
            logger.debug(f"[TTS] Không thể xóa file tạm: {e}")
        
        logger.info(f"[TTS] ✅ Đã phát xong tiếng Việt: {text[:30]}...")
        
    except Exception as e:
        logger.error(f"[TTS] Lỗi phát tiếng Việt: {e}")
        # KHÔNG FALLBACK SANG TIẾNG ANH


def speak_vi(text, slow=False):
    """
    Phát âm thanh tiếng Việt (Hàm rõ ràng)
    
    Args:
        text: Nội dung cần nói (TIẾNG VIỆT)
        slow: Nói chậm (mặc định: False)
    """
    speak(text, lang="vi", slow=slow)


def cleanup_tts():
    """Dọn dẹp tài nguyên TTS khi thoát ứng dụng"""
    global _pygame_initialized
    if _pygame_initialized:
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            _pygame_initialized = False
            logger.info("[TTS] Đã dọn dẹp tài nguyên")
        except:
            pass


# ================================================================
# KIỂM TRA
# ================================================================

def test_speak():
    """Kiểm tra phát âm thanh tiếng Việt"""
    import time
    print("=" * 60)
    print("KIỂM TRA TTS - CHỈ TIẾNG VIỆT")
    print("=" * 60)
    
    print(f"\n[1] Kiểm tra thư viện:")
    print(f"  - gTTS: {'✅ Có' if GTTS_AVAILABLE else '❌ Chưa cài'}")
    print(f"  - pygame: {'✅ Có' if PYGAME_AVAILABLE else '❌ Chưa cài'}")
    
    if not GTTS_AVAILABLE or not PYGAME_AVAILABLE:
        print("\n  Cài đặt bằng lệnh:")
        print("  pip install gtts pygame")
        return
    
    print("\n[2] Phát âm thanh tiếng Việt:")
    
    test_texts = [
        "Xin chào, đây là tiếng Việt",
        "Chào mừng bạn đến với hệ thống nhận diện khuôn mặt",
        "Nhận dạng thành công",
        "Xác minh thất bại, không được phép truy cập",
        "Cảm ơn bạn đã sử dụng hệ thống",
    ]
    
    for text in test_texts:
        print(f"  Đang phát: {text}")
        speak(text)
        time.sleep(2)
    
    # Dọn dẹp
    cleanup_tts()
    
    print("\n" + "=" * 60)
    print("✅ KIỂM TRA HOÀN TẤT")
    print("=" * 60)


if __name__ == "__main__":
    test_speak()