"""
voice_assistant.py — Voice Assistant Module with reliable interrupt support.

Provides:
  - PorcupineInterruptListener : wake-word / interrupt detection during playback
  - VoiceAssistant             : STT (Google via SpeechRecognition),
                                  TTS (Edge-TTS), and interruptible playback

Uses Porcupine (pvporcupine) + pyaudio for low-latency keyword detection.
"""

import speech_recognition as sr
import edge_tts
import asyncio
import pygame
import threading
import os
import tempfile
import re
import struct

from dotenv import load_dotenv

load_dotenv()

# Read Picovoice access key from environment
PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY", "")

# Porcupine / PyAudio imports (optional — graceful degradation)
try:
    import pvporcupine
    import pyaudio
    PV_AVAILABLE = True
except Exception:
    PV_AVAILABLE = False
    print("⚠️  Porcupine unavailable – interrupt detection disabled. "
          "Install via: pip install pvporcupine pyaudio")


# ==============================================
#  INTERRUPT LISTENER (Porcupine-based)
# ==============================================

class PorcupineInterruptListener:
    """Listens for wake/interrupt keywords during TTS playback."""

    def __init__(self, keywords=("computer", "terminator")):
        if not PV_AVAILABLE:
            raise RuntimeError("Porcupine not available")

        if not PICOVOICE_ACCESS_KEY:
            raise RuntimeError("❌ Missing PICOVOICE_ACCESS_KEY in environment.")

        self.keywords = list(keywords)

        self.pp = pvporcupine.create(
            access_key=PICOVOICE_ACCESS_KEY,
            keywords=self.keywords,
        )

        self.pa = pyaudio.PyAudio()

        self.stream = self.pa.open(
            rate=self.pp.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.pp.frame_length,
        )

        self.interrupted = threading.Event()
        self._running = threading.Event()
        self._thread = None

    # ---- internal loop ---------------------------------------------------

    def _loop(self):
        while self._running.is_set() and not self.interrupted.is_set():
            try:
                pcm_data = self.stream.read(
                    self.pp.frame_length, exception_on_overflow=False
                )
                pcm = struct.unpack_from("h" * self.pp.frame_length, pcm_data)

                keyword_index = self.pp.process(pcm)
                if keyword_index >= 0:
                    print(f"🛑 Interrupt detected: {self.keywords[keyword_index]}")
                    self.interrupted.set()
                    break
            except Exception:
                continue

    # ---- public API ------------------------------------------------------

    def start(self):
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=1)

        try:
            self.stream.stop_stream()
            self.stream.close()
        except Exception:
            pass

        try:
            self.pa.terminate()
        except Exception:
            pass

        try:
            self.pp.delete()
        except Exception:
            pass

    def was_interrupted(self):
        return self.interrupted.is_set()


# ==============================================
#  MAIN VOICE ASSISTANT CLASS
# ==============================================

class VoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=512)

        self._temp_dir = tempfile.mkdtemp(prefix="voice_assistant_")
        self.temp_audio_file = os.path.join(self._temp_dir, "temp_response.mp3")

        self.interrupt_keywords = ["computer", "terminator"]

        self.adjust_for_ambient_noise()
        print("✅ Voice Assistant initialized")

    def adjust_for_ambient_noise(self, duration=1):
        """Calibrate the recogniser's energy threshold to ambient noise."""
        print("🎤 Calibrating microphone...")
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=duration)
        except Exception:
            print("⚠️  Mic calibration failed")

    # ------------------------------------------------------------------
    # LISTENING  (Speech-to-Text)
    # ------------------------------------------------------------------

    def listen_for_speech(self, timeout=30, phrase_time_limit=15):
        """
        Block until the user speaks a phrase (or timeout).

        Returns
        -------
        (text, False)  on success
        (None, False)  on timeout / recognition failure
        """
        try:
            with self.microphone as source:
                print("🎤 Listening...")
                audio = self.recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_time_limit
                )

            print("🔄 Recognizing...")
            text = self.recognizer.recognize_google(audio)
            print(f"📝 You said: {text}")
            return text, False

        except sr.WaitTimeoutError:
            return None, False
        except sr.UnknownValueError:
            print("⚠️  Could not understand speech")
            return None, False
        except Exception as e:
            print(f"❌ Speech error: {e}")
            return None, False

    # ------------------------------------------------------------------
    # TTS GENERATION  (Text-to-Speech via Edge-TTS)
    # ------------------------------------------------------------------

    async def generate_speech_async(self, text, voice="en-US-AriaNeural"):
        try:
            await edge_tts.Communicate(text, voice).save(self.temp_audio_file)
            return True
        except Exception as e:
            print(f"❌ TTS error: {e}")
            return False

    def generate_speech(self, text, voice="en-US-AriaNeural"):
        """Synchronous wrapper around the async Edge-TTS generator."""
        return asyncio.run(self.generate_speech_async(text, voice))

    # ------------------------------------------------------------------
    # PLAYBACK WITH INTERRUPT
    # ------------------------------------------------------------------

    def play_audio_with_interrupt(self):
        """
        Play the temp audio file.  If Porcupine detects an interrupt keyword
        during playback, stop immediately and return False.

        Returns True if playback completed, False if interrupted.
        """
        try:
            pygame.mixer.music.load(self.temp_audio_file)
        except Exception as e:
            print(f"❌ Failed loading audio: {e}")
            return False

        listener = None
        if PV_AVAILABLE:
            try:
                listener = PorcupineInterruptListener(
                    keywords=tuple(self.interrupt_keywords)
                )
                listener.start()
            except Exception as e:
                print(f"❌ Porcupine error: {e}")

        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            if listener and listener.was_interrupted():
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
                listener.stop()
                return False

            pygame.time.Clock().tick(50)

        if listener:
            listener.stop()

        try:
            pygame.mixer.music.unload()
        except Exception:
            pass

        return True

    # ------------------------------------------------------------------
    # SPEAK (single utterance, with interrupt)
    # ------------------------------------------------------------------

    def speak_with_interrupt(self, text, voice="en-US-AriaNeural"):
        """
        Generate TTS for *text* and play it back.

        Returns
        -------
        (True,  None)          — playback completed
        (False, "interrupted") — user interrupted via keyword
        (False, None)          — TTS generation failed
        """
        print("🔊 Generating speech...")
        if not self.generate_speech(text, voice):
            return False, None

        print("🔊 Speaking...")
        completed = self.play_audio_with_interrupt()

        if completed:
            return True, None
        return False, "interrupted"

    # ------------------------------------------------------------------
    # SPEAK SENTENCES (sentence-by-sentence with interrupt)
    # ------------------------------------------------------------------

    def speak_sentences(self, text):
        """
        Split *text* at sentence boundaries and speak each sentence.
        Stops immediately if the user interrupts any sentence.

        Returns
        -------
        (True,  None)          — all sentences spoken
        (False, "interrupted") — interrupted mid-way
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        for s in sentences:
            done, reason = self.speak_with_interrupt(s)
            if not done:
                return False, reason
        return True, None

    # ------------------------------------------------------------------
    # CLEANUP
    # ------------------------------------------------------------------

    def cleanup(self):
        """Release audio resources and remove temp files."""
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        except Exception:
            pass

        try:
            if os.path.exists(self.temp_audio_file):
                os.remove(self.temp_audio_file)
        except Exception:
            pass
