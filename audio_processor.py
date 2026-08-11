# Location: ./audio_processor.py
import os
from faster_whisper import WhisperModel

class AudioTranscriber:
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        # Loads Whisper locally to convert audio notes to text
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str) -> str:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file missing: {audio_path}")
            
        segments, _ = self.model.transcribe(audio_path, beam_size=5)
        return " ".join([segment.text.strip() for segment in segments])