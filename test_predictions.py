import os
import tempfile
import numpy as np
import soundfile as sf
from src.audio_processor import AudioProcessor
from src.video_processor import VideoProcessor
from src.text_processor import TextProcessor

print('Testing audio, video, and text processors...')

# Generate a short test WAV file for audio prediction
with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
    sample_rate = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = 0.1 * np.sin(2 * np.pi * 440 * t)
    sf.write(tmp.name, tone, sample_rate)
    wav_path = tmp.name

try:
    print('Testing AudioProcessor...')
    ap = AudioProcessor()
    audio_score = ap.predict(wav_path)
    print(f'Audio score: {audio_score:.4f}')

    print('Testing VideoProcessor model loading...')
    vp = VideoProcessor()
    print(f'Video model loaded: {vp.model is not None}')

    print('Testing TextProcessor...')
    tp = TextProcessor()
    english_score = tp.predict('This is a test text.')
    hindi_score = tp.predict('यह एक परीक्षण है।')
    print(f'English text score: {english_score:.4f}')
    print(f'Hindi text score: {hindi_score:.4f}')

    print('All OK.')
finally:
    try:
        os.remove(wav_path)
    except OSError:
        pass
