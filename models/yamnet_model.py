import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import scipy.io.wavfile as wav
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import YAMNET_MODEL_URL, YAMNET_CLASS_MAP_URL, YAMNET_THRESHOLD

_yamnet_model = None
_sound_class_names = None

def load_model():
    """Load YAMNet model once and reuse."""
    global _yamnet_model, _sound_class_names
    if _yamnet_model is None:
        print("Loading YAMNet model...")
        _yamnet_model = hub.load(YAMNET_MODEL_URL)

        # Load class names
        class_map_path = tf.keras.utils.get_file(
            "yamnet_class_map.csv",
            YAMNET_CLASS_MAP_URL
        )
        _sound_class_names = [
            line.split(",")[2].strip().strip('"')
            for line in open(class_map_path).readlines()[1:]
        ]
        print("YAMNet loaded.")
    return _yamnet_model, _sound_class_names

def analyse_audio(audio_path: str) -> dict:
    """
    Analyse audio file for speech events, pauses and filler sounds.
    Returns dict with summary and timestamped events.
    """
    yamnet_model, sound_class_names = load_model()
    print(f"Analysing audio events in {audio_path}...")
    sample_rate, audio_samples = wav.read(audio_path)

    # Convert to mono
    if len(audio_samples.shape) > 1:
        audio_samples = audio_samples.mean(axis=1)

    audio_samples = audio_samples.astype(np.float32) / 32768.0

    # Run model
    class_scores, _, _ = yamnet_model(audio_samples)
    class_scores = class_scores.numpy()

    # Build timestamped events
    detected_events = []
    chunk_duration = 0.5
    for chunk_index, frame_scores in enumerate(class_scores):
        top_class_index = frame_scores.argmax()
        top_class_score = frame_scores[top_class_index]
        top_class_name = sound_class_names[top_class_index]
        timestamp = round(chunk_index * chunk_duration, 1)

        if top_class_score >= YAMNET_THRESHOLD:
            detected_events.append({
                "time": timestamp,
                "event": top_class_name,
                "confidence": round(float(top_class_score), 3)
            })

    # Count key signals
    speech_event_count = sum(1 for event in detected_events if "Speech" in event["event"])
    silence_event_count = sum(1 for event in detected_events if "Silence" in event["event"])
    notable_event_count = sum(1 for event in detected_events if any(
        keyword in event["event"] for keyword in ["Cough", "Throat", "Laughter", "Breathing"]
    ))

    total_events = len(detected_events) if detected_events else 1
    result = {
        "events": detected_events,
        "summary": {
            "total_chunks": len(class_scores),
            "speech_ratio": round(speech_event_count / total_events, 2),
            "silence_ratio": round(silence_event_count / total_events, 2),
            "notable_audio_events": notable_event_count,
            "dominant_class": sound_class_names[class_scores.mean(axis=0).argmax()]
        }
    }

    print(f"Audio analysis complete. {len(detected_events)} events detected.")
    return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = analyse_audio(sys.argv[1])
        print("\n=== SUMMARY ===")
        for key, value in result["summary"].items():
            print(f"  {key}: {value}")
        print(f"\nFirst 5 events:")
        for event in result["events"][:5]:
            print(f"  {event['time']}s - {event['event']} ({event['confidence']})")
    else:
        print("Usage: python models/yamnet_model.py <audio_file>")