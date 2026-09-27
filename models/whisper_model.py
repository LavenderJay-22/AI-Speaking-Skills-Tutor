import whisper
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WHISPER_MODEL_SIZE

_model = None

FILLER_WORDS = [
    "um", "uh", "er", "ah", "you know", "basically",
    "literally", "i mean", "kind of", "sort of",
    "honestly"
]

def load_model():
    """Load Whisper model once and reuse."""
    global _model
    if _model is None:
        print(f"Loading Whisper {WHISPER_MODEL_SIZE} model...")
        _model = whisper.load_model(WHISPER_MODEL_SIZE)
        print("Whisper loaded.")
    return _model

def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe audio file to text with word timestamps.
    Returns dict with transcript text and word list.
    """
    whisper_model = load_model()
    print(f"Transcribing {audio_path}...")
    transcription_result = whisper_model.transcribe(
        audio_path,
        word_timestamps=True,
        initial_prompt="Um, uh, you know, basically, literally, er,"
    )

    word_list = []
    for segment in transcription_result["segments"]:
        for word_data in segment.get("words", []):
            word_list.append({
                "word": word_data["word"].strip(),
                "start": round(word_data["start"], 2),
                "end": round(word_data["end"], 2)
            })

    transcript = {
        "text": transcription_result["text"].strip(),
        "words": word_list
    }

    print(f"Transcription complete. {len(word_list)} words found.")
    return transcript

def detect_fillers(transcript: dict) -> dict:
    """
    Detect filler words in transcript with timestamps.
    Returns list of filler word occurrences.
    """
    filler_instances = []
    words = transcript.get("words", [])
    for word_data in words:
        cleaned_word = word_data["word"].lower().strip().strip(".,!?;: ")
        if cleaned_word in FILLER_WORDS:
            filler_instances.append({
                "word": word_data["word"].strip(),
                "time": word_data["start"]
            })

    return {
        "count": len(filler_instances),
        "instances": filler_instances
    }

def analyse_pauses(transcript: dict) -> dict:
    """
    Analyse gaps between consecutive words to distinguish natural
    breathing pauses from long hesitant silences. Uses the same
    word timestamps transcribe_audio() already produces.
    """
    words = transcript.get("words", [])
    if len(words) < 2:
        return {
            "pause_count": 0,
            "long_pause_count": 0,
            "longest_pause": 0.0,
            "avg_pause": 0.0,
            "pauses": [],
            "summary": "Not enough speech to analyse pauses."
        }

    LONG_PAUSE_MIN = 1.5   # hesitant/dead silence
    detected_pauses = []

    for word_index in range(1, len(words)):
        gap_duration = words[word_index]["start"] - words[word_index - 1]["end"]
        if gap_duration > 0.15:  # ignore near-zero rounding gaps
            detected_pauses.append({
                "after_word": words[word_index - 1]["word"],
                "duration": round(gap_duration, 2),
                "time": words[word_index - 1]["end"]
            })

    long_pauses = [pause for pause in detected_pauses if pause["duration"] >= LONG_PAUSE_MIN]

    if not detected_pauses:
        summary = "Speech was continuous with no notable pauses."
    elif len(long_pauses) == 0:
        summary = "Pauses were short and natural, consistent with normal breathing/thinking."
    elif len(long_pauses) <= 2:
        summary = "A couple of long hesitant pauses were detected."
    else:
        summary = "Frequent long pauses suggest hesitation or difficulty organising ideas."

    avg_pause_duration = round(sum(pause["duration"] for pause in detected_pauses) / len(detected_pauses), 2) if detected_pauses else 0.0
    longest_pause_duration = round(max((pause["duration"] for pause in detected_pauses), default=0.0), 2)

    return {
        "pause_count": len(detected_pauses),
        "long_pause_count": len(long_pauses),
        "longest_pause": longest_pause_duration,
        "avg_pause": avg_pause_duration,
        "pauses": long_pauses[:10],  # cap for report display
        "summary": summary
    }

def analyse_pace(transcript: dict, window_seconds: float = 5.0) -> dict:
    """
    Compute words-per-minute in rolling time windows to detect
    pace consistency, not just an overall average.
    """
    words = transcript.get("words", [])

    if len(words) < 5:
        return {
            "avg_wpm": 0,
            "wpm_stddev": 0,
            "consistency": "insufficient data",
            "summary": "Not enough speech to analyse pace."
        }

    total_duration = words[-1]["end"] - words[0]["start"]
    avg_wpm = round((len(words) / total_duration) * 60) if total_duration > 0 else 0

    # Bucket words into fixed-size time windows, compute WPM per window
    window_wpm_values = []
    window_start_time = words[0]["start"]
    words_in_current_window = 0

    for word_data in words:
        if word_data["start"] - window_start_time >= window_seconds:
            window_wpm_values.append(words_in_current_window / window_seconds * 60)
            window_start_time = word_data["start"]
            words_in_current_window = 0
        words_in_current_window += 1

    if words_in_current_window > 0:
        window_wpm_values.append(words_in_current_window / window_seconds * 60)

    if len(window_wpm_values) < 2:
        wpm_stddev = 0
    else:
        mean_wpm = sum(window_wpm_values) / len(window_wpm_values)
        wpm_stddev = round((sum((wpm - mean_wpm) ** 2 for wpm in window_wpm_values) / len(window_wpm_values)) ** 0.5)

    if wpm_stddev < 15:
        consistency = "steady"
        summary = "Speaking pace was steady and consistent throughout."
    elif wpm_stddev < 30:
        consistency = "moderate variation"
        summary = "Speaking pace varied somewhat across the response."
    else:
        consistency = "inconsistent"
        summary = "Speaking pace fluctuated significantly, e.g. starting fast then trailing off."

    return {
        "avg_wpm": avg_wpm,
        "wpm_stddev": wpm_stddev,
        "consistency": consistency,
        "summary": summary
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = transcribe_audio(sys.argv[1])
        print("\n=== TRANSCRIPT ===")
        print(result["text"])
        print(f"\nFirst 5 words:")
        for word_data in result["words"][:5]:
            print(f"  '{word_data['word']}' at {word_data['start']}s")
    else:
        print("Usage: python models/whisper_model.py <audio_file>")