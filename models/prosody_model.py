import librosa
import numpy as np

def analyse_pitch(audio_path: str) -> dict:
    """
    Extract F0 (pitch) using librosa.pyin and summarise variation.
    Pure signal processing - no model download, fast on CPU.
    """
    try:
        # Load audio
        audio_signal, sample_rate = librosa.load(audio_path, sr=None)

        # Extract pitch
        pitch_track, voiced_flags, _ = librosa.pyin(
            audio_signal,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=sample_rate
        )

        # Keep only voiced frames
        voiced_pitch_values = pitch_track[voiced_flags]
        voiced_pitch_values = voiced_pitch_values[~np.isnan(voiced_pitch_values)]

        if len(voiced_pitch_values) < 10:
            return {
                "pitch_range_hz": 0,
                "pitch_stddev": 0,
                "monotone": None,
                "summary": "Not enough voiced audio to analyse pitch."
            }

        # Calculate pitch variation
        pitch_stddev = float(np.std(voiced_pitch_values))
        pitch_range = float(np.percentile(voiced_pitch_values, 95) - np.percentile(voiced_pitch_values, 5))

        if pitch_stddev < 15:
            is_monotone = True
            summary = "Delivery was fairly monotone with limited pitch variation."
        elif pitch_stddev < 30:
            is_monotone = False
            summary = "Some natural pitch variation was present."
        else:
            is_monotone = False
            summary = "Good pitch variation, suggesting engaged, expressive delivery."

        return {
            "pitch_range_hz": round(pitch_range, 1),
            "pitch_stddev": round(pitch_stddev, 1),
            "monotone": is_monotone,
            "summary": summary
        }

    except Exception as e:
        return {
            "pitch_range_hz": 0,
            "pitch_stddev": 0,
            "monotone": None,
            "summary": f"Pitch analysis unavailable: {e}"
        }