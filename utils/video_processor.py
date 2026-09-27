import sys
import subprocess
import os

def extract_audio(video_path: str, output_path: str = None) -> str:
    """
    Extract audio from a video file and save as 16000 Hz mono WAV.
    Accepts any video or audio format ffmpeg supports.
    Returns path to the extracted WAV file.
    """
    if output_path is None:
        base_filename = os.path.splitext(video_path)[0]
        output_path = base_filename + "_audio.wav"

    print(f"Extracting audio from {video_path}...")

    # Build the FFmpeg command
    ffmpeg_command = [
        "ffmpeg",
        "-i", video_path,
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        "-y",
        output_path
    ]

    # Run FFmpeg
    ffmpeg_result = subprocess.run(
        ffmpeg_command,
        capture_output=True,
        text=True
    )

    if ffmpeg_result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {ffmpeg_result.stderr}")

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Audio file not created at {output_path}")

    print(f"Audio saved to {output_path}")
    return output_path

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = extract_audio(sys.argv[1])
        print(f"Success: {path}")
    else:
        print("Usage: python utils/video_processor.py <video_file>")