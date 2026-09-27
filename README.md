## AI Speaking Skills Tutor

A locally executed multimodal AI system for practising and improving academic speaking skills.

The system allows students to select a topic and difficulty, generate an academic-style speaking question, record or upload a video response, and receive automated feedback on content, delivery, visual engagement, grammar, and language level.

The main analysis pipeline runs locally on the user's computer without requiring a cloud AI API or subscription.


## Features

* Academic-style speaking question generation
* Beginner, intermediate, and advanced difficulty levels
* Video recording or upload
* Automatic speech transcription
* Filler-word detection
* Pause and speaking-rate analysis
* Audio-event and silence analysis
* Prosody/pitch analysis
* Eye-contact and visual engagement analysis
* Head-orientation analysis
* Grammar checking
* CEFR language-level estimation
* Automated speaking scores
* Targeted coaching suggestions
* Adaptive follow-up practice
* PDF report export
* Local AI processing


## Technologies Used

| Elements               | Technologies                   |
|------------------------|--------------------------------|
| Interface              | Gradio                         |
| Programming            | Python                         |
| Speech recognition     | Whisper Small                  |
| Audio analysis         | YAMNet                         |
| Visual analysis        | MediaPipe Face Landmarker      |
| Language model         | Phi-3                          |
| Local LLM runtime      | Ollama                         |
| Grammar checking       | LanguageTool                   |
| Prosody analysis       | librosa                        |
| Video/audio extraction | FFmpeg via Python `subprocess` |
| PDF export             | fpdf                           |


## Requirements

* Python 3.13.5 (tested)
* Ollama
* Phi-3 model
* FFmpeg
* A modern web browser
* Sufficient local storage for the required models and dependencies

The system is designed for local CPU-based execution. Processing time depends on the computer's hardware and the length of the recording.


## Installation

### 1. Download the Repository 

Download the repository as a ZIP file from GitHub and extract it to a suitable location.

Install Python and ensure that **"Add python.exe to PATH"** is selected during installation.

Open PowerShell in the extracted project folder.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment:

```bash
venv\Scripts\activate
```

If PowerShell prevents the virtual environment from activating, run:

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the environment again:

```bash
venv\Scripts\activate
```

### 3. Install Python dependencies

The repository includes `requirements.txt`. Install the required Python packages using:

```bash
python -m pip install -r requirements.txt
```

Do not create a new `requirements.txt` file manually.

### 4. Install FFmpeg

Install FFmpeg separately and ensure that it is available through the system PATH.

The application uses FFmpeg through Python's `subprocess` module to extract the audio required for analysis.


## Ollama and Phi-3 Setup

The project uses Ollama to run the Phi-3 language model locally.

Install Ollama separately from the official Ollama website.

After installation, open PowerShell or a terminal and run:

```bash
ollama run phi3
```

This downloads the Phi-3 model if it is not already available.

The application communicates with the local Ollama service through:

```text
"http://localhost:11434/api/generate"
```

`ollama serve` does not normally need to be run separately when using `ollama run phi3`.

## Model Resources

The repository does not contain all model weights.

Some model resources are downloaded or obtained automatically when they are first required by the application.

These include:

* **Whisper Small** - downloaded/cached when the Whisper model is loaded.
* **YAMNet** - obtained through TensorFlow Hub.
* **YAMNet AudioSet class map** - downloaded automatically when required.
* **MediaPipe Face Landmarker** - `face_landmarker.task` is downloaded automatically if it is not already present.
* **Phi-3** - downloaded and managed separately through Ollama.

An internet connection may therefore be required during the initial setup. Once the required resources have been obtained, the main inference pipeline runs locally.

## Running the Application

After completing the installation and Ollama setup, make sure the Python virtual environment is activated.

Run:

```bash
python app.py
```

The terminal will display the local Gradio address. Open this address in a web browser.


## Privacy and Limitations

The main speech, audio, video, language, and scoring processes are performed locally on the user's computer. The system does not require a cloud AI API for its main analysis pipeline.

The system is intended as a speaking-practice and coaching tool rather than a replacement for a qualified human assessor. Automatic transcription, grammar analysis, visual analysis, CEFR estimation, and AI-generated feedback may contain errors.

Some analysis thresholds were manually defined and have not been externally validated against large datasets or human assessor scores.

Processing time can also be lengthy on CPU-only hardware, particularly for longer recordings.


## Project Purpose

This project demonstrates how multiple pretrained AI models can be orchestrated to provide multimodal speaking-practice feedback within a single locally executed application.

The system was developed as a final-year project investigating the use of locally executed AI for accessible, private, and repeatable academic speaking practice.