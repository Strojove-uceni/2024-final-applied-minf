# AI Therapy Platform

Topic: Emotion recognition from video stream with feedback: Create a large multi-modal conversational agent using pre-trained open models.

This is an AI-powered therapy tool that analyzes video and audio inputs to extract emotions, generate transcriptions, 
and provide a responsive interaction using a large language model (LLM). 

---

## Features

### Video Analysis
- **Emotion Detection**: Extracts emotions from video frames using DeepFace.

### Audio Analysis
- **Emotion Detection**: Recognizes emotions in audio using Wav2Vec2-based audio classification.
- **Speech-to-Text**: Converts audio input into text using the OpenAI Whisper model.

### Live Interaction
- **Real-Time Recording**: Captures video and audio in real time.
- **Language Model Response**: Interacts with an LLM to generate compassionate and insightful responses based on emotional and verbal input.
- **Text-to-Speech (TTS)**: Converts the LLM-generated responses into audio output using ChatTTS.

---

## Installation

Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

Additional setup:
   - Ensure `ffmpeg` is installed and available in your PATH for video processing.
   - Configure your GPU if available for enhanced performance.
---

## Usage

### Live Interaction
Run the live interactive session:
```bash
python live_interface.py
```

1. Press `S` to start recording. Follow on-screen instructions for optimal results.
2. Recording stops automatically after a pre-configured time limit or by pressing `S` again.
3. Responses are displayed and played back as audio.

### Batch Processing
To process pre-recorded video and audio files (such as for evaluation), use the pipeline provided in `full_pipeline.py`:
```bash
python full_pipeline.py
```

---

## Overview

### Scripts
- **`live_intercafe.py`**: Implements the real-time video and audio recording interface.
- **`full_pipeline.py`**: Contains the core processing pipeline for emotion extraction, transcription, and interaction with the LLM.

---

## Dependencies

Key libraries and frameworks:
- **DeepFace**: For video emotion analysis.
- **Librosa & Wav2Vec2**: For audio processing and classification.
- **OpenAI Whisper**: For speech-to-text conversion.
- **ChatTTS**: For text-to-speech synthesis.
- **Transformers**: For interacting with pre-trained language models.
- **RetinaFace**: For advanced facial recognition and detection.
- **Facenet-Pytorch**: For facial feature extraction.

To successfully run the application, install [LM Studio](https://lmstudio.ai/) and run **`llama-3.2-1b-instruct`** or another LLM of your choice locally.

Refer to `requirements.txt` for the full list of dependencies.


