import os
from moviepy.editor import VideoFileClip
from transformers import pipeline, AutoModelForAudioClassification, AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers import AutoConfig, Wav2Vec2FeatureExtractor
from deepface import DeepFace
import librosa
import numpy as np
import openai
import torch

# Video and audio paths
videos = [
    {"video_path": "/home/katka/PycharmProjects/2024-final-applied-minf/videos/So Sorry.mp4",
     "audio_output_path": "extracted_audio1.wav"},
    {"video_path": "/home/katka/PycharmProjects/2024-final-applied-minf/videos/My Response.mp4",
     "audio_output_path": "extracted_audio2.wav"}
]

# Audio Extraction
def extract_audio_from_video(video_path, output_path):
    try:
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(output_path)
        print(f"Audio extracted to {output_path}")
    except Exception as e:
        print(f"Error extracting audio: {e}")

# Emotion Recognition
def extract_emotion_from_video(video_path):
    frames = []
    video = VideoFileClip(video_path)
    for frame in video.iter_frames(fps=video.fps, dtype="uint8"):
        frames.append(frame)
    emotions = []
    for i, frame in enumerate(frames):
        if i % 100 != 0:
            continue
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if analysis:
            emotions.append(analysis[0]['dominant_emotion'])
    return emotions

def extract_emotion_from_audio(audio_path):
    target_sample_rate = 16000
    audio_data, _ = librosa.load(audio_path, sr=target_sample_rate)
    config = AutoConfig.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    model = AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    inputs = feature_extractor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    predicted_class_ids = torch.argmax(logits, dim=-1)
    return [model.config.id2label[id.item()] for id in predicted_class_ids]

# Transcription
def get_transcription(audio_path):
    model_id = "openai/whisper-small"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id, language='en')
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
    audio_data, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt").to(device)
    with torch.no_grad():
        transcription = model.generate(inputs.input_features)
    return processor.decode(transcription[0], skip_special_tokens=True)

def send_chat_completion(prompt, model="llama-3.2-1b-instruct", temperature=0.7):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error sending prompt to LM server: {e}")
        return "I'm here for you."

# LLM Interaction
def feed_to_LLM(conversation_context, emotions_from_video, emotions_from_audio, transcription):
    openai.api_base = "http://localhost:1234/v1"
    openai.api_key = "lm-studio"
    model = "llama-3.2-1b-instruct"

    prompt = f"""
    You are a psychologist analyzing a person's emotional and verbal state. Based on the following context and new information:
    Context:
    {conversation_context}

    New Information:
    - Observed emotions from video analysis: {emotions_from_video}
    - Detected emotions from audio analysis: {emotions_from_audio}
    - Transcription of their spoken words: "{transcription}"

    Provide a compassionate and thoughtful response, offering insights and guidance. 
    The response should be no longer than two sentences of spoken word.
    """
    response = send_chat_completion(prompt, model)
    return response

# Main Workflow
conversation_context = ""
for i, video in enumerate(videos):
    video_path = video["video_path"]
    audio_output_path = video["audio_output_path"]

    print(f"Processing video {i + 1}...")
    extract_audio_from_video(video_path, audio_output_path)
    video_emotions = extract_emotion_from_video(video_path)
    audio_emotions = extract_emotion_from_audio(audio_output_path)
    transcription = get_transcription(audio_output_path)

    response = feed_to_LLM(conversation_context, video_emotions, audio_emotions, transcription)
    conversation_context += f"\nVideo {i + 1} Analysis:\nResponse: {response}"
    print(f"LLM Response for Video {i + 1}:\n{response}")
