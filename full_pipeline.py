# Unified Emotion and Speech Processing Script

import os
from moviepy.editor import VideoFileClip
from transformers import pipeline, AutoModelForAudioClassification, AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers import pipeline, AutoConfig,Wav2Vec2FeatureExtractor, AutoModelForAudioClassification, AutoFeatureExtractor

from deepface import DeepFace
import librosa
import numpy as np
import openai
import torch

# Video and audio paths
video_path = "/home/katka/PycharmProjects/2024-final-applied-minf/videos/So Sorry.mp4"
audio_output_path = "extracted_audio.wav"

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
    # Process frames with DeepFace
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
    # predicted_class_id = torch.mode(predicted_class_ids[predicted_class_ids != -1]).values.item()
    # predicted_label = model.config.id2label.get(predicted_class_id, "Unknown")
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
    """
    Sends a chat completion request to the LM Studio server and retrieves advice.

    Args:
        prompt (str): The prompt to send to the LM model.
        model (str): The model identifier from LM Studio.
        temperature (float): Controls the creativity of the response.

    Returns:
        str: The advice returned by the LM model, or a default message if an error occurs.
    """
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
def feed_to_LLM(emotions_from_video, emotions_from_audio, transcription):
    # Configure the OpenAI client for LM Studio
    openai.api_base = "http://localhost:1234/v1"
    openai.api_key = "lm-studio"
    # model = 'mistral-small-instruct-2409'
    model = "llama-3.2-1b-instruct"

    prompt = f"""
    You are a psychologist analyzing a person's emotional and verbal state. Based on the following information:
    - Observed emotions from video analysis: {emotions_from_video}
    - Detected emotions from audio analysis: {emotions_from_audio}
    - Transcription of their spoken words: "{transcription}"

    Provide a compassionate and thoughtful response, offering insights and guidance as a psychologist would. 
    Aim to address their emotional state and provide actionable advice or comfort to help them navigate their feelings. 
    The response should be no longer then two sentences of spoken word.
    """
    print()
    print(prompt)
    print()
    response = send_chat_completion(prompt, model)
    return response

# Main Workflow

print("extract_audio_from_video(video_path, audio_output_path)")
extract_audio_from_video(video_path, audio_output_path)

print("video_emotions = extract_emotion_from_video(video_path)")
video_emotions = extract_emotion_from_video(video_path)
# video_emotions = ['sad', 'fear', 'neutral', 'neutral', 'neutral', 'fear', 'neutral', 'fear', 'neutral', 'sad', 'neutral', 'fear', 'neutral', 'neutral', 'fear', 'fear', 'fear', 'neutral', 'neutral', 'neutral', 'sad', 'neutral', 'fear', 'neutral', 'neutral', 'neutral', 'fear', 'neutral', 'neutral', 'neutral', 'sad', 'fear', 'sad', 'fear', 'fear', 'surprise', 'neutral', 'neutral', 'fear', 'fear', 'neutral', 'fear', 'neutral', 'neutral', 'sad', 'neutral', 'neutral', 'neutral', 'fear', 'sad', 'sad', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'fear', 'fear', 'fear', 'neutral', 'fear', 'sad', 'fear']
print(video_emotions)

print("audio_emotions = extract_emotion_from_audio(audio_output_path)")
audio_emotions = extract_emotion_from_audio(audio_output_path)
# audio_emotions = ['sad']
print(audio_emotions)

print("transcription = get_transcription(audio_output_path)")
transcription = get_transcription(audio_output_path)
# transcription = "I've made a severe and continuous lapse in my judgment and I don't expect to be forgiven. I'm simply here to apologize. So what we came across that day in the woods was obviously unplanned and the reactions you saw on tape were raw. They were unfiltered. None of us knew how to react or how to feel. I should have never posted the video. I should have put the cameras down and stopped the video."
print(transcription)

print("response = feed_to_LLM(video_emotions, audio_emotions, transcription)")
response = feed_to_LLM(video_emotions, audio_emotions, transcription)

# Output Result
print("LLM Response:")
print(response)
