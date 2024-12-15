import os
import numpy as np
from moviepy.editor import VideoFileClip
from transformers import AutoConfig, AutoProcessor, AutoModelForAudioClassification, AutoModelForSpeechSeq2Seq, Wav2Vec2FeatureExtractor

from deepface import DeepFace
import librosa
import openai
import torch
from collections import Counter
from datetime import datetime
import threading
from collections import defaultdict
import tensorflow as tf

# suppressing some of the warnings
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
from transformers import logging
logging.set_verbosity_error()  # Suppress all warnings


# Video and audio paths
videos = [
    {'video_path': 'recorded_video_20241215_201518.mp4', 'audio_output_path': 'recorded_audio_20241215_201518.wav'}
]
previous_answer = ''

def extract_audio_from_video(video_path, output_path):
    try:
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(output_path)
        print(f"Audio extracted to {output_path}")
    except Exception as e:
        print(f"Error extracting audio: {e}")


# MAIN AI PROCESSING BLOCKS
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
    # Count the frequency of each emotion
    emotion_counts = Counter(emotions)
    # Get the 2 or 3 most common emotions
    most_common_emotions = emotion_counts.most_common(3)
    # Return only the emotion names
    return [emotion for emotion, count in most_common_emotions]

def extract_emotion_from_audio(audio_path):
    target_sample_rate = 16000
    audio_data, _ = librosa.load(audio_path, sr=target_sample_rate)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    model = AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    inputs = feature_extractor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    predicted_class_ids = torch.argmax(logits, dim=-1)
    return [model.config.id2label[id.item()] for id in predicted_class_ids]

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


# LLM INTERACTION

# def generate_prompt(conversation_context, emotions_from_video, emotions_from_audio, transcription):
#     prompt = f"""
#         You are a psychologist analyzing a person's emotional and verbal state. Based on the following context from previous conversations and new information:
#         Context:
#         {conversation_context}
#
#         New Information:
#         - Observed emotions from video analysis: {emotions_from_video}
#         - Detected emotions from audio analysis: {emotions_from_audio}
#         - Transcription of their spoken words: "{transcription}"
#
#         Provide a compassionate and thoughtful response or answer to the last spoken words.
#         Pay closer attention to spoken words than the detected emotions.
#         The response should be no longer than two sentences of spoken word and should only contain the spoken
#         sentence without additional commentary.
#         """
#     return prompt

def generate_prompt(transcription):
    prompt = f"""
    {transcription}
    """
    return prompt

def update_messages(messages, prompt, detected_emotions, previous_answer):
    if len(messages) == 0:
        messages = [
            {
                "role": "system",
                "content": "You are a psychologist answering questions and providing advice. Keep your response focused, no longer than two sentences, and ensure it addresses both the emotional and practical aspects."
            },
            {
                "role": "system",
                "content": f"detected emotions from audio and video stream: {', '.join(detected_emotions)}"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    else:
        messages.append(
            {
                "role": "assistant",
                "content": previous_answer
            }
        )
        messages.append({
            "role": "system",
            "content": f"detected emotions: {', '.join(detected_emotions)}"
        })
        messages.append({
            "role": "user",
            "content": prompt
        })
    return messages

def send_chat_completion(messages, model="llama-3.2-1b-instruct", temperature=0.7):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error sending prompt to LM server: {e}")
        return "I'm here for you."

# LLM Interaction
def feed_to_LLM(messages, emotions_from_video, emotions_from_audio, transcription):
    global previous_answer
    openai.api_base = "http://localhost:1234/v1"
    openai.api_key = "lm-studio"
    model = "llama-3.2-1b-instruct"

    prompt = generate_prompt(transcription)

    messages = update_messages(messages, prompt, emotions_from_audio + emotions_from_video, previous_answer)
    response = send_chat_completion(messages, model)
    previous_answer = response
    return response, prompt, messages

def pipeline_step(messages, video):
    def format_time():
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]
    def extract_audio_emotion_thread(audio_path, results):
        try:
            print("Processing audio emotions: ", format_time())
            results["audio_emotions"] = extract_emotion_from_audio(audio_path)
        finally:
            print("Audio processed: ", format_time())
            audio_emotion_event.set()  # Signal completion
    def extract_video_emotion_thread(video_path, results):
        try:
            print("Processing video emotions: ", format_time())
            results["video_emotions"] = extract_emotion_from_video(video_path)
        finally:
            print("Video processed: ", format_time())
            video_emotion_event.set()  # Signal completion
    def transcribe_audio_thread(audio_path, results):
        try:
            print("Processing text: ", format_time())
            results["transcription"] = get_transcription(audio_path)
        finally:
            print("Text processed: ", format_time())
            transcription_event.set()  # Signal completion

    video_path = video["video_path"]
    audio_output_path = video["audio_output_path"]

    print("Starting multithreaded processing...")

    results = defaultdict(dict)

    # Synchronization events
    audio_emotion_event = threading.Event()
    video_emotion_event = threading.Event()
    transcription_event = threading.Event()

    # Start threads for each processing task
    threading.Thread(target=extract_audio_emotion_thread, args=(audio_output_path, results), daemon=True).start()
    threading.Thread(target=extract_video_emotion_thread, args=(video_path, results), daemon=True).start()
    threading.Thread(target=transcribe_audio_thread, args=(audio_output_path, results), daemon=True).start()

    # Wait for all tasks to complete
    audio_emotion_event.wait()
    video_emotion_event.wait()
    transcription_event.wait()

    print("All tasks completed. Proceeding to LLM interaction...")

    # Retrieve results
    audio_emotions = results.get("audio_emotions", [])
    video_emotions = results.get("video_emotions", [])
    transcription = results.get("transcription", "")

    detected = [audio_emotions, video_emotions, transcription]

    # Interact with LLM
    response, prompt, messages = feed_to_LLM(messages, video_emotions, audio_emotions, transcription)
    for item in messages: print(item)
    print("LLM response generated!")
    return response, prompt, detected, messages


if __name__ == "__main__":
    # Main Workflow
    messages = []
    for video in videos:
        response, prompt, detected, messages = pipeline_step(messages, video)
        print(f"LLM")
        print(f"LLM Response:\n{response}")
        # conversation_context = update_conversation_context(conversation_context, response, prompt, detected)
