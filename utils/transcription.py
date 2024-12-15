import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa

# Define the path to the video file
video_path = "/videos/So Sorry.mp4"

# Load the first 20 seconds of audio from the video file
try:
    duration = 20  # Duration in seconds to load
    audio_data, sample_rate = librosa.load(video_path, sr=16000, mono=True, offset=0.0, duration=duration)
    print("Audio loaded successfully.")
except Exception as e:
    print(f"An error occurred while loading the audio: {e}")

# Print details about the loaded audio
print(f"Sample rate: {sample_rate}, Audio shape: {audio_data.shape}")
print(f"Audio duration (s): {audio_data.shape[0] / sample_rate}")

# Ensure the audio is in mono format (librosa loads mono by default)
if len(audio_data.shape) > 1:
    audio_data = np.mean(audio_data, axis=1)  # Convert to mono by averaging channels

# Set up the device and model precision
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the model and processor
print("init model")
model_id = "openai/whisper-small"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

print("init processor")
processor = AutoProcessor.from_pretrained(model_id)

print('pipeline')
# Create the speech recognition pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

print('finish pipeline')
# Perform speech recognition on the audio
result = pipe(audio_data, generate_kwargs={"language": "en"})
print(result["text"])
