import numpy as np
from moviepy.editor import VideoFileClip
import torch
import numpy as np
import librosa
from transformers import AutoImageProcessor, AutoModelForImageClassification

video_path = '/home/martin/Coding/2024-final-applied-minf/test_data/sad_recording.mkv'
audio_path = '/home/martin/Coding/2024-final-applied-minf/test_data/sad_audio.wav'

# Load video and prepare to extract frames
video = VideoFileClip(video_path)

frames = []

for frame in video.iter_frames(fps=video.fps, dtype="uint8"):
    frames.append(frame)  # Each frame is an RGB array

# Convert list of frames to a 4D NumPy array (num_frames, height, width, channels)
frames_array = np.array(frames)
print(f"Total frames: {frames_array.shape[0]}, Frames shape: {frames_array.shape}")
target_sample_rate = 16000  # Define your target sample rate
audio_data, sample_rate = librosa.load(audio_path, sr=target_sample_rate)  # sr=None if you want original sample rate

# Print details about the loaded audio
print(f"Sample rate: {sample_rate}, Audio shape: {audio_data.shape}")
print(f"Audio duration (s): {audio_data.shape[0] / sample_rate}")

# Ensure the audio is in mono format (librosa loads mono by default)
if len(audio_data.shape) > 1:
    audio_data = np.mean(audio_data, axis=1)  # Convert to mono by averaging channels
device = "cuda:0" if torch.cuda.is_available() else "cpu"

processor = AutoImageProcessor.from_pretrained("trpakov/vit-face-expression")
model = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression")

for i in range(1, 10):
    frame_num = len(frames)*i/10 - 1 
    inputs = processor(frame, return_tensors="pt")
    logits = model(**inputs).logits
    predicted_label = torch.argmax(logits, dim=-1).int()
    print(model.config.id2label[predicted_label.item()])