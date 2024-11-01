import numpy as np
from moviepy.editor import VideoFileClip
from scipy.io import wavfile
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, Wav2Vec2ForCTC, Wav2Vec2Processor, AutoConfig,Wav2Vec2FeatureExtractor, AutoModelForAudioClassification, AutoFeatureExtractor
import numpy as np
import librosa
import torch.nn.functional as F

video_path = '/home/martin/Coding/2024-final-applied-minf/angry_recording.mkv'
audio_path = '/home/martin/Coding/2024-final-applied-minf/angry_audio.wav'

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
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

result = pipe(audio_data, generate_kwargs={"language": "cs"})
print(result["text"])

# config = AutoConfig.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
# model = AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
# inputs = feature_extractor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
# with torch.no_grad():
#     logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
# predicted_class_ids = torch.argmax(logits, dim=-1)
# predicted_class_id = torch.mode(predicted_class_ids[predicted_class_ids != -1]).values.item()
# predicted_label = model.config.id2label.get(predicted_class_id, "Unknown")
# print(predicted_label)

label2id = {'angry': 0,
  'disgust': 1,
  'fearful': 2,
  'happy': 3,
  'neutral': 4,
  'sad': 5,
  'surprised': 6
}
id2label = {0: 'angry',
  1: 'disgust',
  2: 'fearful',
  3: 'happy',
  4: 'neutral',
  5: 'sad',
  6: 'surprised'
  }
model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"

model = AutoModelForAudioClassification.from_pretrained(
    model_id,     num_labels=7,
    label2id=label2id,
    id2label=id2label,
)
feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id, do_normalize=True,
)
pipe = pipeline(
    "audio-classification",
    model=model,
    tokenizer=feature_extractor,
    feature_extractor=feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

result = pipe(audio_data, generate_kwargs={"language": "en"})
print(result)

