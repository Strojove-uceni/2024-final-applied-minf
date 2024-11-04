import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import librosa

audio_path = '/home/martin/Coding/2024-final-applied-minf/test_data/sad_audio.wav'

target_sample_rate = 16000  # Define your target sample rate
audio_data, sample_rate = librosa.load(audio_path, sr=target_sample_rate)  # sr=None if you want original sample rate

# Print details about the loaded audio
print(f"Sample rate: {sample_rate}, Audio shape: {audio_data.shape}")
print(f"Audio duration (s): {audio_data.shape[0] / sample_rate}")

# Ensure the audio is in mono format (librosa loads mono by default REALLY??)
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

result = pipe(audio_data, generate_kwargs={"language": "en"})
print(result["text"])
