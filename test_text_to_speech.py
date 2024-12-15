# Import necessary libraries and configure settings
import torch
import torchaudio
# import sys
# sys.path.append("/home/martin/Coding/2024-final-applied-minf/ChatTTS")

# from ChatTTS import ChatTTS
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

import ChatTTS
from IPython.display import Audio

# Initialize and load the model: 
chat = ChatTTS.Chat()
chat.load(compile=False) # Set to True for better performance <--- FAKE NEWS

# Define the text input for inference (Support Batching)
texts = [
    "So we found being competitive and collaborative was a huge way of staying motivated towards our goals, so one person to call when you fall off, one person who gets you back on then one person to actually do the activity with.",
    ]

# Perform inference and play the generated audio
wavs = chat.infer(texts)
for i in range(len(wavs)):
    """
    In some versions of torchaudio, the first line works but in other versions, so does the second line.
    """
    try:
        torchaudio.save(f"basic_output{i}.wav", torch.from_numpy(wavs[i]).unsqueeze(0), 24000)
    except:
        torchaudio.save(f"basic_output{i}.wav", torch.from_numpy(wavs[i]), 24000)