from moviepy.editor import VideoFileClip
from scipy.io import wavfile

# Load the video and extract audio as WAV
video_path = '/home/martin/Coding/2024-final-applied-minf/really_sad_recording.mkv'
audio_output_path = '/home/martin/Coding/2024-final-applied-minf/really_sad_recording.wav'

# Extract audio from the video and save it as a WAV file
clip = VideoFileClip(video_path)
clip.audio.write_audiofile(audio_output_path)