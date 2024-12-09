import os.path

import cv2
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
from datetime import datetime
from full_pipeline import pipeline_step

# Global Variables
is_recording = False
conversation_context = ""
video_filename = None
audio_filename = None

# Audio Recording Configuration
audio_sample_rate = 16000
audio_channels = 1
audio_dtype = 'int16'

# Audio Recording Function
def record_audio(output_path):
    global is_recording, audio_frames
    audio_frames = []

    def audio_callback(indata, frames, time, status):
        if is_recording:
            audio_frames.append(indata.copy())

    try:
        with sd.InputStream(samplerate=audio_sample_rate, channels=audio_channels, dtype=audio_dtype, callback=audio_callback):
            while is_recording:
                sd.sleep(100)
        sf.write(output_path, np.vstack(audio_frames), samplerate=audio_sample_rate, subtype='PCM_16')
        print(f"Audio successfully saved to {output_path}")
    except Exception as e:
        print(f"Error during audio recording: {e}")

# Video Recording Function
def record_video(output_path):
    global is_recording
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Compatible codec for mp4
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

    while is_recording:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow('Recording', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Live App Interface
def live_app():
    global is_recording, conversation_context, video_filename, audio_filename
    print("Initializing live app...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"recorded_video_{timestamp}.mp4"
    audio_filename = f"recorded_audio_{timestamp}.wav"  # todo - save more recordings

    def start_recording():
        global is_recording
        is_recording = True
        video_thread = threading.Thread(target=record_video, args=(video_filename,))
        audio_thread = threading.Thread(target=record_audio, args=(audio_filename,))
        video_thread.start()
        audio_thread.start()
        print("Recording started...")

    def stop_recording():
        global is_recording
        is_recording = False
        print("Recording stopped.")
        print("Processing the recorded video and audio...")
        video_data = {"video_path": video_filename, "audio_output_path": audio_filename}
        try:
            import time
            time.sleep(1)
            print(os.path.exists(video_filename))
            print(os.path.exists(audio_filename))
            response = pipeline_step(conversation_context, video_data)
            print("Pipeline response:")
            print(response)
        except Exception as e:
            print(f"Error during processing: {e}")

    print("Press 's' to start recording, 'q' to stop and process, 'e' to exit.")
    while True:
        command = input("Enter your command: ").strip().lower()
        if command == 's':
            start_recording()
        elif command == 'q':
            stop_recording()
        elif command == 'e':
            print("Exiting the app.")
            break
        else:
            print("Invalid command. Please enter 's' to start, 'q' to stop, or 'e' to exit.")

# Run the app
if __name__ == "__main__":
    live_app()
