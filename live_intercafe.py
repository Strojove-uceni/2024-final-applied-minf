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
status_message = ""  # Message to display on the OpenCV window

# Audio Recording Configuration
audio_sample_rate = 16000
audio_channels = 1
audio_dtype = 'int16'

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
    except Exception as e:
        print(f"Error during audio recording: {e}")

def show_video():
    global is_recording, status_message, video_filename
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if is_recording:
            if out is None:
                # Ensure video writer is initialized only once during recording
                out = cv2.VideoWriter(video_filename, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
                if not out.isOpened():
                    print("Error: VideoWriter could not be opened.")
                    break
            out.write(frame)

        # Display status message on the video feed
        if status_message:
            cv2.putText(frame, status_message, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Webcam', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if is_recording:
                stop_recording(out)
                out = None  # Reset VideoWriter object
                process_data(video_filename, audio_filename)
            else:
                start_recording()
        elif key == ord('q'):
            break

    if out:
        out.release()
    cap.release()
    cv2.destroyAllWindows()

def stop_recording(out):
    global is_recording, status_message
    is_recording = False
    status_message = "Recording stopped, analyzing..."
    print("Recording stopped.")

    if out:
        out.release()  # Ensure video file is finalized properly

def process_data(video_filename, audio_filename):
    print("Processing the recorded video and audio...")
    video_data = {"video_path": video_filename, "audio_output_path": audio_filename}
    try:
        response = pipeline_step(conversation_context, video_data)
        status_message = ""
        print("Pipeline response:")
        print(response)
    except Exception as e:
        print(f"Error during processing: {e}")

def start_recording():
    global is_recording, video_filename, audio_filename, status_message
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"recorded_video_{timestamp}.mp4"
    audio_filename = f"recorded_audio_{timestamp}.wav"

    is_recording = True
    status_message = "Recording..."
    audio_thread = threading.Thread(target=record_audio, args=(audio_filename,))
    audio_thread.start()
    print("Recording started...")

def live_app():
    global is_recording, conversation_context, video_filename, audio_filename, status_message

    print("Initializing live app...")
    show_video()

if __name__ == "__main__":
    live_app()