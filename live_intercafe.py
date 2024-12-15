import os.path
import cv2
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
from datetime import datetime, timedelta
import torch
import torchaudio
import ChatTTS
from IPython.display import Audio
from full_pipeline import pipeline_step, update_conversation_context

# Global Variables
is_recording = False
conversation_context = ""
video_filename = None
audio_filename = None
status_message = "Press S and start talking"  # Message to display on the OpenCV window
start_time = None  # Start time of the recording

# Recording Configuration
audio_sample_rate = 16000
audio_channels = 1
audio_dtype = 'int16'
recording_limit = 30  # Recording limit in seconds

# Global Variables
audio_recording_thread = None  # Thread for audio recording
video_recording_thread = None

text_to_speech_model = None

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
    global is_recording, status_message, video_filename, start_time
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam could not be opened.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        if is_recording:
            if out is None:
                try:
                    out = cv2.VideoWriter(video_filename, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
                    if not out.isOpened():
                        print("Error: VideoWriter could not be opened.")
                        break
                except Exception as e:
                    print(f"Error initializing VideoWriter: {e}")
                    break

            try:
                out.write(frame)
            except Exception as e:
                print(f"Error writing frame to VideoWriter: {e}")
                break

            elapsed_time = (datetime.now() - start_time).total_seconds()

            if elapsed_time >= recording_limit:
                stop_recording(out)
                out = None  # Reset VideoWriter object
                process_data_thread = threading.Thread(target=process_data, args=(video_filename, audio_filename))
                process_data_thread.start()

            # Display elapsed time and limit in the bottom-right corner
            time_text = f"{int(elapsed_time)}s/{recording_limit}s"
            cv2.putText(frame, time_text, (frame.shape[1] - 200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display status message on the video feed
        if status_message:
            cv2.putText(frame, status_message, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Webcam', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if is_recording:
                stop_recording(out)
                out = None  # Reset VideoWriter object
                process_data_thread = threading.Thread(target=process_data, args=(video_filename, audio_filename))
                process_data_thread.start()
            else:
                start_recording()
        elif key == ord('q'):
            break

    if out:
        try:
            out.release()
            print("VideoWriter released successfully.")
        except Exception as e:
            print(f"Error releasing VideoWriter: {e}")

    cap.release()
    cv2.destroyAllWindows()

def stop_recording(out):
    global is_recording, status_message, audio_recording_thread
    is_recording = False
    status_message = "Recording stopped, analyzing..."
    print("Recording stopped.")

    if out:
        try:
            out.release()  # Finalize video file
            print(f"Video file '{video_filename}' finalized successfully.")
        except Exception as e:
            print(f"Error releasing VideoWriter: {e}")

    # Wait for audio thread to finish
    if audio_recording_thread and audio_recording_thread.is_alive():
        audio_recording_thread.join()
        print("Audio recording thread finished.")

def process_data(video_filename, audio_filename):
    global status_message, conversation_context
    print("Processing the recorded video and audio...")
    video_data = {"video_path": video_filename, "audio_output_path": audio_filename}

    print(video_data)
    # Check if video file exists and is non-empty
    if not os.path.exists(video_filename) or os.path.getsize(video_filename) == 0:
        print(f"Error: Video file '{video_filename}' is missing or corrupted.")
        status_message = "Error: Video file is corrupted."
        return
    if not os.path.exists(audio_filename) or os.path.getsize(audio_filename) == 0:
        print(f"Error: Audio file '{audio_filename}' is missing or corrupted.")
        status_message = "Error: Audio file is corrupted."
        return

    try:
        response, prompt, detected = pipeline_step(conversation_context, video_data)
        status_message = ""
        print(f"LLM prompt: {prompt}")
        print(f"LLM response: {response}")
        conversation_context = update_conversation_context(conversation_context, response, prompt, detected)
        generate_audio(text_to_speech_model, response)
        print("Pipeline FINISHED!\n")
    except Exception as e:
        print(f"Error during processing: {e}")


def start_recording():
    global is_recording, video_filename, audio_filename, status_message, start_time, audio_recording_thread
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"recorded_video_{timestamp}.mp4"
    audio_filename = f"recorded_audio_{timestamp}.wav"
    initialize_audio_model()
    is_recording = True
    status_message = "Recording..."
    start_time = datetime.now()
    audio_recording_thread = threading.Thread(target=record_audio, args=(audio_filename,))
    audio_recording_thread.start()
    print("Recording started...")

def live_app():
    global is_recording, conversation_context, video_filename, audio_filename, status_message

    print("Initializing live app...")
    show_video()

def initialize_audio_model():
    """Initialize the ChatTTS model and return the loaded model instance."""
    global text_to_speech_model
    if text_to_speech_model is None:
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.suppress_errors = True
        torch.set_float32_matmul_precision('high')

        text_to_speech_model = ChatTTS.Chat()
        text_to_speech_model.load(compile=False)  # Set to True for better performance
    return text_to_speech_model

def generate_audio(text_to_speech_model, texts):
    """Generate and play audio from the provided text inputs using the ChatTTS model."""
    wavs = text_to_speech_model.infer(texts)
    for i, wav in enumerate(wavs):
        # Convert to numpy array and play audio
        audio_array = np.array(wav, dtype=np.float32)
        sd.play(audio_array, samplerate=24000)
        sd.wait()

if __name__ == "__main__":
    live_app()
