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
from full_pipeline import pipeline_step
import tkinter as tk
from tkinter import messagebox

# Global Variables
is_recording = False
messages = []
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
params_infer_code = None

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
    show_instructions = True  # Flag to control the display of instructions
    # Instruction message to display
    instructions = [
        "Welcome to AI Therapy!",
        "Please ensure the following:",
        "1. Quiet environment",
        "2. Good lighting",
        "3. Stable device position",
        "4. Privacy for comfort"
    ]
    initialize_audio_model()

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
            text_x = frame.shape[1] - 200
            text_y = frame.shape[0] - 20
            cv2.putText(frame, time_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display status message on the video feed
        if show_instructions:
            overlay = frame.copy()
            alpha = 0.6  # Transparency factor
            cv2.rectangle(overlay, (10, 10), (500, 200), (0, 0, 0), -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            y_start = 40
            for i, line in enumerate(instructions):
                font_scale = 0.8 if i == 0 else 0.7
                font_thickness = 2 if i == 0 else 1
                color = (255, 255, 255) if i == 0 else (200, 200, 200)
                cv2.putText(frame, line, (20, y_start + i * 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv2.LINE_AA)

        if status_message:
            overlay = frame.copy()
            alpha = 0.6  # Transparency for the status message background
            text_size = cv2.getTextSize(status_message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = 10
            text_y = frame.shape[0] - 20
            background_end_x = text_x + text_size[0] + 20
            background_start_y = text_y - text_size[1] - 10
            background_end_y = text_y + 10
            cv2.rectangle(overlay, (text_x, background_start_y), (background_end_x, background_end_y), (0, 0, 0),
                          -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.putText(frame, status_message, (text_x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        2, cv2.LINE_AA)
        cv2.imshow('Webcam', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if is_recording:
                stop_recording(out)
                out = None  # Reset VideoWriter object
                process_data_thread = threading.Thread(target=process_data, args=(video_filename, audio_filename))
                process_data_thread.start()
            else:
                if show_instructions:
                    show_instructions = False  # Disable instructions after the first "S" press
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
    global status_message, conversation_context, messages
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
        response, prompt, detected, messages = pipeline_step(messages, video_data)
        print(f"LLM prompt: {prompt}")
        print(f"LLM response: {response}")

        status_message = "Getting voice response..."
        generate_audio(text_to_speech_model, response+"[uv_break]")

        status_message = "Press S to continue recording"
        print("Pipeline FINISHED!\n")
    except Exception as e:
        print(f"Error during processing: {e}")


def start_recording():
    global is_recording, video_filename, audio_filename, status_message, start_time, audio_recording_thread
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"recorded_video_{timestamp}.mp4"
    audio_filename = f"recorded_audio_{timestamp}.wav"
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
    global text_to_speech_model, params_infer_code
    if text_to_speech_model is None:
        
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.suppress_errors = True
        torch.set_float32_matmul_precision('high')

        text_to_speech_model = ChatTTS.Chat()
        text_to_speech_model.load(compile=False)  # Set to True for better performance
        # rand_spk = text_to_speech_model.sample_random_speaker()
        rand_spk = "蘁淰救欀嚤廝玄敹撑纋窯娺楛砨芨舓蔤礈磮仃乶磍珔緒瓳蘷旪藫促賅膅浠妁廖双测職茧籙氁塙蜺疽噍煏浏熊牂槿啸澼蟎憊甏睢僬稲嘘僛祜君嬐琶癷暩叩嗅硰俦煋廞坐壞臤完堣倳幨彮玠瓹蒈趒慆夁膰淍偿掺晐妹琷簝贎旇圔塈泐孯匬嶅豦欨腿莒蒵嗁榁绤卒皾漄欢呏糤澡粖橉濫歈舘萳刵咔瑽珔妜蠻壑濝滤寐俶膜缅巛噧卬浉諭趝范挌葔潓埚虚刅庸烓専糴衼超侈萢慸惃癭笈珔媚衃藳虒縍幯蜂纷細瞋撊囚欋懛揎橻磂涭癃壷慸瘼僘赘仚贒嬹棨悂巵葭墴疺虈缈燻巗碢谚枘膚蛖痧忲篵忼虎慰蔏橑話悫瑹秩胕乄摾嚼嗀肍兔凹澿枑乥姖杊编羆瘆蓴岌漨监莯暇炶挹掉檏圙拡澪纴膐巍菎暾渔攣稵蓏萠炇但賌傱秱檷朗癸蔛樖檷笆烡扺芗勧偨賹謩姒潴癸穧扵恵愽座糧曋慩粂肸奙壥嗢咹识湛犜绶勬趼摱妟拫縒暕粉湇宮仃攏仝堣劔嗷炆粱堒厈硺奩呌望耋狾摹曥欃蕤性擢琣諃張氃舃援质羀裬痗蕑獣贸湀綡牼寫痨狥峩傽蝗勼狹篧秧岛塦紉捎尦譶燖謷桴稻哵溰挌瞀夰墷廟豯槦嬊摻杗省扵琶胷葈瘉汀螱涣桤诟唺汦秇時詨臭姈录跧祍拁埈灞猜枪貸莹贁纪殫谓竾名玎贕朽槣聢琭安勱繤媼勶墅姰瑑询崺碅梅忭扎灕摿坺喜謚蚊態民贒芒柍蚿暉掮婈媁碥憼秱烈榅殰箒煄桞纏滔苕傍史硒笤畷征嶬抳虀脩絮榋殊窪岂呚内劗爆盡棐糾栀蛦疰谗憏傕牣穩椱勝噒咱犉烁跃渍倩蚰崁吢若育垥緌丟粯衈泷塆弍畮癔什娪玺卉蝙皡偫貝俋樱萢俜縥渞炾栰翪稕旧嚎戲蒨崳虀倻垉沒敮栄胱繆蠇惓蒌美菮喆丶縱秲総旲友墶謮覧芡編嘿岵烹晪桀傲禄姸搴品旨烗堭娗惀弘岆士嬅烗病噅覘縙窅繋洲丄挏焛咐瀕瑣胳趫窣窴噊櫱袒蠯弥楷溢肛栲胊畔夅蟸楪巔篇簎厱孋甩牿壿籮舭豕蒫彧絰罡袔艨懻悞粐蛼疲訅搗柒冚琏丸脒桃翽絺衯蛿螃趭冮瓕孉懎柂覺仓氮欃啒籛歟潜弸歋汯椹哈揤淪哔岼臋傁暲妳剜琒墆溕剆蠸妹弦握嘑綶袥赜補烅跧書獇嘝嗃崅娶乒欥昴豂机的幔夐亝奓谻峖炁傍笺嫗哪欠卢柟籷暕窣膲朥瞷洣譿僋氝嚪汔綟硎牷昢殁滯屷姩拳伝牡蚗笿乏甲妺窮苤竿穫秌葕糔減桁夈穎檀甘綰瘠蔩浨敤癜屏蘧狴蛘思憘衚繓廂虶芐湍穫俘嘈窘伕萶厩編嶵殞姜礂肋娗喃筚牎痌羦皼葠栩枒唶虣袲脒臽娺膽縒盙忔耻啙篝硃办摘欌劷梃呣浶瀝曗埢毘紒蜹藀张嶝梱份螾妚夕千觕峜螧夠捉禵卬枏茰汇紂栀缣礓氃褛幪琭槛晰縚泳褸亙繘刀一㴂"
        # print(rand_spk) # save it for later timbre recovery
        params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb = rand_spk, # add sampled speaker 
            temperature = .3,   # using custom temperature
            top_P = 0.7,        # top P decode
            top_K = 20,         # top K decode
        )

    return text_to_speech_model

def show_response_popup(response_text):
    # Create a new Tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Display the popup message box with the response text
    messagebox.showinfo("LLM Response", response_text)

    # Destroy the root window after the popup is closed
    root.destroy()

def generate_audio(text_to_speech_model, texts):
    """Generate and play audio from the provided text inputs using the ChatTTS model."""
    # Display a popup window with the response text

    wavs = text_to_speech_model.infer(texts, params_infer_code=params_infer_code)

    threading.Thread(target=show_response_popup, args=(texts.split("[")[0],)).start()

    for i, wav in enumerate(wavs):
        # Convert to numpy array and play audio
        audio_array = np.array(wav, dtype=np.float32)
        sd.play(audio_array, samplerate=24000)
        sd.wait()

if __name__ == "__main__":
    live_app()
