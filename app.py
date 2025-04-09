import os
import wave
import threading
import numpy as np
import pyaudio
import concurrent.futures
import pandas as pd
import speech_recognition as sr
from flask import Flask, render_template, jsonify, request
from pyannote.audio import Pipeline
from dotenv import load_dotenv
import openai
import math
import time

# Load env and API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Flask app
app = Flask(__name__)

# Globals
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 3
recording = False
chunk_index = 1
recorded_audio = np.array([], dtype=np.int16)
result_df = pd.DataFrame(columns=["fileId", "speaker", "utterance"])
executor = concurrent.futures.ThreadPoolExecutor()
futures_list = []
selected_language = "en-IN"
summary_ready = False
app.config["SUMMARY"] = {"summary": "", "key_points": "", "actions": ""}

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token="token"
)

def save_wav(filename, data):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(data.tobytes())

def rttm_to_dataframe(rttm_file_path):
    columns = ["type", "fileId", "channel", "start time", "duration", "orthology", "confidence", "speaker", 'x', 'y']
    with open(rttm_file_path, "r") as rttm_file:
        lines = rttm_file.readlines()
        data = [line.strip().split() for line in lines]
        df = pd.DataFrame(data, columns=columns)
        df = df.drop(['x', 'y', "orthology", "confidence", "type", "channel"], axis=1)
        return df

def extract_text_from_audio(audio_file_path, start_time, end_time):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)
        start_ms = int(start_time * 1000)
        end_ms = int((end_time + 0.2) * 1000)
        segment = audio.get_segment(start_ms, end_ms)
        try:
            return recognizer.recognize_google(segment, language=selected_language)
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            return f"API error: {e}"

def process_rttm_and_transcribe(rttm_file_path, audio_file_path):
    global result_df
    df = rttm_to_dataframe(rttm_file_path)
    df = df.astype({'start time': 'float', 'duration': 'float'})
    df['end time'] = df['start time'] + df['duration']

    df['utterance'] = df.apply(
        lambda row: extract_text_from_audio(audio_file_path, row['start time'], row['end time']),
        axis=1
    )

    grouped = []
    prev_speaker = None
    current_text = ""

    for _, row in df.iterrows():
        if not row['utterance']:
            continue
        if row['speaker'] == prev_speaker:
            current_text += " " + row['utterance']
        else:
            if prev_speaker is not None and current_text:
                grouped.append((audio_file_path, prev_speaker, current_text.strip()))
            prev_speaker = row['speaker']
            current_text = row['utterance']

    if prev_speaker and current_text:
        grouped.append((audio_file_path, prev_speaker, current_text.strip()))

    result_df = pd.DataFrame(grouped, columns=["fileId", "speaker", "utterance"])

def process_chunk(file, audio_file_path):
    print(f"Processing {file} for diarization...")
    diarization = pipeline(file)
    with open("audio.rttm", "w") as rttm:
        diarization.write_rttm(rttm)
    print(f"Overwritten audio.rttm with {file}")
    process_rttm_and_transcribe("audio.rttm", audio_file_path)
    
def final_processing_and_summary(file):
    global summary_ready, result_df
    process_chunk(file, file)

    chunk_name = os.path.splitext(file)[0]
    rttm_path = "audio.rttm"

    while True:
        if os.path.exists(rttm_path):
            with open(rttm_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 1 and parts[1] == chunk_name:
                        print("✅ Final chunk found in RTTM.")
                        break
                else:
                    print(f"⏳ Waiting for {chunk_name} to be included in {rttm_path}...")
                    time.sleep(2)
                    continue
                break
        else:
            time.sleep(1)

    full_text = " ".join(result_df['utterance'].dropna())
    if full_text.strip():
        prompt = f"""
        You are a helpful assistant. Please read the following meeting transcript and return the following:

        1. A brief summary of the conversation 
        2. Key discussion points (as bullet points)
        3. Action items (as bullet points)

        Transcript:
        {full_text}

        Format your response as:
        Summary: ...
        Key Points:
        - ...
        Actions:
        - ...
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3
            )
            content = response['choices'][0]['message']['content']
            summary_part = content.split("Key Points:")[0].replace("Summary:", "").strip()
            keypoints_part = content.split("Key Points:")[1].split("Actions:")[0].strip()
            actions_part = content.split("Actions:")[1].strip()

            app.config["SUMMARY"] = {
                "summary": summary_part,
                "key_points": keypoints_part,
                "actions": actions_part
            }
            summary_ready = True
            print("✅ Summary generation complete.")
        except Exception as e:
            print(f"❌ Error during summary generation: {e}")
            app.config["SUMMARY"] = {"summary": "", "key_points": "", "actions": ""}
            summary_ready = True
    else:
        print("⚠️ Empty transcript. No summary generated.")
        app.config["SUMMARY"] = {"summary": "", "key_points": "", "actions": ""}
        summary_ready = True

def record_audio():
    global recording, chunk_index, recorded_audio, futures_list
    FORMAT = pyaudio.paInt16
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=SAMPLE_RATE)

    chunk_index = 1
    recorded_audio = np.array([], dtype=np.int16)
    print("Recording started...")

    while recording:
        frames = stream.read(CHUNK_DURATION * SAMPLE_RATE, exception_on_overflow=False)
        chunk = np.frombuffer(frames, dtype=np.int16)
        recorded_audio = np.concatenate((recorded_audio, chunk))

        filename = f"chunk_{chunk_index}.wav"
        save_wav(filename, recorded_audio)
        print(f"Saved: {filename}")

        if int(math.sqrt(chunk_index))**2 == chunk_index:
            future = executor.submit(process_chunk, filename, filename)
            futures_list.append(future)

        chunk_index += 1

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Recording stopped.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start-recording')
def start_recording():
    global recording, selected_language
    selected_language = request.args.get("lang", "en-IN")
    recording = True
    threading.Thread(target=record_audio).start()
    return jsonify({"status": "Recording started", "language": selected_language})

@app.route('/stop-recording')
def stop_recording():
    global recording, chunk_index, recorded_audio, futures_list
    recording = False
    if chunk_index!=int(math.sqrt(chunk_index))**2:
        last_chunk = chunk_index-1
    else:
        last_chunk=chunk_index
    filename = f"chunk_{last_chunk}.wav"
    print(f"Stopping recording. Last chunk: {filename}")

    if not os.path.exists(filename):
        print(f"Saving final chunk: {filename}")
        chunk_samples = CHUNK_DURATION * SAMPLE_RATE
        chunk = recorded_audio[-chunk_samples:]
        save_wav(filename, chunk)

    for future in futures_list:
        if not future.done():
            future.cancel()
    futures_list = []

    print(f"Submitting final chunk {filename} for diarization and summary.")
    threading.Thread(target=final_processing_and_summary, args=(filename,)).start()

    return jsonify({"status": f"Recording stopped. Final chunk {filename} handled."})

@app.route('/get-transcript')
def get_transcript():
    global result_df
    return result_df.to_json(orient="records")

@app.route('/summary-ready')
def check_summary_ready():
    print("📡 /summary-ready called —", "✅ Ready" if summary_ready else "⏳ Not ready yet")
    return jsonify({"ready": summary_ready})

@app.route('/clear', methods=['POST'])
def clear_files():
    global result_df, summary_ready
    result_df = pd.DataFrame(columns=['fileId', 'speaker', 'utterance'])
    summary_ready = False
    app.config["SUMMARY"] = {"summary": "", "key_points": "", "actions": ""}

    for file in os.listdir():
        if file.endswith(".wav") or file.endswith(".rttm"):
            os.remove(file)

    return "All audio and transcript files deleted."

@app.route('/get-summary')
def get_summary():
    return jsonify(app.config.get("SUMMARY", {
        "summary": "",
        "key_points": "",
        "actions": ""
    }))

if __name__ == '__main__':
    app.run(debug=True)
