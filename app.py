import streamlit as st
import sounddevice as sd
import numpy as np
import noisereduce as nr
from scipy.io.wavfile import write
import time
import os
import queue
import whisper
from openai import OpenAI
from dotenv import load_dotenv
import pyttsx3

# Force reload environment variables
load_dotenv(override=True)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Voice Assistant",
    page_icon="🎙️",
    layout="wide"
)

# --- CLEAN UI THEME & MIC LOGO CSS ---
st.markdown("""
<style>
    /* Clean Base Theme */
    .stApp {
        background-color: #ffffff;
        color: #1f2937;
    }

    /* Hide the Audio Player Bar */
    audio {
        display: none !important;
        height: 0;
        width: 0;
        visibility: hidden;
    }

    /* Chat Container styling */
    [data-testid="stVerticalBlockBorderWrapper"] > div > div > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] > div > div.stContainer {
        height: 70vh !important;
        overflow-y: auto !important;
        padding: 20px !important;
        border-radius: 14px !important;
        background: #f9fafb !important;
        border: 1px solid #e5e7eb !important;
    }

    /* Round Mic Button Styling */
    .round-button > button {
        width: 70px !important;
        height: 70px !important;
        border-radius: 50% !important;
        font-size: 30px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 10px auto !important;
        background: #ffffff !important;
        border: 2px solid #e5e7eb !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    /* On/Recording State */
    .mic-on > button {
        background: #fee2e2 !important;
        border-color: #ef4444 !important;
        color: #ef4444 !important;
        animation: pulse 1.5s infinite;
    }

    /* Stop AI Voice Button Styling */
    .stop-voice > button {
        background: #f3f4f6 !important;
        border-color: #9ca3af !important;
        color: #374151 !important;
    }
    .stop-voice > button:hover {
        background: #e5e7eb !important;
        color: #000000 !important;
    }

    /* Delete Button Styling in Sidebar */
    .del-btn > button {
        background: transparent !important;
        border: none !important;
        color: #9ca3af !important;
        margin-top: 5px !important;
    }
    .del-btn > button:hover {
        color: #ef4444 !important;
        background: #fee2e2 !important;
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); box-shadow: 0 0 15px rgba(239, 68, 68, 0.4); }
        100% { transform: scale(1); }
    }

    /* Sidebar/Panel Headers */
    .stSubheader {
        color: #6b7280 !important;
        text-align: center;
        text-transform: uppercase;
        font-weight: bold;
        margin-top: 20px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIG ---
RESPONSE_AUDIO_PATH = "ai_response.wav"
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 30 
CHUNK_SIZE = int(SAMPLE_RATE * (CHUNK_DURATION_MS / 1000))
VAD_SILENCE_LIMIT_SEC = 2.0 
ENERGY_THRESHOLD = 0.015 
API_KEY = os.getenv("GROK_API_KEY")

# --- RESOURCES ---
def get_shared_queue():
    if "audio_q" not in st.session_state:
        st.session_state.audio_q = queue.Queue()
    return st.session_state.audio_q

audio_q = get_shared_queue()

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small")

@st.cache_resource
def setup_vector_db():
    try:
        from langchain_community.document_loaders import TextLoader
        from langchain_text_splitters import CharacterTextSplitter
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        if not os.path.exists("knowledge.txt"):
            with open("knowledge.txt", "w") as f:
                f.write("Artificial intelligence is machines simulating human intelligence.\n")
        loader = TextLoader("knowledge.txt")
        docs = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_documents(text_splitter.split_documents(docs), embeddings)
    except: return None

whisper_model = load_whisper_model()
vector_db = setup_vector_db()

# --- BACKEND ---
def audio_callback(indata, frames, time_info, status):
    audio_q.put(indata.copy())

def handle_recording():
    st.session_state.is_recording = True
    st.session_state.recorded_data = []
    st.session_state.silence_time = None
    while not audio_q.empty(): audio_q.get()
    
    stream = sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE, callback=audio_callback)
    with stream:
        while st.session_state.is_recording:
            try:
                chunk = audio_q.get(timeout=0.5)
                st.session_state.recorded_data.append(chunk)
                rms = np.sqrt(np.mean(chunk**2))
                if rms < ENERGY_THRESHOLD:
                    if st.session_state.silence_time is None: st.session_state.silence_time = time.time()
                    elif time.time() - st.session_state.silence_time > VAD_SILENCE_LIMIT_SEC:
                        st.session_state.is_recording = False
                else: st.session_state.silence_time = None
            except queue.Empty: break
    return np.concatenate(st.session_state.recorded_data, axis=0)

def run_pipeline(raw_audio):
    try:
        audio_flat = raw_audio.flatten()
        clean_audio = nr.reduce_noise(y=audio_flat, sr=SAMPLE_RATE, prop_decrease=0.8)
        with st.status("Thinking...", expanded=False):
            audio_for_whisper = clean_audio.astype(np.float32) / (np.max(np.abs(clean_audio)) + 1e-6)
            transcript = whisper_model.transcribe(audio_for_whisper, fp16=False, language="en")["text"].strip()
            if not transcript: return
            
            # Store message in session history
            st.session_state.sessions[st.session_state.current_session].append({"role": "user", "content": transcript})
            
            client = OpenAI(api_key=API_KEY, base_url="https://api.groq.com/openai/v1")
            limit_prompt = "Keep your answer concise, strictly between 40 to 50 words."
            if "briefly" in transcript.lower():
                limit_prompt = "Provide a detailed answer."
            
            messages = [{"role": "system", "content": f"You are a helpful assistant. {limit_prompt}"}] + st.session_state.sessions[st.session_state.current_session]
            answer = client.chat.completions.create(model="llama-3.1-8b-instant", messages=messages).choices[0].message.content
            
            # Store AI response in session history
            st.session_state.sessions[st.session_state.current_session].append({"role": "assistant", "content": answer})
            
            # Reset stop flag for new speech
            st.session_state.stop_speaking = False
            
            # TTS Background execution
            engine = pyttsx3.init()
            engine.save_to_file(answer, RESPONSE_AUDIO_PATH)
            engine.runAndWait()
            
            st.session_state.play_audio = True
    except Exception as e:
        st.error(f"Error: {e}")

# --- MAIN UI ---
def main():
    # Initialize session storage
    if "sessions" not in st.session_state:
        st.session_state.sessions = {"Chat 1": []}

    if "current_session" not in st.session_state:
        st.session_state.current_session = "Chat 1"
    
    if "stop_speaking" not in st.session_state:
        st.session_state.stop_speaking = False

    if "play_audio" not in st.session_state: st.session_state.play_audio = False

    session_col, chat_col, control_col = st.columns([1.2, 4, 1.2])

    with session_col:
        st.subheader("Session")
        # New Chat button
        if st.button("🗑 New Chat"):
            # Find a unique title
            i = 1
            while f"Chat {i}" in st.session_state.sessions:
                i += 1
            new_id = f"Chat {i}"
            st.session_state.sessions[new_id] = []
            st.session_state.current_session = new_id
            st.session_state.play_audio = False
            st.session_state.stop_speaking = True
            st.rerun()

        # Clear active chat
        if st.button("🧹 Clear Chat"):
            st.session_state.sessions[st.session_state.current_session] = []
            st.session_state.stop_speaking = True
            st.rerun()

        st.markdown("---")
        # Display chat list with delete icon
        chats = list(st.session_state.sessions.keys())
        for chat in chats:
            col1, col2 = st.columns([4, 1.2])
            with col1:
                # Highlight active session
                btn_label = f"💬 {chat}" if chat == st.session_state.current_session else chat
                if st.button(btn_label, key=f"btn_{chat}", use_container_width=True):
                    st.session_state.current_session = chat
                    st.session_state.stop_speaking = True
                    st.rerun()
            with col2:
                # Step: Add delete icon button
                if st.button("🗑️", key=f"del_{chat}", help=f"Delete {chat}"):
                    del st.session_state.sessions[chat]
                    # If we deleted the current active chat, switch to another one
                    if st.session_state.current_session == chat:
                        remaining_chats = list(st.session_state.sessions.keys())
                        if remaining_chats:
                            st.session_state.current_session = remaining_chats[0]
                        else:
                            # Re-initialize to Chat 1 if all are gone
                            st.session_state.current_session = "Chat 1"
                            st.session_state.sessions["Chat 1"] = []
                    st.session_state.stop_speaking = True
                    st.rerun()

    with chat_col:
        st.title(f"🎙️ {st.session_state.current_session}")
        
        # Render conversation from the active session
        messages = st.session_state.sessions.get(st.session_state.current_session, [])
        with st.container(height=500):
            for msg in messages:
                st.chat_message(msg["role"]).write(msg["content"])
        
        # Audio playback (Invisible)
        if st.session_state.play_audio and not st.session_state.stop_speaking:
            st.audio(RESPONSE_AUDIO_PATH, format="audio/wav", autoplay=True)

    with control_col:
        # MIC ON (SPEAK)
        st.subheader("Voice On")
        if not st.session_state.get('is_recording', False):
            st.markdown('<div class="round-button">', unsafe_allow_html=True)
            if st.button("🎙️", key="mic_ready"):
                st.session_state.stop_speaking = False 
                raw_audio = handle_recording()
                run_pipeline(raw_audio)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption("<p style='text-align:center;'>Start Talking</p>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="round-button mic-on">', unsafe_allow_html=True)
            if st.button("🎙️", key="mic_recording"):
                st.session_state.is_recording = False
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption("<p style='text-align:center; color:red;'>Recording...</p>", unsafe_allow_html=True)

        # STOP AI (INTERRUPT SPEAKING)
        st.subheader("Stop AI")
        st.markdown('<div class="round-button stop-voice">', unsafe_allow_html=True)
        if st.button("⏹ Stop AI", key="stop_ai_voice"):
            st.session_state.stop_speaking = True
            st.session_state.play_audio = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("<p style='text-align:center;'>Stop AI Speaking</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
