# 🎙️ Conversational Voice AI Assistant

A professional, real-time speech-to-speech AI assistant built with **Streamlit**, **Whisper**, and **Llama 3.1**. This assistant features a modern three-panel interface, retrieval-augmented generation (RAG) for expert knowledge, and persistent chat sessions.

---

## 🧠 How It Works (The AI Workflow)

Even if you are a beginner, here is how the assistant processes your voice step-by-step:

### 1. 🎤 Sound Capture (`sounddevice`)
When you click the microphone, the app uses **sounddevice** to "listen" to your computer's mic. It captures your speech as small chunks of digital data.

### 2. 🧹 Cleaning & Smart Silence (`noisereduce` & `VAD`)
- **Noise Suppression**: Using **noisereduce**, the app filters out background static (like your laptop fan) so the AI hears only your voice.
- **VAD (Voice Activity Detection)**: The app is smart! It monitors your voice level and automatically "stops" the recording as soon as it detects 2 seconds of silence. You don't even have to click "Stop"!

### 3. ✍️ Sound to Text (`OpenAI Whisper ASR`)
The cleaned audio is converted into actual written text by **Whisper**, a world-class **ASR (Automatic Speech Recognition)** engine.

### 4. 📚 Personal Knowledge Search (`RAG` using `FAISS`)
This is the **RAG (Retrieval Augmented Generation)** phase. The AI looks into your local `knowledge.txt` file using **FAISS** (a super-fast digital library) to find specific facts related to your question. This makes the AI "smarter" about your personal data.

### 5. 🧠 Thinking & Reasoning (`Llama 3.1 LLM`)
The text is sent to the **LLM (Large Language Model)**—**Llama 3.1** via the Grok API. The AI combines its general knowledge with your "personal library" data to write a concise response.

### 6. 🗣️ Text to Voice (`pyttsx3 TTS`)
Finally, the written response is sent to **pyttsx3**, a **TTS (Text-to-Speech)** engine. It acts like artificial vocal cords, turning the answer back into a human voice.

---

## 🌟 Key Features

### 🛠️ Advanced Voice Pipeline
- **Real-Time Capture**: High-quality audio recording with `sounddevice`.
- **Intelligent VAD**: Built-in Voice Activity Detection to automatically stop recording when you finish speaking.
- **Noise Suppression**: Clean audio processing for crystal-clear transcription.
- **Whisper ASR**: Industry-leading speech-to-text accuracy.
- **Groq-Powered LLM**: Blazing-fast reasoning using the Llama 3.1 8B model.
- **Seamless TTS**: Professional text-to-speech synthesis that plays back automatically.

### 🖥️ Modern Experience
- **3-Panel Dashboard**: 
  - **Left**: Chat Session History (New, Switch, and Delete chats).
  - **Center**: Chronological conversation history with native chat bubbles.
  - **Right**: Elegant Microphone controls with pulsing animations.
- **Smart Response Control**: Responses are optimized (40-50 words) unless you specifically ask for "briefly" detailed info.
- **AI Interruption**: Dedicated "Stop AI" button to pause the assistant mid-speech.

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.9+**
- **FFmpeg**: Must be installed and added to your System PATH (Required for Whisper audio processing).
  - *Windows*: `choco install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/).

### 2. Installation
Clone this repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Configuration
1. Obtain an API Key from **Groq Cloud**.
2. Create or edit the `.env` file in the root directory:
   ```env
   GROK_API_KEY=your_actual_api_key_here
   ```

### 4. Launch the App
Run the following command to start the assistant:
```bash
streamlit run app.py
```

---

## 📂 Project Structure
- `app.py`: The main engine containing the UI and Voice Pipeline.
- `knowledge.txt`: Source document for the RAG system (Vector DB).
- `requirements.txt`: Project dependencies and libraries.
- `.env`: Secure storage for API credentials.

---

## 🎮 How to Use
1. **Start a Session**: Click **New Chat** on the left.
2. **Talk to the AI**: Click the **🎙️ Mic** icon on the right. Speak clearly and stop—the AI will detect your silence automatically!
3. **Interrupt**: If the AI is talking too much, hit the **⏹ Stop AI** button.
4. **Manage History**: Switch between "Chat 1", "Chat 2", etc., or delete old sessions using the **🗑️ icon**.

---

## 🧪 Technology Stack
- **Frontend**: Streamlit
- **ASR**: OpenAI Whisper
- **LLM**: Meta Llama 3.1 (via Groq)
- **RAG**: FAISS & LangChain
- **TTS**: pyttsx3
- **Processing**: NumPy, noisereduce, SciPy

---
*Created for the Hackathon Demo - Seamless Voice Interaction.*
