# Conversational Voice AI Assistant

A real-time speech-to-speech conversational assistant built using Streamlit, Whisper ASR, and Llama 3.1.
The system captures live voice input, converts speech to text, retrieves relevant knowledge using Retrieval Augmented Generation (RAG), generates an intelligent response using a Large Language Model, and converts the response back into natural speech.

This project demonstrates a complete end-to-end conversational AI pipeline.

## System Architecture

The assistant follows a modular speech-to-speech pipeline. Each component performs a dedicated role in transforming raw voice input into an intelligent spoken response.

### High-Level Processing Pipeline
User Speech
    │
    ▼
Microphone Capture
(sounddevice)
    │
    ▼
Audio Preprocessing
Noise Suppression
(noisereduce)
    │
    ▼
Voice Activity Detection
Silence Detection Logic
    │
    ▼
Automatic Speech Recognition
Whisper ASR
    │
    ▼
Query Processing
Prompt Structuring
    │
    ▼
Knowledge Retrieval
RAG Layer
FAISS + LangChain
    │
    ▼
Reasoning Engine
Llama 3.1 LLM
(Groq API)
    │
    ▼
Response Generation
Structured Output
    │
    ▼
Speech Synthesis
pyttsx3 TTS
    │
    ▼
Audio Playback
AI Spoken Response

## How the System Works
### 1. Microphone Input

The application captures real-time audio using the `sounddevice` library.
Audio is streamed as small chunks of digital data to enable responsive processing.

**Responsibilities**
- Capture microphone input
- Stream audio frames
- Buffer incoming audio data

### 2. Audio Processing

Before sending the audio to the speech recognition engine, the signal is cleaned and optimized.

**Noise Suppression**
Background noise is reduced using the `noisereduce` library to improve transcription quality.

**Voice Activity Detection**
Silence detection logic automatically stops recording once the user finishes speaking.

**Responsibilities**
- Remove background noise
- Detect speech boundaries
- Prepare audio for ASR processing

### 3. Automatic Speech Recognition (ASR)

The cleaned audio signal is converted into text using OpenAI Whisper.

Whisper is a deep learning based ASR model capable of accurate speech recognition across various environments.

**Example output**
```json
{
  "transcription": "Explain artificial intelligence"
}
```

**Responsibilities**
- Speech to text conversion
- Sentence segmentation
- Accurate transcription

### 4. Knowledge Retrieval (RAG)

The assistant uses Retrieval Augmented Generation to enhance its responses using domain-specific knowledge.

The system searches a local knowledge base (`knowledge.txt`) using FAISS, a fast vector similarity search engine.

**Process**
User Query
      │
      ▼
Vector Search
      │
      ▼
Retrieve Relevant Knowledge
      │
      ▼
Send Context to LLM

**Responsibilities**
- Embed text into vector representations
- Perform semantic similarity search
- Retrieve relevant contextual information

### 5. Reasoning Layer

The user query and retrieved knowledge are passed to the Llama 3.1 Large Language Model through the Groq API.

The LLM performs reasoning and generates a contextual response based on both
- General knowledge
- Retrieved knowledge context

**Responsibilities**
- Intent understanding
- Contextual reasoning
- Natural language response generation

### 6. Text to Speech (TTS)

The generated text response is converted back into audio using the `pyttsx3` text-to-speech engine.

**Responsibilities**
- Convert text response to audio
- Synthesize natural speech
- Playback response to the user

## End-to-End Conversational Loop

The assistant operates as a continuous conversational system.

User Speech
      │
      ▼
Speech Recognition
      │
      ▼
Knowledge Retrieval
      │
      ▼
LLM Reasoning
      │
      ▼
Response Generation
      │
      ▼
Text-to-Speech
      │
      ▼
AI Spoken Response

This architecture enables real-time speech-to-speech interaction.

## Key Features
### Voice Pipeline
- Real-time audio capture using `sounddevice`
- Noise suppression for improved speech clarity
- Voice activity detection for automatic recording control
- Whisper based speech recognition
- Fast Llama 3.1 reasoning using Groq API
- Text to speech synthesis using `pyttsx3`

### Conversational Interface
- Multi-session chat management
- Persistent conversation history
- Scrollable conversation window
- Dedicated microphone controls
- Ability to interrupt AI speech

### Retrieval Augmented Generation
- Custom knowledge base using `knowledge.txt`
- Vector search using FAISS
- Context-aware responses using LangChain

## Technology Stack

### Frontend
- Streamlit

### Speech Recognition
- OpenAI Whisper

### Language Model
- Llama 3.1 via Groq API

### Retrieval System
- FAISS
- LangChain

### Speech Synthesis
- `pyttsx3`

### Audio Processing
- NumPy
- SciPy
- `noisereduce`

## Project Structure
```text
AweTails_Ai-Voice-Tech
│
├── app.py
├── knowledge.txt
├── requirements.txt
├── .env
└── README.md
```

**app.py**
Main application containing the voice pipeline and user interface.

**knowledge.txt**
Source document used by the RAG system.

**requirements.txt**
List of project dependencies.

**.env**
Stores API credentials.

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/Raghuram1784/AweTails_Ai-Voice-Tech.git
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Configuration

**Create a .env file in the project root directory**
```env
GROK_API_KEY=your_api_key_here
```

## Running the Application

**Start the Streamlit server**
```bash
streamlit run app.py
```
The application will launch automatically in your browser.

## Usage

1. Create a new chat session
2. Click the microphone button
3. Speak your query clearly
4. The system automatically detects silence and stops recording
5. Speech is converted into text and processed by the AI model
6. The assistant generates a response and converts it back into speech
