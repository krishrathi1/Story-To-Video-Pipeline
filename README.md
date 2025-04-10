# Story-To-Video-Pipeline 🎥✨

An end-to-end AI pipeline that transforms your text prompt into a full animated story video using the power of Large Language Models (LLMs), Stable Diffusion, and text-to-speech narration.

---

## 🚀 Features

- ✅ **LLM-powered Story Generation** (using Mistral-7B)
- 🎨 **AI Image Generation** (via Stable Diffusion)
- 🎙️ **Text-to-Speech Narration** (gTTS)
- 🎬 **Video Creation** (MoviePy)

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/krishrathi1/Story-To-Video-Pipeline.git
cd Story-To-Video-Pipeline
```

### 2. Install Required Packages
```bash
pip install -r requirements.txt
```

Or install them manually:
```bash
pip install transformers accelerate bitsandbytes huggingface_hub diffusers scipy safetensors gTTS moviepy
```

### 3. Set Your Hugging Face Token
```python
from huggingface_hub import login
login("<your_huggingface_token>")
```

### 4. Run the Pipeline
Run the Python file to:
- Generate a story from your prompt
- Create image scenes for the story
- Generate narration
- Compile the video

```bash
python story_to_video.py
```

---

## 🧠 Model Info
- **LLM Model**: Mistral-7B-Instruct-v0.1 (via Hugging Face)
- **Image Model**: Stable Diffusion v1.5 (`runwayml/stable-diffusion-v1-5`)
- **TTS**: Google Text-to-Speech (`gTTS`)

---

## 📁 Output
- `story_video.mp4`: Final generated story video
- `story_images/`: AI-generated image scenes
- `narration.mp3`: Generated narration audio

---

## 📌 Credits
Developed with ❤️ by [Krish Rathi](https://github.com/krishrathi1)




