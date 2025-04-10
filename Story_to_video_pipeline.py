# 📦 Install Required Packages
!pip install -q transformers accelerate bitsandbytes huggingface_hub diffusers scipy safetensors gTTS moviepy

# ✅ Hugging Face Login
from huggingface_hub import login
login("") #use your token here

# ✅ Load Mistral Model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True,
)

# ✅ Generate Story
def generate_story(prompt, max_tokens=500):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

prompt = "Write a short fantasy story about a boy who discovers a magical forest."
story = generate_story(prompt)
print("\n📖 Story:\n", story)

# ✅ Extract Image Prompts
def extract_image_prompts(story_text):
    lines = story_text.split('.')
    prompts = [line.strip() for line in lines if len(line.strip()) > 20]
    return prompts

image_prompts = extract_image_prompts(story)
print("\n🎨 Image Prompts:\n")
for i, p in enumerate(image_prompts[:5], 1):
    print(f"{i}. {p}")

# ✅ Load Stable Diffusion
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# ✅ Generate Images
import os
os.makedirs("story_images", exist_ok=True)

image_paths = []
for i, prompt in enumerate(image_prompts[:5]):
    image = pipe(prompt).images[0]
    path = f"story_images/scene_{i}.png"
    image.save(path)
    image_paths.append(path)

# ✅ Voice-over with gTTS
from gtts import gTTS

narration_text = " ".join(image_prompts[:5])
tts = gTTS(narration_text)
audio_path = "narration.mp3"
tts.save(audio_path)

# ✅ Combine into Video
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

audio = AudioFileClip(audio_path)
duration = audio.duration / len(image_paths)

clips = [
    ImageClip(img).set_duration(duration).resize(width=720)
    for img in image_paths
]

video = concatenate_videoclips(clips, method="compose").set_audio(audio)
video.write_videofile("story_video.mp4", fps=24)

print("🎬 Video saved as story_video.mp4")
