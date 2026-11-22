#!/usr/bin/env python3
"""
Multimodal Generative AI-Based E-Education System.

Integrates Coqui Tacotron2-DDC/YourTTS speech synthesis, SadTalker + GFPGAN
talking-head video generation, Stable Diffusion v1.5 + Rembg (u2netp) classroom
background compositing, OpenAI Whisper subtitle alignment, FFmpeg video
processing, and local 4-bit TinyLlama-1.1B doubt resolution in a role-based
Gradio web application.
"""

import gc
import json
import os
import shutil
import subprocess
import time
import traceback
import uuid
from datetime import datetime

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps

try:
    from sadtalker_patches import patch_sadtalker_repo
except ImportError:
    from src.sadtalker_patches import patch_sadtalker_repo

# Optional model and document dependencies
try:
    from diffusers import StableDiffusionPipeline
except ImportError:
    StableDiffusionPipeline = None
    print("Warning: diffusers is not installed.")

try:
    from rembg import new_session, remove
except ImportError:
    remove = None
    new_session = None
    print("Warning: rembg is not installed.")

try:
    import whisper
except ImportError:
    whisper = None
    print("Warning: openai-whisper is not installed.")

try:
    import docx
except ImportError:
    docx = None
    print("Warning: python-docx is not installed.")

try:
    from pptx import Presentation
except ImportError:
    Presentation = None
    print("Warning: python-pptx is not installed.")

try:
    from TTS.api import TTS
except ImportError:
    TTS = None
    print("Warning: TTS library is not installed.")

try:
    from ctransformers import AutoModelForCausalLM
    import torch

    _orig_torch_load = torch.load

    def _safe_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)

    torch.load = _safe_torch_load
except ImportError:
    AutoModelForCausalLM = None
    torch = None
    print("Warning: ctransformers or torch is not installed.")


# ---------------------------------------------------------------------------
# Configuration and Directory Paths
# ---------------------------------------------------------------------------
BASE_DATA_DIR = os.environ.get("E_EDU_DATA_DIR", "/content")
KB_PATH = os.path.join(BASE_DATA_DIR, "knowledge_base.json")
TMP_DIR = os.path.join(BASE_DATA_DIR, "tmp")
SADTALKER_ROOT = os.environ.get("SADTALKER_ROOT", os.path.join(BASE_DATA_DIR, "SadTalker"))
LESSON_CONTENT_PATH = os.path.join(BASE_DATA_DIR, "lesson_content.json")
QUESTIONS_HISTORY_PATH = os.path.join(BASE_DATA_DIR, "questions_history.json")
SHARED_FILES_DIR = os.path.join(BASE_DATA_DIR, "shared_files")
FILE_REGISTRY_PATH = os.path.join(SHARED_FILES_DIR, "file_registry.json")
TINYLLAMA_DIR = os.environ.get("TINYLLAMA_DIR", ".")
TINYLLAMA_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

STATIC_DIR = "static"
DEFAULT_FACE_SIZE = 256
GENERATION_TIMEOUT = 400

for directory in [
    TMP_DIR,
    STATIC_DIR,
    os.path.join(SHARED_FILES_DIR, "notes"),
    os.path.join(SHARED_FILES_DIR, "assignments"),
    os.path.join(SHARED_FILES_DIR, "resources"),
]:
    os.makedirs(directory, exist_ok=True)


# ---------------------------------------------------------------------------
# Application State and Persistence
# ---------------------------------------------------------------------------
USERS = {
    "teacher": {"password": os.environ.get("TEACHER_PASS", "teach123"), "role": "teacher"},
    "student": {"password": os.environ.get("STUDENT_PASS", "learn123"), "role": "student"},
    "admin": {"password": os.environ.get("ADMIN_PASS", "admin123"), "role": "teacher"},
}

current_user = {"username": None, "role": None, "logged_in": False}
teacher_profile = {"face": None, "audio": None}
video_library = []
uploaded_logo = None
last_generated_video = None

avatar_image_data = None
avatar_voice_data = None
avatar_raw_image_data = None

shared_files_registry = {"files": []}
knowledge_base = {}
asked_questions_history = []

lesson_content = {
    "notes": (
        "Welcome to the E-Education System. Use the tabs below to view lesson notes, "
        "download shared course files, attempt quizzes, watch lecture videos, and ask "
        "questions in the doubt-resolution panel."
    ),
    "quiz": [
        {
            "question": "Which tab provides access to files shared by your instructor?",
            "options": ["A) Notes", "B) Downloads", "C) Quiz", "D) Extras"],
            "answer": "B) Downloads",
        }
    ],
    "assignment": (
        "Assignment: Review the current lecture video and summarize the main concepts "
        "using the notes and doubt-resolution panel."
    ),
    "resources": "",
    "extras": "",
    "flashcards": "",
    "tips": "",
}


def load_shared_files():
    global shared_files_registry
    if os.path.exists(FILE_REGISTRY_PATH):
        try:
            with open(FILE_REGISTRY_PATH, "r", encoding="utf-8") as f:
                shared_files_registry = json.load(f)
        except Exception:
            shared_files_registry = {"files": []}


def save_shared_files():
    try:
        with open(FILE_REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(shared_files_registry, f, indent=2)
    except Exception as e:
        print(f"Error saving shared files registry: {e}")


def load_kb():
    global knowledge_base
    if os.path.exists(KB_PATH):
        try:
            with open(KB_PATH, "r", encoding="utf-8") as f:
                knowledge_base = json.load(f)
        except Exception:
            knowledge_base = {}


def save_kb():
    try:
        with open(KB_PATH, "w", encoding="utf-8") as f:
            json.dump(knowledge_base, f, indent=2)
    except Exception as e:
        print(f"Error saving knowledge base: {e}")


def save_lesson_content():
    try:
        with open(LESSON_CONTENT_PATH, "w", encoding="utf-8") as f:
            json.dump(lesson_content, f, indent=2)
    except Exception as e:
        print(f"Error saving lesson content: {e}")


def load_lesson_content():
    global lesson_content
    if os.path.exists(LESSON_CONTENT_PATH):
        try:
            with open(LESSON_CONTENT_PATH, "r", encoding="utf-8") as f:
                lesson_content = json.load(f)
        except Exception:
            pass


def save_questions_history():
    try:
        with open(QUESTIONS_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(asked_questions_history, f, indent=2)
    except Exception as e:
        print(f"Error saving question history: {e}")


def load_questions_history():
    global asked_questions_history
    if os.path.exists(QUESTIONS_HISTORY_PATH):
        try:
            with open(QUESTIONS_HISTORY_PATH, "r", encoding="utf-8") as f:
                asked_questions_history = json.load(f)
        except Exception:
            asked_questions_history = []
    else:
        asked_questions_history = []


load_shared_files()
load_kb()
load_lesson_content()
load_questions_history()


# ---------------------------------------------------------------------------
# Model Initialization
# ---------------------------------------------------------------------------
tiny_llama = None
coqui_tts = None
yourtts = None
whisper_model = None
pipe = None
rembg_session = None


def init_models():
    """Load local LLM, TTS, Whisper, Rembg, and Stable Diffusion models."""
    global tiny_llama, coqui_tts, yourtts, whisper_model, pipe, rembg_session

    print("Loading models...")

    # 1. Quantized TinyLlama-1.1B (4-bit Q4_K_M GGUF) for local Q&A
    if AutoModelForCausalLM is not None:
        try:
            tiny_llama = AutoModelForCausalLM.from_pretrained(
                TINYLLAMA_DIR,
                model_file=TINYLLAMA_FILE,
                model_type="llama",
                gpu_layers=20,
            )
            print("Loaded TinyLlama-1.1B-Chat-v1.0 (Q4_K_M).")
        except Exception as e:
            print(f"Warning: TinyLlama not loaded: {e}")

    # 2. Coqui Tacotron2-DDC and YourTTS for English speech synthesis and voice cloning
    if TTS is not None:
        try:
            coqui_tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
            print("Loaded Coqui Tacotron2-DDC TTS.")
            try:
                yourtts = TTS("tts_models/multilingual/multi-dataset/your_tts", progress_bar=False)
                print("Loaded YourTTS for voice cloning.")
            except Exception as e:
                print(f"Warning: YourTTS unavailable ({e}); falling back to Tacotron2-DDC.")
        except Exception as e:
            print(f"Warning: TTS initialization failed: {e}")

    # 3. OpenAI Whisper (base) for English subtitle transcription
    if whisper is not None:
        try:
            whisper_model = whisper.load_model("base")
            print("Loaded Whisper (base).")
        except Exception as e:
            print(f"Warning: Whisper not loaded: {e}")

    # 4. Lightweight Rembg (u2netp) session for foreground portrait matting
    if new_session is not None:
        try:
            rembg_session = new_session("u2netp")
            print("Initialized Rembg u2netp session.")
        except Exception as e:
            print(f"Warning: Rembg initialization failed: {e}")

    # 5. Stable Diffusion v1.5 (FP16) for classroom background generation
    device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    if StableDiffusionPipeline is not None:
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                low_cpu_mem_usage=True,
            ).to(device)
            pipe.enable_attention_slicing()
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Loaded Stable Diffusion v1.5 on {device}.")
        except Exception as e:
            print(f"Warning: Stable Diffusion v1.5 not loaded: {e}")

    print("Model initialization complete.")


# ---------------------------------------------------------------------------
# UI Styling
# ---------------------------------------------------------------------------
custom_css = """
<style>
    .gradio-container {
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        background: linear-gradient(135deg, #f0f4f8 0%, #e3f2fd 100%) !important;
        font-size: 16px !important;
        padding: 20px !important;
        color: #000000 !important;
    }
    .prose, .prose * { color: #000000 !important; }
    .app-header {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 50%, #e0f2fe 100%) !important;
        padding: 28px !important;
        border-radius: 20px !important;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.1) !important;
        margin-bottom: 24px !important;
        border: 1px solid rgba(59, 130, 246, 0.1) !important;
        text-align: center !important;
    }
    .app-title {
        font-size: 2.6em !important;
        font-weight: 800 !important;
        color: #1e40af !important;
        margin: 0 !important;
    }
    .app-subtitle {
        background: linear-gradient(90deg, #dbeafe 0%, #bfdbfe 100%) !important;
        color: #1e40af !important;
        padding: 12px 24px !important;
        border-radius: 25px !important;
        font-weight: 600 !important;
        margin: 12px 0 !important;
        font-size: 1.15em !important;
        display: inline-block !important;
    }
    .left-panel, .center-panel, .right-panel {
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
        align-items: stretch !important;
        box-sizing: border-box !important;
        border-radius: 20px !important;
        padding: 24px !important;
        margin: 10px !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08) !important;
        border: 2px solid rgba(0, 0, 0, 0.08) !important;
    }
    .left-panel { background: linear-gradient(135deg, #fff4e6 0%, #ffeaa7 50%, #fdcb6e 100%) !important; }
    .center-panel { background: linear-gradient(135deg, #e8f5e8 0%, #a8e6a3 50%, #81c784 100%) !important; }
    .right-panel { background: linear-gradient(135deg, #e3f2fd 0%, #90caf9 50%, #64b5f6 100%) !important; }
    .section-header {
        font-size: 1.6em !important;
        font-weight: 800 !important;
        color: #1e293b !important;
        margin: 16px 0 12px 0 !important;
        text-align: center !important;
    }
    .primary-button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 50%, #1d4ed8 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        cursor: pointer !important;
    }
    .upload-button {
        background: linear-gradient(135deg, #10b981 0%, #059669 50%, #047857 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 20px !important;
        font-weight: 700 !important;
        cursor: pointer !important;
    }
    .logout-button {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 50%, #b45309 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 8px 18px !important;
        font-weight: 700 !important;
        cursor: pointer !important;
    }
    .login-container {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 20px !important;
        padding: 40px !important;
        box-shadow: 0 12px 30px rgba(59, 130, 246, 0.15) !important;
        max-width: 460px !important;
        margin: 0 auto !important;
    }
    .splash-screen {
        background: linear-gradient(135deg, #f0f4f8 0%, #e3f2fd 50%, #e1f5fe 100%) !important;
        height: 100vh !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        flex-direction: column !important;
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100% !important;
        z-index: 9999 !important;
    }
    .logo-container {
        text-align: center !important;
        padding: 36px !important;
        background: rgba(255, 255, 255, 0.92) !important;
        border-radius: 20px !important;
        box-shadow: 0 12px 30px rgba(59, 130, 246, 0.15) !important;
    }
    .spinner {
        border: 7px solid #e3f2fd;
        border-top: 7px solid #3b82f6;
        border-radius: 50%;
        width: 54px;
        height: 54px;
        animation: spin 1s linear infinite;
        margin: 18px auto;
    }
    @keyframes spin { 100% { transform: rotate(360deg); } }
</style>
"""


# ---------------------------------------------------------------------------
# Authentication Helpers
# ---------------------------------------------------------------------------
def get_welcome_name(default="Student"):
    if not current_user or not isinstance(current_user, dict):
        return default
    username = current_user.get("username") or default
    return str(username).title()


def authenticate_user(username, password):
    if username in USERS and USERS[username]["password"] == password:
        current_user["username"] = username
        current_user["role"] = USERS[username]["role"]
        current_user["logged_in"] = True
        load_lesson_content()
        return True, f"Welcome, {username.title()} ({USERS[username]['role'].title()})."
    return False, "Invalid username or password."


def check_access(required_role="student"):
    if not current_user["logged_in"]:
        return False
    if required_role == "teacher":
        return current_user["role"] == "teacher"
    return True


# ---------------------------------------------------------------------------
# Image Processing & Background Compositing
# ---------------------------------------------------------------------------
def clean_alpha(img, threshold=10):
    img = img.convert("RGBA")
    arr = np.array(img)
    arr[arr[:, :, 3] < threshold] = [0, 0, 0, 0]
    return Image.fromarray(arr, "RGBA")


def feather_edges(img, radius=3):
    r, g, b, a = img.split()
    a_blur = a.filter(ImageFilter.GaussianBlur(radius))
    a = ImageChops.blend(a, a_blur, 0.7)
    return Image.merge("RGBA", (r, g, b, a))


def enhance_image(img):
    img = img.convert("RGB")
    img = ImageOps.autocontrast(img, cutoff=1)
    img = ImageEnhance.Color(img).enhance(1.1)
    img = ImageEnhance.Sharpness(img).enhance(1.05)
    return img.convert("RGBA")


def color_match(source, target):
    """Transfer CIE LAB mean and standard deviation from source to target."""
    try:
        src = cv2.cvtColor(np.array(source.convert("RGB")), cv2.COLOR_RGB2LAB).astype("float32")
        tgt = cv2.cvtColor(np.array(target.convert("RGB")), cv2.COLOR_RGB2LAB).astype("float32")
        (lm_s, am_s, bm_s), (ls_s, as_s, bs_s) = cv2.meanStdDev(src)
        (lm_t, am_t, bm_t), (ls_t, as_t, bs_t) = cv2.meanStdDev(tgt)
        l_ch, a_ch, b_ch = cv2.split(tgt)
        l_ch = (l_ch - lm_t[0]) * (ls_s[0] / (ls_t[0] + 1e-6)) + lm_s[0]
        a_ch = (a_ch - am_t[0]) * (as_s[0] / (as_t[0] + 1e-6)) + am_s[0]
        b_ch = (b_ch - bm_t[0]) * (bs_s[0] / (bs_t[0] + 1e-6)) + bm_s[0]
        merged = np.clip(cv2.merge([l_ch, a_ch, b_ch]), 0, 255)
        rgb = cv2.cvtColor(merged.astype("uint8"), cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb).convert("RGB")
    except Exception:
        return target.convert("RGB")


def composite_images(fg, bg):
    bg = bg.resize(fg.size, Image.Resampling.LANCZOS).convert("RGB")
    fg = fg.convert("RGBA")
    arr_fg = np.array(fg)
    arr_bg = np.array(bg)
    alpha = cv2.GaussianBlur(arr_fg[:, :, 3] / 255.0, (3, 3), 1.0)
    alpha_3 = np.dstack([alpha] * 3)
    out = arr_fg[:, :, :3] * alpha_3 + arr_bg * (1.0 - alpha_3)
    return Image.fromarray(out.astype(np.uint8), "RGB")


def generate_with_bg(uploaded_img, prompt, bg_upload_img, resolution=512, steps=25):
    """Segment the instructor portrait and composite it over a generated or uploaded background."""
    global rembg_session
    if uploaded_img is None:
        return None, None

    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

    fg_input = uploaded_img.convert("RGB").resize((resolution, resolution), Image.Resampling.LANCZOS)

    if bg_upload_img is not None:
        bg = bg_upload_img.convert("RGB").resize((resolution, resolution), Image.Resampling.LANCZOS)
    elif prompt and prompt.strip() and pipe is not None:
        with torch.inference_mode():
            bg = pipe(
                prompt=f"{prompt}, realistic classroom lighting",
                negative_prompt="blurry, distorted, low quality",
                num_inference_steps=steps,
                guidance_scale=7.5,
                width=resolution,
                height=resolution,
            ).images[0]
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        return None, None

    fg = enhance_image(fg_input)
    if remove is not None:
        try:
            if rembg_session is None and new_session is not None:
                rembg_session = new_session("u2netp")
            fg = remove(fg, session=rembg_session).convert("RGBA")
        except Exception as e:
            print(f"Rembg segmentation failed, using unsegmented foreground: {e}")
            fg = fg.convert("RGBA")

    fg = clean_alpha(fg, threshold=10)
    fg = feather_edges(fg, radius=3)
    result = composite_images(fg, bg)

    out_path = os.path.join(TMP_DIR, f"bg_composited_{uuid.uuid4().hex[:8]}.png")
    result.save(out_path)

    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result, out_path


def auto_resize_face_image(img, target_size=DEFAULT_FACE_SIZE):
    """Convert portrait to RGB, take a centered square crop, and resize via Lanczos."""
    try:
        if img is None:
            return None
        if hasattr(img, "size") and hasattr(img, "mode"):
            im = img.copy()
        elif isinstance(img, np.ndarray) and img.ndim == 3:
            im = Image.fromarray(img.astype(np.uint8))
        elif isinstance(img, str) and os.path.exists(img):
            im = Image.open(img)
        else:
            return None

        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")

        w, h = im.size
        if w != h:
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            im = im.crop((left, top, left + side, top + side))

        if im.size[0] != target_size:
            im = im.resize((target_size, target_size), Image.Resampling.LANCZOS)
        return im
    except Exception as e:
        print(f"Face preprocessing error: {e}")
        return None


def auto_resize_side_image(img, target_height):
    """Resize concept diagram to match the video height while preserving aspect ratio."""
    try:
        if img is None:
            return None
        if hasattr(img, "size") and hasattr(img, "mode"):
            im = img.copy()
        elif isinstance(img, np.ndarray) and img.ndim == 3:
            im = Image.fromarray(img.astype(np.uint8))
        elif isinstance(img, str) and os.path.exists(img):
            im = Image.open(img)
        else:
            return None

        if im.mode != "RGB":
            im = im.convert("RGB")

        if im.height != target_height:
            new_width = int(im.width * (target_height / im.height))
            im = im.resize((new_width, target_height), Image.Resampling.LANCZOS)
        return im
    except Exception as e:
        print(f"Side image preprocessing error: {e}")
        return None


def save_image(img, prefix="img"):
    if img is None:
        return None
    try:
        path = os.path.join(TMP_DIR, f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}.png")
        img.save(path, "PNG", optimize=True)
        return path if os.path.exists(path) and os.path.getsize(path) > 0 else None
    except Exception as e:
        print(f"Image save error: {e}")
        return None


def ensure_file_from_audio(audio):
    if isinstance(audio, str) and os.path.exists(audio):
        return audio
    return None


def get_video_dims(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return DEFAULT_FACE_SIZE, DEFAULT_FACE_SIZE
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return w, h
    except Exception:
        return DEFAULT_FACE_SIZE, DEFAULT_FACE_SIZE


def has_video_stream(video_path):
    try:
        if not video_path or not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            return False
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        cap.release()
        return bool(ret and frame is not None)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Generative Pipeline (LLM, TTS, SadTalker, FFmpeg HStack, Subtitles)
# ---------------------------------------------------------------------------
def safe_llm_answer(question):
    """Generate a concise local answer using 4-bit TinyLlama-1.1B without external APIs."""
    try:
        if tiny_llama is not None:
            prompt = f"<|user|>\n{question}\n<|assistant|>\n"
            raw_answer = tiny_llama(prompt, max_new_tokens=64, temperature=0.7).strip()
            for ending in (". ", "! ", "? "):
                pos = raw_answer.rfind(ending)
                if pos != -1:
                    return raw_answer[: pos + 1]
            return raw_answer
        return f"Here is a brief overview of {question}: please review the lecture notes for details."
    except Exception as e:
        print(f"TinyLlama inference error: {e}")
        return f"Unable to generate a response for {question} right now."


def safe_tts_synthesis(text, voice_audio=None):
    """Synthesize English speech via YourTTS voice cloning or Tacotron2-DDC."""
    output_path = os.path.join(TMP_DIR, f"tts_{uuid.uuid4().hex[:8]}.wav")
    text = (text or "").strip() or "Processing request."

    if voice_audio and yourtts is not None and os.path.exists(voice_audio):
        try:
            yourtts.tts_to_file(
                text=text,
                speaker_wav=voice_audio,
                file_path=output_path,
                language="en",
            )
            waited = 0.0
            while waited < 10.0:
                if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                    return output_path
                time.sleep(0.5)
                waited += 0.5
        except Exception as e:
            print(f"YourTTS synthesis error: {e}")

    if coqui_tts is not None:
        try:
            coqui_tts.tts_to_file(text=text, file_path=output_path)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
        except Exception as e:
            print(f"Tacotron2-DDC synthesis error: {e}")

    # Offline fallback tone if TTS checkpoints are missing
    sample_rate = 22050
    duration = max(2.0, len(text) * 0.08)
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = 0.1 * np.sin(440 * 2 * np.pi * t).astype(np.float32)
    try:
        from scipy.io.wavfile import write

        write(output_path, sample_rate, (audio_data * 32767).astype(np.int16))
    except Exception as e:
        print(f"Fallback WAV write error: {e}")
    return output_path


def generate_talking_video_safe(face_path, wav_path):
    """Run SadTalker with GFPGAN restoration to produce a lip-synced instructor video."""
    try:
        if not face_path or not os.path.exists(face_path):
            print("Error: Teacher face image not found.")
            return None
        if not wav_path or not os.path.exists(wav_path):
            print("Error: Speech WAV file not found.")
            return None
        if not os.path.exists(SADTALKER_ROOT):
            print(f"Error: SadTalker directory not found at {SADTALKER_ROOT}.")
            return None

        patch_sadtalker_repo(SADTALKER_ROOT)

        output_dir = os.path.join(TMP_DIR, f"sadtalker_{int(time.time())}_{uuid.uuid4().hex[:8]}")
        os.makedirs(output_dir, exist_ok=True)
        face_input = os.path.join(output_dir, "input_face.png")
        audio_input = os.path.join(output_dir, "input_audio.wav")
        shutil.copy2(face_path, face_input)
        shutil.copy2(wav_path, audio_input)

        cmd = [
            "python",
            "inference.py",
            "--driven_audio",
            audio_input,
            "--source_image",
            face_input,
            "--result_dir",
            output_dir,
            "--preprocess",
            "full",
            "--size",
            "256",
            "--batch_size",
            "4",
            "--expression_scale",
            "0.7",
            "--still",
            "--enhancer",
            "gfpgan",
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

        result = subprocess.run(
            cmd,
            cwd=SADTALKER_ROOT,
            capture_output=True,
            text=True,
            timeout=GENERATION_TIMEOUT,
            env=env,
        )

        if result.returncode == 0:
            for root, _, files in os.walk(output_dir):
                for fname in files:
                    if fname.endswith(".mp4"):
                        candidate = os.path.join(root, fname)
                        if has_video_stream(candidate):
                            return candidate
        else:
            print(f"SadTalker exited with code {result.returncode}: {result.stderr[-500:]}")
        return None
    except subprocess.TimeoutExpired:
        print(f"SadTalker timed out after {GENERATION_TIMEOUT}s.")
        return None
    except Exception as e:
        print(f"SadTalker execution error: {e}")
        traceback.print_exc()
        return None


def fast_side_image_stack(video_path, side_img, out_path):
    """Horizontally stack a concept diagram beside the talking-head video via FFmpeg hstack."""
    try:
        if not video_path or not os.path.exists(video_path):
            return
        if side_img is None:
            shutil.copy2(video_path, out_path)
            return

        vid_width, vid_height = get_video_dims(video_path)
        side_resized = auto_resize_side_image(side_img, vid_height)
        if not side_resized:
            shutil.copy2(video_path, out_path)
            return

        side_width = side_resized.width
        if (vid_width + side_width) % 2 != 0:
            padded = Image.new("RGB", (side_width + 1, vid_height), (255, 255, 255))
            padded.paste(side_resized, (0, 0))
            side_resized = padded
            side_width += 1

        side_temp_path = save_image(side_resized, "side_temp")
        if not side_temp_path:
            shutil.copy2(video_path, out_path)
            return

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 5.0
        cap.release()

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-loop",
            "1",
            "-i",
            side_temp_path,
            "-filter_complex",
            f"[1:v]scale={side_width}:{vid_height}[side];[0:v][side]hstack=inputs=2",
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            out_path,
        ]

        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if res.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            shutil.copy2(video_path, out_path)

        if os.path.exists(side_temp_path):
            os.remove(side_temp_path)
    except Exception as e:
        print(f"Side-diagram stacking error: {e}")
        if video_path and out_path:
            shutil.copy2(video_path, out_path)


def add_subtitles(video_path, text, out_path):
    """Burn time-aligned SRT subtitles onto the lecture video using FFmpeg."""
    try:
        if not video_path or not os.path.exists(video_path):
            return
        if not text or not text.strip():
            shutil.copy2(video_path, out_path)
            return

        srt_path = os.path.join(TMP_DIR, f"subtitle_{int(time.time())}.srt")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = int(frame_count / fps) if fps > 0 else 5
        cap.release()

        srt_content = f"1\n00:00:00,000 --> 00:00:{duration:02d},000\n{text.strip()}\n"
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vf",
            f"subtitles={srt_path}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-c:a",
            "copy",
            out_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if res.returncode != 0:
            shutil.copy2(video_path, out_path)

        if os.path.exists(srt_path):
            os.remove(srt_path)
    except Exception as e:
        print(f"Subtitle burn-in error: {e}")
        if video_path and out_path:
            shutil.copy2(video_path, out_path)


# ---------------------------------------------------------------------------
# Course Material, Knowledge-Base Search, and Teacher/Student Workflows
# ---------------------------------------------------------------------------
def extract_text_from_file(file_path):
    """Extract UTF-8 lesson script text from .docx or .pptx files."""
    if not file_path:
        return ""
    try:
        if file_path.endswith(".docx") and docx is not None:
            doc = docx.Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        if file_path.endswith(".pptx") and Presentation is not None:
            prs = Presentation(file_path)
            lines = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        lines.append(shape.text.strip())
            return "\n".join(lines)
        return "Unsupported file format. Please upload a .docx or .pptx file."
    except Exception as e:
        return f"Error reading file: {e}"


def handle_script_file_upload(file):
    if file is None:
        return ""
    return extract_text_from_file(file.name)


def share_file_with_students(uploaded_file, category="notes", description=""):
    """Copy an uploaded course resource into the shared directory and register it."""
    try:
        if uploaded_file is None:
            return "No file selected."
        timestamp = int(time.time())
        original_name = os.path.basename(uploaded_file.name)
        file_ext = os.path.splitext(original_name)[1]
        unique_filename = f"{timestamp}_{uuid.uuid4().hex[:8]}{file_ext}"
        destination_path = os.path.join(SHARED_FILES_DIR, category, unique_filename)
        shutil.copy2(uploaded_file.name, destination_path)

        file_size = os.path.getsize(destination_path)
        if file_size < 1024:
            size_str = f"{file_size} B"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"

        record = {
            "id": f"file_{timestamp}_{uuid.uuid4().hex[:8]}",
            "original_name": original_name,
            "stored_filename": unique_filename,
            "category": category,
            "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "uploaded_by": current_user.get("username", "teacher"),
            "file_size": size_str,
            "description": description or f"{category.title()} file",
            "file_path": destination_path,
        }
        shared_files_registry["files"].append(record)
        save_shared_files()
        return (
            f"Shared **{original_name}** ({size_str}) in **{category.title()}** "
            f"at {record['upload_date']}."
        )
    except Exception as e:
        return f"Error sharing file: {e}"


def get_shared_files(category=None):
    load_shared_files()
    files = shared_files_registry.get("files", [])
    if category:
        files = [f for f in files if f.get("category") == category]
    return files


def improved_knowledge_search(question):
    """Four-tier lexical-semantic search over knowledge_base.json."""
    if not question or not question.strip():
        return None, None

    query = question.strip().lower()
    stop_words = {"what", "is", "are", "how", "why", "when", "where", "define", "explain"}
    query_words = [w for w in query.split() if w not in stop_words and len(w) > 2]

    # 1. Exact key match
    if query in knowledge_base:
        entry = knowledge_base[query]
        return entry.get("video"), entry.get("answer")

    # 2. Topic prefix/suffix match
    for key, entry in knowledge_base.items():
        if key.startswith("library_"):
            continue
        for word in query_words:
            if key == word or key.startswith(word + " ") or key.endswith(" " + word):
                return entry.get("video"), entry.get("answer")

    # 3. Partial substring match
    for key, entry in knowledge_base.items():
        if key.startswith("library_"):
            continue
        if query in key or key in query:
            return entry.get("video"), entry.get("answer")

    # 4. Weighted word-overlap score
    best_match = None
    best_score = 0
    for key, entry in knowledge_base.items():
        if key.startswith("library_"):
            continue
        stored_keywords = entry.get("keywords", key.split())
        key_words = key.split()
        score = 0
        matched = 0
        for word in query_words:
            if word in stored_keywords:
                score += 3
                matched += 1
            elif word in key_words:
                score += 2
                matched += 1
            else:
                for kw in stored_keywords + key_words:
                    if word in kw or kw in word:
                        score += 1
                        matched += 1
                        break
        if matched > 1:
            score += matched
        if score >= 2 and score > best_score:
            best_score = score
            best_match = entry

    if best_match is not None:
        return best_match.get("video"), best_match.get("answer")
    return None, None


def add_question_to_history(question):
    global asked_questions_history
    if question and question.strip():
        clean_q = question.strip()
        if clean_q not in asked_questions_history:
            asked_questions_history.append(clean_q)
            if len(asked_questions_history) > 25:
                asked_questions_history = asked_questions_history[-25:]
            save_questions_history()


def get_question_choices():
    if not asked_questions_history:
        return ["No questions asked yet"]
    choices = []
    for i, q in enumerate(reversed(asked_questions_history[-10:]), 1):
        choices.append(f"Q{i}: {q[:45]}..." if len(q) > 45 else f"Q{i}: {q}")
    return choices


def select_from_dropdown(selected):
    if not selected or "No questions" in selected:
        return ""
    if ": " in selected:
        part = selected.split(": ", 1)[1]
        if part.endswith("..."):
            prefix = part[:-3]
            for q in asked_questions_history:
                if q.startswith(prefix):
                    return q
        return part
    return selected


def upload_avatar_image(image):
    global avatar_image_data, avatar_raw_image_data
    if image is not None:
        avatar_image_data = image
        avatar_raw_image_data = image
        return "Avatar image uploaded. You may generate a classroom background or click Set as Profile."
    return "Please select an instructor portrait image."


def upload_avatar_voice(audio):
    global avatar_voice_data
    if audio is not None:
        avatar_voice_data = audio
        return "Reference voice sample uploaded."
    return "Please select a reference WAV audio sample."


def teacher_generate_avatar_bg(uploaded_img, prompt, bg_upload_img):
    global avatar_image_data, avatar_raw_image_data
    source_img = avatar_raw_image_data if avatar_raw_image_data is not None else uploaded_img
    if source_img is None:
        return None, None, None, "Please upload an avatar image first."
    if bg_upload_img is None and (not prompt or not prompt.strip()):
        return uploaded_img, None, None, "Please enter a background prompt or upload a background image."

    result_img, download_path = generate_with_bg(source_img, prompt, bg_upload_img)
    if result_img is not None:
        avatar_image_data = result_img
        return result_img, result_img, download_path, "Background applied. Click Set as Profile to save."
    return uploaded_img, None, None, "Background generation failed."


def set_profile(prompt="", bg_upload_img=None):
    global avatar_image_data, avatar_raw_image_data, avatar_voice_data, teacher_profile
    if not check_access("teacher"):
        return "Access denied. Teacher login required."
    if avatar_image_data is None:
        return "Please upload an avatar image first."
    if avatar_voice_data is None:
        return "Please upload a reference voice sample first."

    if (
        (bg_upload_img is not None or (prompt and str(prompt).strip()))
        and avatar_raw_image_data is not None
        and avatar_image_data is avatar_raw_image_data
    ):
        bg_res, _ = generate_with_bg(avatar_raw_image_data, str(prompt), bg_upload_img)
        if bg_res is not None:
            avatar_image_data = bg_res

    processed_face = auto_resize_face_image(avatar_image_data)
    face_path = save_image(processed_face, "teacher_face")
    voice_path = ensure_file_from_audio(avatar_voice_data)
    if not face_path or not voice_path:
        return "Failed to save teacher profile assets."

    teacher_profile["face"] = face_path
    teacher_profile["audio"] = voice_path
    return "Teacher profile saved. Ready to generate lecture videos."


def teacher_generate_video(question, script_text, side_img):
    global last_generated_video
    if not check_access("teacher"):
        return "Access denied. Teacher login required.", None, gr.update(value=None)

    face_path = teacher_profile.get("face")
    voice_path = teacher_profile.get("audio")
    if not face_path or not os.path.exists(face_path):
        return "Please set up your teacher profile first.", None, gr.update(value=None)
    if not question or not question.strip():
        return "Please enter a title or topic for this video.", None, gr.update(value=None)

    script = (script_text or "").strip()
    if not script:
        if voice_path and whisper_model is not None and os.path.exists(voice_path):
            script = whisper_model.transcribe(voice_path)["text"].strip()
        else:
            return "Please enter a teaching script or upload a voice sample.", None, gr.update(value=None)

    audio_path = safe_tts_synthesis(script, voice_path)
    if not audio_path:
        return "Failed to synthesize speech audio.", None, gr.update(value=None)

    main_video = generate_talking_video_safe(face_path, audio_path)
    if not main_video:
        return "Failed to generate talking-head video. Verify SadTalker setup.", None, gr.update(value=None)

    final_video = main_video
    if side_img is not None:
        side_video = os.path.join(TMP_DIR, f"with_side_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp4")
        fast_side_image_stack(final_video, side_img, side_video)
        if os.path.exists(side_video) and has_video_stream(side_video):
            final_video = side_video

    subtitled_video = os.path.join(TMP_DIR, f"final_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp4")
    add_subtitles(final_video, script, subtitled_video)
    if os.path.exists(subtitled_video) and has_video_stream(subtitled_video):
        final_video = subtitled_video

    question_key = question.strip().lower()
    knowledge_base[question_key] = {
        "answer": script,
        "video": final_video,
        "question": question,
        "keywords": question_key.split(),
    }
    save_kb()

    video_library.append(
        {
            "id": len(video_library) + 1,
            "question": question,
            "script": script,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "concept_diagram": side_img is not None,
            "video_path": final_video,
        }
    )
    last_generated_video = final_video
    return f"Video generated for '{question}'.", final_video, gr.update(value=final_video)


def student_query_flow(question):
    if not check_access("student"):
        return None, "Please log in to ask questions."
    if not question or not question.strip():
        return None, "Please enter a question."

    add_question_to_history(question)
    found_video, found_answer = improved_knowledge_search(question)
    if found_video and found_answer and os.path.exists(found_video):
        return found_video, found_answer

    face_path = teacher_profile.get("face")
    voice_path = teacher_profile.get("audio")
    if not face_path or not os.path.exists(face_path):
        return None, "Teacher profile is not configured yet."

    answer_text = safe_llm_answer(question)
    audio_path = safe_tts_synthesis(answer_text, voice_path)
    video_path = generate_talking_video_safe(face_path, audio_path)
    if not video_path:
        return None, answer_text

    question_key = question.strip().lower()
    knowledge_base[question_key] = {
        "answer": answer_text,
        "video": video_path,
        "question": question,
        "keywords": question_key.split(),
    }
    save_kb()
    return video_path, answer_text


def update_notes(notes_text):
    if not check_access("teacher"):
        return "Access denied. Teacher login required."
    lesson_content["notes"] = notes_text
    save_lesson_content()
    return "Notes updated."


def add_quiz_question(question, option1, option2, option3, option4, correct_answer):
    if not check_access("teacher"):
        return "Access denied. Teacher login required."
    if not all([question, option1, option2, option3, option4, correct_answer]):
        return "Please complete all quiz fields."
    if correct_answer not in [option1, option2, option3, option4]:
        return "Correct answer must match one of the four options."
    lesson_content["quiz"].append(
        {"question": question, "options": [option1, option2, option3, option4], "answer": correct_answer}
    )
    save_lesson_content()
    return f"Quiz question added (total: {len(lesson_content['quiz'])})."


def get_quiz_question_choices():
    if not lesson_content["quiz"]:
        return ["No questions available"]
    return [
        f"Q{i + 1}: {q['question'][:50]}..." if len(q["question"]) > 50 else f"Q{i + 1}: {q['question']}"
        for i, q in enumerate(lesson_content["quiz"])
    ]


def delete_quiz_question(selected_question):
    if not check_access("teacher"):
        return "Access denied. Teacher login required.", gr.update()
    if not selected_question or selected_question == "No questions available":
        return "Please select a question to delete.", gr.update()
    try:
        idx = int(selected_question.split(":")[0][1:]) - 1
        if 0 <= idx < len(lesson_content["quiz"]):
            removed = lesson_content["quiz"].pop(idx)
            save_lesson_content()
            return f"Deleted: {removed['question'][:50]}", gr.update(choices=get_quiz_question_choices(), value=None)
    except Exception as e:
        return f"Error deleting question: {e}", gr.update()
    return "Invalid selection.", gr.update()


def refresh_student_components():
    load_lesson_content()
    quiz_text = f"Total Quiz Questions: {len(lesson_content['quiz'])}\n\n" + "\n".join(
        f"Q{i + 1}: {q['question']}\nAnswer: {q['answer']}\n"
        for i, q in enumerate(lesson_content["quiz"])
    )
    return (
        gr.update(value=lesson_content["notes"]),
        gr.update(value=quiz_text),
        gr.update(value=lesson_content["assignment"]),
        gr.update(value=lesson_content["resources"]),
        gr.update(value=lesson_content["extras"]),
        gr.update(value=lesson_content["flashcards"]),
        gr.update(value=lesson_content["tips"]),
    )


def get_video_options():
    options = []
    for key, entry in knowledge_base.items():
        if not key.startswith("library_") and "video" in entry and os.path.exists(entry["video"]):
            q = entry.get("question", key)
            options.append(f"Q: {q[:60]}..." if len(q) > 60 else f"Q: {q}")
    return options if options else ["No videos available"]


def play_selected_video(selected_option):
    if not selected_option or selected_option == "No videos available":
        return None
    for key, entry in knowledge_base.items():
        if "video" in entry and os.path.exists(entry["video"]):
            q = entry.get("question", key)
            if q in selected_option:
                return entry["video"]
    return None


def refresh_video_dropdown():
    return gr.update(choices=get_video_options(), value=None)


# ---------------------------------------------------------------------------
# Gradio Web Interface
# ---------------------------------------------------------------------------
def create_logo():
    return (
        '<div style="text-align: center; margin: 16px 0;">'
        '<div style="font-size: 2.1em; font-weight: 800; color: #1e40af;">E-Education System</div>'
        '<div style="font-size: 1.15em; color: #475569; margin-top: 4px;">AI-Powered Virtual Tutor</div>'
        "</div>"
    )


def create_teacher_file_sharing_tab():
    with gr.Column():
        with gr.Row():
            file_upload = gr.File(
                label="Select File to Share",
                file_types=[".pdf", ".docx", ".pptx", ".txt", ".png", ".jpg", ".jpeg", ".xlsx"],
                type="filepath",
            )
            category_select = gr.Dropdown(
                choices=["notes", "assignments", "resources"],
                value="notes",
                label="Category",
            )
        share_btn = gr.Button("Share File with Students", variant="primary")
        share_status = gr.Markdown("")
        refresh_btn = gr.Button("Refresh Files", variant="secondary")
        shared_files_display = gr.Markdown("")

        def get_current_files_display():
            files = get_shared_files()
            return f"Total Files Shared: {len(files)}" if files else "No files shared yet."

        def handle_file_share(uploaded_file, category):
            if not uploaded_file:
                return "Please select a file.", get_current_files_display()
            return share_file_with_students(uploaded_file, category), get_current_files_display()

        share_btn.click(handle_file_share, inputs=[file_upload, category_select], outputs=[share_status, shared_files_display])
        refresh_btn.click(get_current_files_display, outputs=[shared_files_display])
        shared_files_display.value = get_current_files_display()

    return file_upload, category_select, share_btn, share_status


def create_simple_downloads():
    with gr.Column():
        gr.Markdown("### Download Course Files")
        file_dropdown = gr.Dropdown(label="Select File to Download", choices=[], value=None)
        download_btn = gr.Button("Download Selected File", variant="primary")
        downloaded_file = gr.File(label="Downloaded File")
        refresh_btn = gr.Button("Refresh File List")
        files_info = gr.Markdown("Click Refresh to view available files.")

        def get_all_files_for_download():
            all_files = get_shared_files()
            if not all_files:
                return [], "No files available yet."
            choices = [
                (f"{f['original_name']} ({f.get('category', 'file').title()}) - {f.get('file_size', '')}", f["id"])
                for f in all_files
            ]
            return choices, f"Total Available Files: {len(all_files)}"

        def download_file(file_id):
            if not file_id:
                return None
            for record in get_shared_files():
                if record.get("id") == file_id and os.path.exists(record.get("file_path", "")):
                    return record["file_path"]
            return None

        def refresh_files():
            choices, info = get_all_files_for_download()
            return gr.update(choices=choices), info

        refresh_btn.click(refresh_files, outputs=[file_dropdown, files_info])
        download_btn.click(download_file, inputs=[file_dropdown], outputs=[downloaded_file])
        choices, info = get_all_files_for_download()
        file_dropdown.choices = choices
        files_info.value = info

    return file_dropdown, downloaded_file, files_info


def create_complete_app():
    with gr.Blocks(css=custom_css, fill_height=True, title="E-Education System") as app:
        with gr.Group(visible=True) as splash_screen:
            gr.HTML(
                f'<div class="splash-screen"><div class="logo-container">{create_logo()}'
                '<div class="spinner"></div>'
                '<p style="color: #475569;">Initializing AI-Powered Virtual Tutor...</p>'
                "</div></div>"
            )

        with gr.Group(visible=False) as login_screen:
            with gr.Row():
                with gr.Column(scale=1):
                    pass
                with gr.Column(scale=2):
                    gr.HTML(f'<div class="login-container">{create_logo()}')
                    username_input = gr.Textbox(label="Username", placeholder="Enter username")
                    password_input = gr.Textbox(label="Password", type="password", placeholder="Enter password")
                    login_btn = gr.Button("Sign In", variant="primary", elem_classes=["primary-button"])
                    login_message = gr.HTML()
                    gr.HTML(
                        '<div style="background: #f0f9ff; padding: 16px; border-radius: 12px; margin-top: 16px;">'
                        "<strong>Demo Accounts:</strong><br>"
                        "Teacher: <code>teacher</code> / <code>teach123</code><br>"
                        "Student: <code>student</code> / <code>learn123</code>"
                        "</div></div>"
                    )
                with gr.Column(scale=1):
                    pass

        with gr.Group(visible=False) as teacher_dashboard:
            gr.HTML(
                '<div class="app-header">'
                '<h1 class="app-title">E-Education System</h1>'
                '<div class="app-subtitle">Teacher Dashboard - AI-Powered Virtual Tutor</div>'
                '<div style="margin-top: 12px;">'
                '<button class="logout-button" onclick="location.reload()">Logout</button>'
                "</div></div>"
            )
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, elem_classes=["left-panel"]):
                    gr.HTML('<div class="section-header">Avatar & Video Generation</div>')
                    avatar_img_upload = gr.Image(label="Upload Teacher Portrait", type="pil", height=240)
                    prompt_input = gr.Textbox(label="Classroom Background Prompt", placeholder="e.g., modern digital classroom", lines=2)
                    bg_upload = gr.Image(label="Or Upload Background Image", type="pil", height=160)
                    generate_bg_btn = gr.Button("Generate Background on Avatar", elem_classes=["upload-button"])
                    img_output = gr.Image(label="Composited Avatar Preview", type="pil", height=220)
                    download_btn = gr.File(label="Download Composited Avatar")
                    avatar_voice_upload = gr.Audio(label="Upload Reference Voice (WAV)", type="filepath")
                    set_profile_btn = gr.Button("Set as Profile", elem_classes=["upload-button"])
                    profile_status = gr.HTML()

                    question_input = gr.Textbox(label="Lecture Topic / Question Title", lines=2)
                    script_file_upload = gr.File(label="Upload Lesson Script (.docx / .pptx)", file_types=[".pptx", ".docx"])
                    teaching_script = gr.Textbox(label="Teaching Script", lines=6)
                    concept_diagram = gr.Image(label="Concept Diagram (Optional)", type="pil", height=150)
                    generate_btn = gr.Button("Generate Lecture Video", elem_classes=["primary-button"])
                    generation_status = gr.HTML()

                with gr.Column(scale=1, elem_classes=["center-panel"]):
                    gr.HTML('<div class="section-header">Video Library & Learning Materials</div>')
                    teacher_video_display = gr.Video(label="Generated Lecture Video", height=260)
                    teacher_video_dropdown = gr.Dropdown(label="Select Video to Play", choices=get_video_options())
                    with gr.Row():
                        play_teacher_video_btn = gr.Button("Play", elem_classes=["upload-button"])
                        refresh_teacher_dropdown_btn = gr.Button("Refresh", elem_classes=["upload-button"])
                        download_teacher_video_btn = gr.DownloadButton("Download", elem_classes=["upload-button"])

                    with gr.Tabs():
                        with gr.Tab("Share Files"):
                            create_teacher_file_sharing_tab()
                        with gr.Tab("Notes"):
                            notes_editor = gr.Textbox(label="Edit Lesson Notes", value=lesson_content["notes"], lines=6)
                            update_notes_btn = gr.Button("Update Notes", elem_classes=["upload-button"])
                            notes_status = gr.HTML()
                        with gr.Tab("Quiz"):
                            quiz_question = gr.Textbox(label="Question")
                            with gr.Row():
                                quiz_option1 = gr.Textbox(label="Option 1")
                                quiz_option2 = gr.Textbox(label="Option 2")
                            with gr.Row():
                                quiz_option3 = gr.Textbox(label="Option 3")
                                quiz_option4 = gr.Textbox(label="Option 4")
                            quiz_answer = gr.Dropdown(label="Correct Answer", choices=[])
                            add_quiz_btn = gr.Button("Add Question", elem_classes=["upload-button"])
                            quiz_status = gr.HTML()
                            quiz_delete_dropdown = gr.Dropdown(label="Select Question to Delete", choices=get_quiz_question_choices())
                            delete_quiz_btn = gr.Button("Delete Selected Question", elem_classes=["logout-button"])
                            delete_quiz_status = gr.HTML()

                with gr.Column(scale=1, elem_classes=["right-panel"], visible=False):
                    student_notes_preview = gr.Markdown(lesson_content["notes"])
                    student_quiz_preview = gr.Markdown("")
                    teacher_ai_video = gr.Video(label="Response Video", height=160)
                    teacher_ai_response = gr.Textbox(label="Teacher Answer", lines=6, interactive=False)
                    teacher_history_dropdown = gr.Dropdown(label="Previous Questions", choices=get_question_choices())
                    teacher_use_previous_btn = gr.Button("Use Selected", elem_classes=["upload-button"])
                    teacher_question_input = gr.Textbox(label="Ask Question", lines=4)
                    teacher_ask_btn = gr.Button("Ask Teacher", elem_classes=["primary-button"])

        with gr.Group(visible=False) as student_dashboard:
            gr.HTML(
                '<div class="app-header">'
                '<h1 class="app-title">E-Education System</h1>'
                '<div class="app-subtitle">Student Dashboard - AI-Powered Virtual Tutor</div>'
                '<div style="margin-top: 12px;">'
                '<button class="logout-button" onclick="location.reload()">Logout</button>'
                "</div></div>"
            )
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, elem_classes=["left-panel"]):
                    gr.HTML('<div class="section-header">Learning Materials</div>')
                    with gr.Tabs():
                        with gr.TabItem("Notes"):
                            student_notes = gr.Markdown(lesson_content["notes"])
                        with gr.TabItem("Downloads"):
                            create_simple_downloads()
                        with gr.TabItem("Quiz"):
                            quiz_result = gr.HTML()
                            for i, q in enumerate(lesson_content["quiz"]):
                                gr.Markdown(f"**Q{i + 1}: {q['question']}**")
                                radio = gr.Radio(choices=q["options"], label="", value=None)

                                def make_checker(correct_ans):
                                    def _check(sel):
                                        if sel == correct_ans:
                                            return "<div style='color:#16a34a;font-weight:700;'>Correct.</div>"
                                        return f"<div style='color:#dc2626;font-weight:700;'>Incorrect. Correct answer: {correct_ans}</div>"

                                    return _check

                                radio.change(make_checker(q["answer"]), inputs=[radio], outputs=[quiz_result])
                        with gr.TabItem("Assignment"):
                            student_assignment = gr.Markdown(lesson_content["assignment"])
                        with gr.TabItem("Resources"):
                            student_resources = gr.Markdown(lesson_content["resources"])
                        with gr.TabItem("Extras"):
                            student_extras = gr.Markdown(lesson_content["extras"])
                        with gr.TabItem("Flashcards"):
                            student_flashcards = gr.Markdown(lesson_content["flashcards"])
                        with gr.TabItem("Tips"):
                            student_tips = gr.Markdown(lesson_content["tips"])

                with gr.Column(scale=1, elem_classes=["center-panel"]):
                    gr.HTML('<div class="section-header">Video Library</div>')
                    video_display = gr.Video(label="Lesson Video", height=300, interactive=False)
                    student_video_dropdown = gr.Dropdown(label="Browse Videos", choices=get_video_options())
                    play_video_btn = gr.Button("Play Video", elem_classes=["upload-button"])
                    refresh_student_dropdown_btn = gr.Button("Refresh", elem_classes=["upload-button"])

                with gr.Column(scale=1, elem_classes=["right-panel"]):
                    gr.HTML('<div class="section-header">Ask Doubt (AI Tutor)</div>')
                    ai_video_response = gr.Video(label="AI Teacher Response Video", height=220)
                    ai_text_response = gr.Textbox(label="Detailed Answer", interactive=False, lines=5)
                    student_history_dropdown = gr.Dropdown(label="Recently Asked Questions", choices=get_question_choices())
                    use_previous_btn = gr.Button("Use Selected", elem_classes=["upload-button"])
                    doubt_input = gr.Textbox(label="Your Question", lines=3, placeholder="Type your question here...")
                    submit_doubt_btn = gr.Button("Ask Teacher", elem_classes=["primary-button"])

        # Event wiring
        def show_login_after_splash():
            time.sleep(1.5)
            return {splash_screen: gr.update(visible=False), login_screen: gr.update(visible=True)}

        def handle_login(username, password):
            success, message = authenticate_user(username, password)
            if not success:
                return {login_message: gr.update(value=f'<div style="color:#dc2626;font-weight:600;">{message}</div>')}
            is_teacher = current_user["role"] == "teacher"
            is_student = current_user["role"] == "student"
            load_questions_history()
            q_choices = get_question_choices()
            updates = {
                login_screen: gr.update(visible=False),
                teacher_dashboard: gr.update(visible=is_teacher),
                student_dashboard: gr.update(visible=is_student),
                login_message: gr.update(value=""),
                student_history_dropdown: gr.update(choices=q_choices),
                teacher_history_dropdown: gr.update(choices=q_choices),
            }
            if is_student:
                n_u, _, a_u, r_u, e_u, f_u, t_u = refresh_student_components()
                updates.update(
                    {
                        student_notes: n_u,
                        student_assignment: a_u,
                        student_resources: r_u,
                        student_extras: e_u,
                        student_flashcards: f_u,
                        student_tips: t_u,
                    }
                )
            return updates

        login_outputs = [
            login_screen,
            teacher_dashboard,
            student_dashboard,
            login_message,
            student_notes,
            student_assignment,
            student_resources,
            student_extras,
            student_flashcards,
            student_tips,
            student_history_dropdown,
            teacher_history_dropdown,
        ]
        app.load(show_login_after_splash, outputs=[splash_screen, login_screen])
        login_btn.click(handle_login, inputs=[username_input, password_input], outputs=login_outputs)
        password_input.submit(handle_login, inputs=[username_input, password_input], outputs=login_outputs)

        avatar_img_upload.change(upload_avatar_image, inputs=[avatar_img_upload], outputs=[profile_status])
        generate_bg_btn.click(
            teacher_generate_avatar_bg,
            inputs=[avatar_img_upload, prompt_input, bg_upload],
            outputs=[avatar_img_upload, img_output, download_btn, profile_status],
        )
        avatar_voice_upload.change(upload_avatar_voice, inputs=[avatar_voice_upload], outputs=[profile_status])
        set_profile_btn.click(set_profile, inputs=[prompt_input, bg_upload], outputs=[profile_status])
        script_file_upload.change(handle_script_file_upload, inputs=[script_file_upload], outputs=[teaching_script])
        generate_btn.click(
            teacher_generate_video,
            inputs=[question_input, teaching_script, concept_diagram],
            outputs=[generation_status, teacher_video_display, download_teacher_video_btn],
        )

        def update_notes_and_refresh(notes_text):
            msg = update_notes(notes_text)
            n_u, *_ = refresh_student_components()
            return msg, n_u, n_u

        update_notes_btn.click(update_notes_and_refresh, inputs=[notes_editor], outputs=[notes_status, student_notes_preview, student_notes])

        def update_quiz_opts(o1, o2, o3, o4):
            return gr.update(choices=[o for o in [o1, o2, o3, o4] if o and o.strip()])

        for opt_box in [quiz_option1, quiz_option2, quiz_option3, quiz_option4]:
            opt_box.change(update_quiz_opts, inputs=[quiz_option1, quiz_option2, quiz_option3, quiz_option4], outputs=[quiz_answer])

        def add_quiz_and_refresh(q, o1, o2, o3, o4, ans):
            msg = add_quiz_question(q, o1, o2, o3, o4, ans)
            _, q_u, *_ = refresh_student_components()
            return msg, q_u, gr.update(choices=get_quiz_question_choices(), value=None)

        add_quiz_btn.click(
            add_quiz_and_refresh,
            inputs=[quiz_question, quiz_option1, quiz_option2, quiz_option3, quiz_option4, quiz_answer],
            outputs=[quiz_status, student_quiz_preview, quiz_delete_dropdown],
        )

        def delete_quiz_and_refresh(sel):
            msg, dd = delete_quiz_question(sel)
            _, q_u, *_ = refresh_student_components()
            return msg, dd, q_u

        delete_quiz_btn.click(delete_quiz_and_refresh, inputs=[quiz_delete_dropdown], outputs=[delete_quiz_status, quiz_delete_dropdown, student_quiz_preview])

        play_teacher_video_btn.click(play_selected_video, inputs=[teacher_video_dropdown], outputs=[teacher_video_display])
        refresh_teacher_dropdown_btn.click(refresh_video_dropdown, outputs=[teacher_video_dropdown])
        play_video_btn.click(play_selected_video, inputs=[student_video_dropdown], outputs=[video_display])
        refresh_student_dropdown_btn.click(refresh_video_dropdown, outputs=[student_video_dropdown])
        use_previous_btn.click(select_from_dropdown, inputs=[student_history_dropdown], outputs=[doubt_input])
        teacher_use_previous_btn.click(select_from_dropdown, inputs=[teacher_history_dropdown], outputs=[teacher_question_input])

        def submit_with_update(question):
            vid, txt = student_query_flow(question)
            ch = get_question_choices()
            return vid, txt, gr.update(choices=ch), gr.update(choices=ch)

        submit_doubt_btn.click(
            submit_with_update,
            inputs=[doubt_input],
            outputs=[ai_video_response, ai_text_response, student_history_dropdown, teacher_history_dropdown],
        )
        submit_doubt_btn.click(refresh_video_dropdown, outputs=[student_video_dropdown])
        teacher_ask_btn.click(
            submit_with_update,
            inputs=[teacher_question_input],
            outputs=[teacher_ai_video, teacher_ai_response, student_history_dropdown, teacher_history_dropdown],
        )
        teacher_ask_btn.click(refresh_video_dropdown, outputs=[teacher_video_dropdown])

    return app


if __name__ == "__main__":
    init_models()
    demo = create_complete_app()
    demo.launch(
        share=True,
        allowed_paths=[
            TMP_DIR,
            BASE_DATA_DIR,
            "/tmp",
            os.path.join(SHARED_FILES_DIR, "notes"),
            os.path.join(SHARED_FILES_DIR, "assignments"),
            os.path.join(SHARED_FILES_DIR, "resources"),
        ],
        server_name="0.0.0.0",
        server_port=7860,
    )
