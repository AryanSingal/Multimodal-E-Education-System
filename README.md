# Multimodal Generative AI-Based E-Education System

Official implementation for the manuscript:

> **Quality Evaluation of a Generative AI-Based E-Education System**  
> Aryan, Janisha Sharma, Nishu Singh, Anuja Soni, and Vibha Gaur  
> Department of Computer Science, Acharya Narendra Dev College & Deen Dayal Upadhyaya College, University of Delhi, India

## Overview

This repository provides an end-to-end, open-source generative AI e-education platform that converts static instructional material (`.docx`, `.pptx`, or plain text), an instructor portrait, and a short reference voice recording into synchronized, subtitled lecture videos with local student doubt resolution.

The web application is built with Gradio and separated into two role-authenticated dashboards:

- **Teacher Dashboard:** Upload instructor portrait and reference voice, generate or upload a classroom background, convert lesson scripts into lip-synced talking-head videos with optional side-by-side concept diagrams and burned-in captions, and manage shared notes, files, and quizzes.
- **Student Dashboard:** Browse and play generated lecture videos, access notes/downloads/quizzes, and ask real-time doubts answered via a 4-tier knowledge-base search and a local 4-bit quantized `TinyLlama-1.1B` model.

> **Language Scope:** The current release is configured and validated for **English-language instruction** (`language="en"`).

---

## System Pipeline

1. **Input & Portrait Preprocessing:** Extracts UTF-8 text from `.docx`/`.pptx`/plain text and center-crops the portrait to `256 x 256` RGB via Lanczos interpolation.
2. **Classroom Background Generation & Matting:** Generates a `512 x 512` classroom scene with Stable Diffusion v1.5 (FP16, CFG `7.5`, `25–50` steps), segments the foreground with Rembg (`u2netp`), and composites with 3-pixel Gaussian alpha feathering.
3. **Speech Synthesis & Voice Cloning:** Synthesizes `22,050 Hz` 16-bit mono English speech using Coqui Tacotron2-DDC + HiFi-GAN (`v0.24.2`) or YourTTS voice cloning.
4. **3DMM Facial Animation & Restoration:** Renders `25 fps` lip-synced talking-head video with SadTalker (`v0.0.2`, `--preprocess full --still --size 256 --batch_size 4 --expression_scale 0.7`) and GFPGAN (`v1.4`).
5. **Layout & Subtitle Burn-In:** Horizontally stacks optional concept diagrams via FFmpeg `hstack` (even-pixel padded for `yuv420p`) and burns Whisper (`base`) `.srt` captions into `H.264` (CRF `23`) / `AAC 128k` MP4.
6. **Local Doubt Resolution:** Searches `knowledge_base.json` using 4-tier lexical-semantic scoring and falls back to zero-shot on-device `TinyLlama-1.1B-Chat-v1.0 (Q4_K_M.gguf)` inference (`gpu_layers=20`, `temperature=0.7`) without external API calls.

---

## Upstream Repositories, Models & Versions

All third-party open-source models, checkpoints, and libraries integrated in this project are listed below with their exact versions and official sources:

| Component | Version / Checkpoint | Official Repository / Model Source |
| :--- | :--- | :--- |
| **SadTalker** | `v0.0.2-rc` (`SadTalker_V0.0.2_256.safetensors`, `mapping_00109/00229-model.pth.tar`) | [OpenTalker/SadTalker](https://github.com/OpenTalker/SadTalker) |
| **GFPGAN** | `v1.4` (`GFPGANv1.4.pth`) | [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) |
| **BasicSR** | `v1.4.2` | [XPixelGroup/BasicSR](https://github.com/XPixelGroup/BasicSR) |
| **FaceXLib** | `v0.3.0` (`alignment_WFLW_4HG.pth`, `detection_Resnet50_Final.pth`, `parsing_parsenet.pth`) | [xinntao/facexlib](https://github.com/xinntao/facexlib) |
| **Coqui TTS** | `coqui-tts==0.24.2` (`tacotron2-DDC` + HiFi-GAN, `your_tts`) | [idiap/coqui-ai-TTS](https://github.com/idiap/coqui-ai-TTS) |
| **OpenAI Whisper** | `base` (74M parameters) | [openai/whisper](https://github.com/openai/whisper) |
| **Stable Diffusion** | `runwayml/stable-diffusion-v1-5` (FP16) | [Hugging Face: stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) / [huggingface/diffusers](https://github.com/huggingface/diffusers) |
| **Rembg** | `v2.0.50+` (`u2netp` lightweight session) | [danielgatis/rembg](https://github.com/danielgatis/rembg) |
| **TinyLlama-1.1B** | `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` (4-bit GGUF, 669 MB) | [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) / [jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama) |
| **CTransformers** | `>=0.2.27` | [marella/ctransformers](https://github.com/marella/ctransformers) |
| **FFmpeg** | `v4.4.2+` (`libx264`, `aac`, `hstack`, `subtitles`) | [FFmpeg/FFmpeg](https://github.com/FFmpeg/FFmpeg) |
| **Gradio** | `>=4.20.0` | [gradio-app/gradio](https://github.com/gradio-app/gradio) |

---

## Tested Hardware & Software Environment

The system was developed and validated under the following environment (Google Colaboratory):

- **OS:** Ubuntu 22.04 LTS
- **GPU:** 1x NVIDIA Tesla T4 (15 GB GDDR6 VRAM, CUDA 12.2)
- **CPU / RAM:** 2-core Intel Xeon @ 2.20 GHz, 12.7 GB System RAM
- **Python / Framework:** Python 3.10+, PyTorch 2.x
- **System Binaries:** `ffmpeg`, `espeak-ng`, `git`, `libportaudio2`

---

## Selected Module Hyperparameters

| Module | Parameter | Selected Setting |
| :--- | :--- | :--- |
| Coqui Tacotron2-DDC + HiFi-GAN | Sampling rate, Mel bands, FFT, silence trim | `22,050 Hz`, `80` Mel bands, `FFT 1024`, `60 dB` trim |
| SadTalker + GFPGAN | Pose mode, expression scale, resolution, batch | `--preprocess full --still`, `exp_scale=0.7`, `256 x 256`, `batch_size=4`, `GFPGANv1.4` |
| Stable Diffusion v1.5 + Rembg | Guidance scale, steps, resolution, matting | `CFG 7.5`, `25–50` steps, `512 x 512`, `u2netp` + 3-px feathering |
| Whisper + FFmpeg | ASR model, subtitle style, video/audio codec | `whisper-base`, 12 pt bottom-center burned-in `.srt`, `H.264 CRF 23` + `AAC 128k` |
| TinyLlama-1.1B | Quantization, GPU offload, sampling temp | `4-bit Q4_K_M` (~890 MB VRAM), `gpu_layers=20`, `temperature=0.7` (zero-shot) |

---

## Repository Structure

```text
Multimodal-E-Education-System/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── assets/
│   └── sample_teacher_portrait.png
├── data/
│   └── knowledge_base_sample.json
├── notebooks/
│   └── E_Education_System.ipynb
├── scripts/
│   └── setup_models.sh
└── src/
    ├── app.py
    └── sadtalker_patches.py
```

---

## Quick Start

### Option A: Google Colab (Recommended)

1. Open `notebooks/E_Education_System.ipynb` in Google Colab.
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. Run Cells 1 through 5 sequentially.
4. Open the generated Gradio URL and sign in with the demo accounts:
   - **Teacher:** `teacher` / `teach123`
   - **Student:** `student` / `learn123`

### Option B: Local Linux / CUDA Workstation

```bash
git clone https://github.com/aryansingal823/Multimodal-E-Education-System.git
cd Multimodal-E-Education-System

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash scripts/setup_models.sh
python3 src/app.py
```

---

## Ethical Considerations & Safeguards

- **Transparency:** The interface explicitly labels generated lectures and Q&A outputs as an **AI-Powered Virtual Tutor**.
- **Uncanny Valley Mitigation:** Uses damped 3DMM expression scaling (`--expression_scale 0.7 --still`) and GFPGAN restoration to reduce unnatural facial distortion.
- **Privacy:** Instructor portraits, voice samples, and student doubt queries are processed locally on the host runtime without transmitting student data to external commercial LLM APIs.

---
