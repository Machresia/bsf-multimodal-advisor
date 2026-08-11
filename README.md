<div align="center">

# Black Soldier Fly (BSF) Multimodal Advisor

**An offline, vision-guided RAG pipeline for smart BSF colony management and tray diagnostic assistance.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/RAG-LlamaIndex-orange.svg)](https://www.llamaindex.ai/)
[![Vision Model](https://img.shields.io/badge/Vision-Fine--Tuned%20SigLIP-purple.svg)](https://huggingface.co/google/siglip-base-patch16-224)
[![Speech Model](https://img.shields.io/badge/Audio-Faster--Whisper-green.svg)](https://github.com/SYSTRAN/faster-whisper)
[![LLM Backend](https://img.shields.io/badge/LLM-Ollama%20(Gemma3:4b)-black.svg)](https://ollama.com/)

---

</div>

## Overview

Managing Black Soldier Fly (*Hermetia illucens*) rearing trays requires constant monitoring of moisture levels, feed ratios, and lifecycle progression. 

The **BSF Multimodal Advisor** is an edge-ready assistant that combines a **fine-tuned SigLIP vision transformer** with a **local LlamaIndex Retrieval-Augmented Generation (RAG) engine**. It delivers real-time diagnostic feedback and tailored troubleshooting advice directly in your terminal—**completely offline with zero API costs.**

---

## Key Features

* **Specialized Visual Classification:** Utilizes a fine-tuned `google/siglip-base-patch16-224` vision checkpoint to identify BSF developmental stages and tray conditions.
* **Local Speech Transcription:** Integrates `faster-whisper` for offline processing and automatic transcription of farmer voice notes (`.wav`, `.mp3`, `.m4a`).
* **Modular Multi-Input Prompt Building:** Leverages dedicated audio processing (`audio_processor.py`) and prompt synthesis (`prompt_builder.py`) modules to cleanly fuse visual, voice, and text observations into unified RAG queries.
* **Local Context-Aware RAG:** Integrates `LlamaIndex` and local vector embeddings (`BAAI/bge-small-en-v1.5`) to pull verified advisory knowledge.
* **Privacy-First & Offline:** Runs entirely on local hardware using `Ollama` (`gemma3:4b`) and Hugging Face `transformers`.

---

## System Architecture

```text
  ┌────────────────────────┐    ┌────────────────────────┐
  │   Tray Image Input     │    │   Voice Note Input     │
  └───────────┬────────────┘    └───────────┬────────────┘
              │                             │
              ▼                             ▼
  ┌────────────────────────┐    ┌────────────────────────┐
  │  SigLIP Classifier     │    │ Faster-Whisper Engine  │
  │ (Lifecycle Condition)  │    │ (Speech-to-Text Trans) │
  └───────────┬────────────┘    └───────────┬────────────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                             ▼
              ┌────────────────────────────┐
              │ Dynamic Prompt Builder     │  ──► Fuses Image, Voice & Text Queries
              │   (prompt_builder.py)      │
              └──────────────┬─────────────┘
                             │ (Injected Context)
                             ▼
              ┌────────────────────────────┐
              │ LlamaIndex Knowledge Base  │  ──► Retrieves Contextual Advisory Data
              │ (Embedding: bge-small-v1.5)│
              └──────────────┬─────────────┘
                             │
                             ▼
              ┌────────────────────────────┐
              │  Local Ollama (gemma3:4b)  │  ──► Generates Actionable Advice
              └────────────────────────────┘

```
---

## Installation & Setup

### 1. Prerequisites
Ensure **Python 3.10+** and **Ollama** are installed. Pull the LLM base:
```bash
ollama pull gemma3:4b
```

### 2. Environment Setup
```bash
# Activate your environment
source venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Advisor
```bash
python app.py
```