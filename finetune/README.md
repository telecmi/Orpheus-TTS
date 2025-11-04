# 🚀 Orpheus-TTS Fine-Tuning Guide

This guide walks you through downloading data, fine-tuning Orpheus-TTS, merging LoRA weights (if used), and publishing the final model to Hugging Face.

---

## 📦 Repository Contents

| File | Description |
|------|------------|
`hf_data_download.py` | Script to download Hugging Face dataset into cache  
`finetune_orpheus.ipynb` | Notebook for fine-tuning Orpheus-TTS  
`README.md` | Usage and training instructions (this file)

---

## Push model to Huggingface

```bash 
login to huggingface allow the repo to access through your access token for read/write access
After merging (or directly if no LoRA), upload the finished model:

from huggingface_hub import HfApi, upload_folder

api = HfApi()
api.upload_folder(
    folder_path="MERGED_MODEL_PATH",
    repo_id="speechriv/OrpheusTTS-en-v1",   # Change to your HF repo name (First create the repo first otherwise it will throw error)
    repo_type="model",
    path_in_repo=".",
    commit_message="Upload merged Orpheus model",
)
```