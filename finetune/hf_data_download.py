import os
import pandas as pd
from huggingface_hub import hf_hub_download
import librosa
import time, random
import concurrent.futures

# Set your Hugging Face dataset info
REPO_ID = "speechriv/tts_common_hindi"
METADATA_FILENAME = "Metadata.csv"
TOKEN = os.environ.get("HF_TOKEN", "PUT YOUR HF TOKEN HERE")
CACHE_DIR = "/home/user/voice/Orpheus-TTS/finetune/hf_cache"


def download_and_load_audio(audio_path):
    for attempt in range(5):
        try:
            local_audio_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=audio_path,
                repo_type="dataset",
                token=TOKEN,
                cache_dir=CACHE_DIR
            )
            wav, sr = librosa.load(local_audio_path, sr=None)
            print(f"[OK] {audio_path}: loaded with shape={wav.shape}, sr={sr}")
            return local_audio_path
        except Exception as e:
            print(f"[WARN] Attempt {attempt+1}: {e}")
            time.sleep(2 + random.random() * 3)  # exponential backoff is better
    print(f"[FAIL] Could not download {audio_path}")
    return None


def main():
    print("==== HuggingFace Dataset Fetch Test ====")
    print(f"Repo: {REPO_ID}")

    # 1. Download metadata CSV
    try:
        metadata_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=METADATA_FILENAME,
            repo_type="dataset",
            token=TOKEN,
            cache_dir=CACHE_DIR
        )
        print(f"[OK] Downloaded metadata.csv: {metadata_path}")
    except Exception as e:
        print(f"[FAIL] Error downloading metadata.csv: {e}")
        return

    # 2. Read the CSV and show a sample
    try:
        df = pd.read_csv(metadata_path)
        print(f"[OK] Read {len(df)} rows in metadata.csv")
        print("Sample row(s):")
        print(df.head(3))
    except Exception as e:
        print(f"[FAIL] Error reading metadata.csv: {e}")
        return

    # 3. Parallel download and test-load audio files
    local_audio_paths = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download_and_load_audio, row['PATH']) for _, row in df.iterrows()]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                local_audio_paths.append(result)

    # Print downloaded paths
    for path in local_audio_paths:
        print(path)
        print("############")

if __name__ == "__main__":
    main()




