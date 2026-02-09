import os
import json
import subprocess
import time
import fsspec

# =============================
# Config
# =============================
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

GCS_BUCKET = "gs://oceanic-sky-486903-n2-llm-data"
DATA_PATH = f"{GCS_BUCKET}/hours_test.jsonl"
FINAL_GCS_OUT = f"{GCS_BUCKET}/final_adapter"

LOCAL_WORKDIR = "/tmp/llama_final"
LOCAL_DATA_DIR = f"{LOCAL_WORKDIR}/data"
LOCAL_OUT_DIR = f"{LOCAL_WORKDIR}/adapter"

os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
os.makedirs(LOCAL_OUT_DIR, exist_ok=True)

os.environ["MKL_THREADING_LAYER"] = "GNU"

start = time.time()

# =============================
# Load full dataset
# =============================
print("📥 Loading dataset from GCS...")
with fsspec.open(DATA_PATH, "r") as f:
    data = [json.loads(line) for line in f]

# =============================
# Write full_train.jsonl
# =============================
full_train_path = f"{LOCAL_DATA_DIR}/full_train.jsonl"
with open(full_train_path, "w", encoding="utf-8") as f:
    for rec in data:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# =============================
# Write dataset_info.json
# =============================
dataset_info = {
    "full_train": {
        "path": LOCAL_DATA_DIR,
        "file_name": "full_train.jsonl",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "system": None},
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "user",
            "assistant_tag": "assistant"
        }
    }
}

dataset_info_path = f"{LOCAL_DATA_DIR}/dataset_info.json"
with open(dataset_info_path, "w", encoding="utf-8") as f:
    json.dump(dataset_info, f, indent=2, ensure_ascii=False)

print("✅ Dataset prepared")

# =============================
# Train LoRA model (FULL DATA)
# =============================
cmd = [
    "llamafactory-cli", "train",
    "--stage", "sft",
    "--do_train",
    "--model_name_or_path", MODEL_NAME,
    "--dataset", "full_train",
    "--dataset_dir", LOCAL_DATA_DIR,
    "--template", "llama3",
    "--finetuning_type", "lora",
    "--num_train_epochs", "3",
    "--per_device_train_batch_size", "1",
    "--learning_rate", "2e-4",
    "--logging_steps", "10",
    "--overwrite_output_dir",
    "--save_only_model",            # ⭐ saves ONLY adapters
    "--output_dir", LOCAL_OUT_DIR,
    "--report_to", "none"
]

print("🚀 Training final model...")
subprocess.run(cmd, check=True)

# =============================
# Upload adapter to GCS
# =============================
print("☁️ Uploading adapter to GCS...")
fs = fsspec.filesystem("gs")
fs.put(
    LOCAL_OUT_DIR,
    FINAL_GCS_OUT,
    recursive=True
)

print("✅ Training complete")
print(f"📦 Adapter saved at: {FINAL_GCS_OUT}")
print(f"⏱ Runtime: {time.time() - start:.2f} seconds")
