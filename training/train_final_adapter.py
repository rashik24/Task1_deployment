# training/train_final_adapter.py

import os
import subprocess
import datetime

# -------------------------
# Config
# -------------------------
LOCAL_OUT = "/tmp/final_adapter"
os.makedirs(LOCAL_OUT, exist_ok=True)

RUN_ID = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M")
GCS_BUCKET = "gs://llama-adapters"  # change if your bucket name differs
GCS_PATH = f"{GCS_BUCKET}/hours/llama-3.2-1b/{RUN_ID}"

# -------------------------
# Train command (CPU)
# -------------------------
cmd = [
    "llamafactory-cli", "train",
    "--stage", "sft",
    "--do_train",

    "--model_name_or_path", "meta-llama/Llama-3.2-1B-Instruct",

    "--dataset", "hours_test",
    "--dataset_dir", "training",




    "--template", "llama3",
    "--finetuning_type", "lora",

    "--num_train_epochs", "3",
    "--per_device_train_batch_size", "1",
    "--learning_rate", "2e-4",

   
  
    "--fp16", "false",
    "--bf16", "false",
    
    "--gradient_checkpointing", "true",

    "--output_dir", "/tmp/final_train",
    "--overwrite_output_dir",

    
   

    "--report_to", "none"
]

env = os.environ.copy()
env["OMP_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"
env["TOKENIZERS_PARALLELISM"] = "false"

print("🚀 Training final LoRA adapter (CPU mode)")
subprocess.run(cmd, check=True, env=env)

print("✅ Training finished. Adapter directory:", LOCAL_OUT)

# -------------------------
# Verify adapter export
# -------------------------
if not os.path.exists(LOCAL_OUT):
    raise RuntimeError(f"❌ Adapter directory does not exist: {LOCAL_OUT}")

files = os.listdir(LOCAL_OUT)
if len(files) == 0:
    raise RuntimeError(
        f"❌ Adapter directory is empty. Training did not export adapter to {LOCAL_OUT}"
    )

print("✅ Adapter files found:", files)

# -------------------------
# Upload to GCS
# -------------------------
print(f"📦 Uploading adapter to: {GCS_PATH}")
subprocess.run(
    ["gsutil", "-m", "cp", "-r", LOCAL_OUT, GCS_PATH],
    check=True
)

print("✅ Adapter uploaded to:", GCS_PATH)
