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
# Verify adapter files (in output_dir)
# -------------------------
OUTPUT_DIR = "/tmp/final_train"

required_files = [
    "adapter_model.safetensors",
    "adapter_config.json"
]

missing = [
    f for f in required_files
    if not os.path.exists(os.path.join(OUTPUT_DIR, f))
]

if missing:
    raise RuntimeError(
        f"❌ Missing adapter files in {OUTPUT_DIR}: {missing}"
    )

print("✅ Adapter files found in output_dir:", required_files)


# -------------------------
# Upload adapter to GCS
# -------------------------
RUN_ID = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M")
GCS_PATH = f"gs://llama-adapters/hours/llama-3.2-1b/{RUN_ID}"

subprocess.run(
    ["gsutil", "-m", "cp", "-r", OUTPUT_DIR, GCS_PATH],
    check=True
)

print("📦 Adapter uploaded to:", GCS_PATH)

