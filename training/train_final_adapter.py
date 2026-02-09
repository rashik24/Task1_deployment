import os
import subprocess
import datetime
import subprocess
import os
import subprocess
import datetime

# -------------------------
# Paths
# -------------------------
LOCAL_OUT = "/tmp/final_adapter"
os.makedirs(LOCAL_OUT, exist_ok=True)

RUN_ID = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M")
GCS_PATH = f"gs://llama-adapters/hours/llama-3.2-1b/{RUN_ID}"

RUN_ID = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M")
GCS_PATH = f"gs://llama-adapters/hours/llama-3.2-1b/{RUN_ID}"

subprocess.run([
    "gsutil", "-m", "cp", "-r",
    LOCAL_OUT,
    GCS_PATH
], check=True)

print("📦 Adapter uploaded to:", GCS_PATH)

LOCAL_OUT = "/tmp/final_adapter"
os.makedirs(LOCAL_OUT, exist_ok=True)

cmd = [
    "llamafactory-cli", "train",
    "--stage", "sft",
    "--do_train",

    "--model_name_or_path", "meta-llama/Llama-3.2-1B-Instruct",

    "--dataset", "hours",
    "--dataset_dir", "training",

    "--template", "llama3",
    "--finetuning_type", "lora",

    "--num_train_epochs", "3",
    "--per_device_train_batch_size", "1",
    "--learning_rate", "2e-4",

    # 🔥 CPU-SAFE FLAGS
    "--device", "cpu",
    "--num_workers", "0",
    "--fp16", "false",
    "--bf16", "false",
    "--torch_dtype", "float32",
    "--gradient_checkpointing", "true",

    "--output_dir", "/tmp/final_train",
    "--overwrite_output_dir",

    "--export_adapter",
    "--adapter_dir", LOCAL_OUT,

    "--report_to", "none"
]

env = os.environ.copy()

# CPU-safe environment
env["OMP_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"
env["TOKENIZERS_PARALLELISM"] = "false"

print("🚀 Training final LoRA adapter (CPU mode)")
subprocess.run(cmd, check=True, env=env)

print("✅ Adapter saved to:", LOCAL_OUT)
