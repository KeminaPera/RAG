import os
import sys

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
sys.stdout.reconfigure(encoding='utf-8')

from huggingface_hub import hf_hub_download

model_name = "BAAI/bge-reranker-base"
local_dir = "./bge-reranker-base"

files_to_download = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model"
]

os.makedirs(local_dir, exist_ok=True)

for file_name in files_to_download:
    print("Downloading: " + file_name)
    try:
        hf_hub_download(
            repo_id=model_name,
            filename=file_name,
            local_dir=local_dir,
            resume_download=True
        )
        print("OK: " + file_name + " downloaded")
    except Exception as e:
        print("FAIL: " + file_name + " download failed: " + str(e))

print("\nModel download completed!")