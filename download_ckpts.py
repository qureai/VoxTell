from huggingface_hub import snapshot_download

MODEL_NAME = "voxtell_v1.1" # Updated models may be available in the future
DOWNLOAD_DIR = "/raid13/mohammad.fahim/repos/VoxTell/ckpts" # Optionally specify the download directory

download_path = snapshot_download(
repo_id="mrokuss/VoxTell",
allow_patterns=[f"{MODEL_NAME}/*", "*.json"],
local_dir=DOWNLOAD_DIR
)
