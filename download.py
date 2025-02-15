from modelscope import snapshot_download

# Specify the directory where you want the model to be downloaded
download_dir = "/root/autodl-tmp"

# Download the model to the specified directory
model_id = snapshot_download('AI-ModelScope/stable-diffusion-3.5-large', cache_dir=download_dir)
# model_dir = snapshot_download('muse/google_t5_v1_1_xxl', cache_dir=download_dir)
