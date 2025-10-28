import os
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path='./bin/',
    repo_id="Frustrated-B4S1C/gpt-2-from-scratch",
    repo_type="model",
    commit_message="Added PyTorch Implementation"
)
