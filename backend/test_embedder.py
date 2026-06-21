import os
import torch
import open_clip
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    print("HF token loaded from .env")
else:
    print("No HF token found in .env")

MODEL_NAME = "ViT-L-16-SigLIP-256"
PRETRAINED = "webli"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=PRETRAINED,
    device=DEVICE,
    precision="fp32",
)

tokenizer = open_clip.get_tokenizer(MODEL_NAME)

tokens = tokenizer(["a person riding a bicycle"])

print("DEVICE:", DEVICE)
print("TOKENS SHAPE:", tokens.shape)
print("OK")