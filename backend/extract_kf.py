import argparse

from src.utils.config import load_config
from src.pipelines.extract_keyframes import ExtractKeyframePipeline

from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

from huggingface_hub import login

if hf_token:
    login(token=hf_token)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/keyframe_extraction.yaml",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    pipeline = ExtractKeyframePipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()