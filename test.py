import time
from datasets import load_dataset

import os
os.environ["HF_DATASETS_OFFLINE"] = "1"

# ds = load_dataset('HuggingFaceM4/VQAv2', split="train[:5%]")

# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText

path = "/home/jerryli/CS228-Project/paligemma-3b-pt-224"
processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224", cache_dir = path)
model = AutoModelForImageTextToText.from_pretrained("google/paligemma-3b-pt-224", cache_dir = path)