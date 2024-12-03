import time
from datasets import load_dataset

import os
os.environ["HF_DATASETS_OFFLINE"] = "1"

# dataset = load_dataset('HuggingFaceM4/VQAv2', split="train[:100%]")
# dataset = dataset.remove_columns(["question_type", "answers", "answer_type", "image_id", "question_id"])
# split_ds = dataset.train_test_split(test_size=0.10)
# print(len(split_ds["train"]))
# print(len(split_ds["test"]))


# Load model directly
# from transformers import AutoProcessor, AutoModelForImageTextToText

# path = "/home/jerryli/CS228-Project/paligemma-3b-pt-224"
# processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224", cache_dir = path)
# model = AutoModelForImageTextToText.from_pretrained("google/paligemma-3b-pt-224", cache_dir = path)




