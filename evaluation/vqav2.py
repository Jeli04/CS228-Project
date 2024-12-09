import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModel, AutoModelForImageTextToText, PaliGemmaForConditionalGeneration
from peft import PeftModel
from PIL import Image
import requests
import os
import re
import torch
import torch.nn as nn
from datasets import load_dataset

def resize_images(img, target_size=(224, 224)):
    resized_img = img.resize(target_size, Image.Resampling.LANCZOS)

    return resized_img 

def load_finetuned_model(base_model_name, adapter_path, new_weights_path=None):
    # Load the processor/tokenizer
    processor = AutoProcessor.from_pretrained(base_model_name)

    # Step 1: Load the base model from its pretrained weights
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(base_model_name)
    
    # Step 2: If you have additional fine-tuned weights, load them into the base model
    if new_weights_path is not None:
        # hard set the lambda parameters
        for name, param in base_model.named_parameters():
            if ('lambda_q1' in name or
            'lambda_k1' in name or
            'lambda_q2' in name or
            'lambda_k2' in name or
            'subln' in name):
                print(f"Setting custom values for {name}")
                param.data = nn.Parameter(torch.empty(512 // 2, dtype=torch.bfloat16, device="cuda").normal_(mean=0.0, std=0.1))  # Example: Set random values

        # new_weights should be a dict of parameter_name -> tensor
        new_weights = torch.load(new_weights_path, map_location="cpu")

        # Merge any new weights with existing model parameters
        # If the new_weights only contain the updated parameters, you can load them directly:
        adjusted_new_weights = {}
        for k, v in new_weights.items():
            # Remove the "base_model.model." prefix if present
            new_key = k.replace("base_model.model.", "")
            adjusted_new_weights[new_key] = v

        # Now load the adjusted weights into the base model
        base_model.load_state_dict(adjusted_new_weights, strict=False)


    # Step 3: Load the adapter (LoRA or similar) on top of the base model
    # The adapter_path should contain the adapter configuration and weights
    model_with_adapter = PeftModel.from_pretrained(base_model, adapter_path)

    # Move model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_with_adapter.to(device)

    return model_with_adapter, processor


def extract_answer_from_generated_text(question, generated_text):
    """
    Extract the answer from the generated text. 
    The answer comes after the repeated question in the text.
    
    Args:
        question (str): The question asked.
        generated_text (str): The model-generated text.
        
    Returns:
        str: The extracted answer from the generated text.
    """
    # Find the position where the question is repeated
    if question in generated_text:
        # Extract the part of the string after the last occurrence of the question
        split_text = generated_text.split(question)
        
        # We assume the answer appears on the next line after the repeated question
        if len(split_text) > 1:
            possible_answer = split_text[-1].strip().split("\n")[0].strip()  # Take the first non-empty line
            return possible_answer
    return generated_text.strip()  # Fallback to the full generated text if no question is found


def vqav2_evaluate(model, processor, test_dataset):
    """
    Evaluate the model on the VQAv2 dataset using the VQAv2 scoring system.
    
    Args:
        model: The fine-tuned VQA model.
        processor: The processor for the VQA model.
        test_dataset: The test split of the VQAv2 dataset, which contains:
            - "image" (PIL Image)
            - "question" (str)
            - "answers" (list of 10 annotator answers)
            
    Returns:
        float: The overall VQAv2 accuracy score for the test set.
    """
    device = model.device
    total_score = 0
    total_questions = 0
    skipped_images = 0
    
    for example in test_dataset:
        question = example["question"]
        annotator_answers = example["answers"]  # List of 10 annotator answers
        image = example["image"]

        # Check if the image is valid
        try:
            # Ensure image is a single image (not a batch) and has 3 dimensions (H, W, C)
            if not hasattr(image, "size") and not hasattr(image, "mode"):
                print(f"Skipping example due to invalid image type: {type(image)}")
                skipped_images += 1
                continue

            # Skip image if it doesn't have the correct dimensions (e.g., it should be a PIL image or similar)
            if hasattr(image, "size") and len(image.size) != 2 and len(image.size) != 3:
                print(f"Skipping image with unexpected dimensions: {image.size}")
                skipped_images += 1
                continue

            # Resize image
            raw_image = resize_images(image)

            # Prepare inputs for the model
            inputs = processor(text=question, images=raw_image, return_tensors="pt").to(device)

            # Generate output
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=50)  # Increase token limit to handle larger outputs

            # Decode generated text
            generated_text = processor.decode(output[0], skip_special_tokens=True).strip()

            # Extract the answer using the new logic
            predicted_answer = extract_answer_from_generated_text(question, generated_text)

            # Calculate VQAv2 score for this question
            match_count = sum([1 for answer in annotator_answers if answer['answer'].lower() == predicted_answer.lower()])
            score = min(match_count / 3, 1.0)  # VQAv2 scoring formula
            total_score += score
            total_questions += 1

        except Exception as e:
            print(f"Skipping example due to processing error: {e}")
            skipped_images += 1

    accuracy = total_score / total_questions if total_questions > 0 else 0.0
    print(f"Skipped {skipped_images} images due to dimension issues or errors.")
    return accuracy


if __name__ == "__main__":
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    # Load model and processor
    base_model_name = "/home/jerryli/CS228-Project/paligemma-3b-pt-224"
    # adapter_path = "/home/jerryli/CS228-Project/paligemma_vqav2/2024-12-07_19-08-31/checkpoints/checkpoint-23500"
    # new_weights_path = "/home/jerryli/CS228-Project/paligemma_vqav2/2024-12-07_19-08-31/new_params/new_params_step_23500/diff_attention_params.pth"
    adapter_path = "/home/jerryli/CS228-Project/paligemma_vqav2/2024-12-06_12-00-27/checkpoints/checkpoint-12000"
    new_weights_path = "/home/jerryli/CS228-Project/paligemma_vqav2/2024-12-06_12-00-27/new_params/new_params_step_12000/diff_attention_params.pth"

    model, processor = load_finetuned_model(base_model_name, adapter_path, new_weights_path=new_weights_path)

    # Load the VQAv2 dataset
    dataset = load_dataset('HuggingFaceM4/VQAv2', split="train[:100%]")
    dataset = dataset.remove_columns(["question_type", "image_id", "question_id"])
    split_ds = dataset.train_test_split(test_size=0.10)
    test_ds = split_ds["test"]

    # Evaluate the model
    accuracy = vqav2_evaluate(model, processor, test_ds)
    print(f"VQAv2 10% Test Set Accuracy: {accuracy * 100:.2f}%")