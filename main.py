import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModel, AutoModelForImageTextToText, PaliGemmaForConditionalGeneration
from peft import PeftModel
from PIL import Image
import requests
import os
import re
import torch
import torch.nn as nn

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

def generate_text(model, tokenizer, input_text, max_length=50, num_beams=3, temperature=1.0):
    """
    Generates text from the fine-tuned PaliGemma model.

    Args:
    - model: The fine-tuned PaliGemma model.
    - tokenizer: The tokenizer for the model.
    - input_text (str): The input text for generation.
    - max_length (int): The maximum length of the generated text.
    - num_beams (int): The number of beams for beam search (for better diversity).
    - temperature (float): Sampling temperature (higher = more random).

    Returns:
    - generated_text (str): The generated text.
    """
    # Tokenize the input text and move to the appropriate device
    inputs = processor(prompt, raw_image.convert("RGB"), return_tensors="pt").to(model.device)

    # Generate the output
    output = model.generate(**inputs, max_new_tokens=20)

    generated_text = processor.decode(output[0], skip_special_tokens=True)[len(prompt):]
    return generated_text

if __name__ == "__main__":
    # Define the base model and adapter paths
    base_model_name = "/home/jerryli/CS228-Project/paligemma-3b-pt-224" # "google/paligemma-3b-pt-224"  # actual model name
    adapter_path = "/home/jerryli/CS228-Project/paligemma_vqav2/2024-12-07_19-08-31/checkpoints/checkpoint-23500"  
    new_weights_path = "/home/jerryli/CS228-Project/paligemma_vqav2/2024-12-07_19-08-31/new_params/new_params_step_23500/diff_attention_params.pth"
    
    # model_path  = "/home/jerryli/CS228-Project/paligemma-3b-pt-224" 
    # adapter_path = "/home/jerryli/CS228-Project/paligemma_vqav2/no_diff_attn/checkpoint-400"  
    # new_weights_path = None

    # Folder containing images
    image_folder = "/home/jerryli/CS228-Project-images"

    # List of prompts corresponding to the images
    prompts = [
        "Where is the caption. The answers are: Top or Bottom. Caption: A person sitting with luggages",
        "Where is the caption. The answers are: Top or Bottom. Caption: A platter with waffles",
        "Where is the caption. The answers are: Top or Bottom. Caption: A hotdog meal",
        "Where is the caption. The answers are: Top or Bottom. Caption: A closed toilet",
        "Where is the caption. The answers are: Top or Bottom. Caption: A boat in the ocean",
        "Where is the caption. The answers are: Top or Bottom. Caption: A white bus",
        "Where is the caption. The answers are: Top or Bottom. Caption: A red fire hydrant",
        "Where is the caption. The answers are: Top or Bottom. Caption: A green army jeep",
        "Where is the caption. The answers are: Top or Bottom. Caption: A baby with a toothbrush",
        "Where is the caption. The answers are: Top or Bottom. Caption: A bed with two pillows",
        "Where is the caption. The answers are: Top or Bottom. Caption: A cup of coffee",
        "Where is the caption. The answers are: Top or Bottom. Caption: A bear in the snows",
        "Where is the caption. The answers are: Top or Bottom. Caption: A red ferry",
        "Where is the caption. The answers are: Top or Bottom. Caption: A man on a bike",
        "Where is the caption. The answers are: Top or Bottom. Caption: A pack of giraffes",
        "Where is the caption. The answers are: Top or Bottom. Caption: A green bus"
    ]

    # Load the model and processor
    model, processor = load_finetuned_model(base_model_name, adapter_path, new_weights_path=new_weights_path)

    # Check if the number of images matches the number of prompts
    # Custom sorting function to extract the number at the end of each file name
    def extract_number(file_name):
        match = re.search(r'(\d+)(?!.*\d)', file_name)  # Find the last number in the file name
        return int(match.group(1)) if match else float('inf')  # Use infinity if no number is found

    # Sort files based on the extracted number
    image_files = sorted(os.listdir(image_folder), key=extract_number)
    print(len(image_files))
    print(len(prompts))
    if len(image_files) != len(prompts):
        raise ValueError("The number of images and prompts must be the same.")

    for image_file, prompt in zip(image_files, prompts):
        # Construct the full path for the image file
        image_path = os.path.join(image_folder, image_file)

        # Process the image
        raw_image = resize_images(Image.open(image_path))

        # Generate text
        generated_text = generate_text(model, processor, prompt, max_length=20)

        # Print results
        print("Image:", image_file)
        print("Prompt:", prompt)
        print("Generated Text:", generated_text)
        print("-" * 50)
    
    prompt = "The future of ai is "
    generated_text = generate_text(model, processor, prompt, max_length=20)
    print("Prompt:", prompt)
    print("Generated Text:", generated_text)

    prompt = "What is behind the cat?"
    image_file = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cat.png?download=true"
    raw_image = Image.open(requests.get(image_file, stream=True).raw)
    generated_text = generate_text(model, processor, prompt, max_length=20)
    print("Prompt:", prompt)
    print("Generated Text:", generated_text)


