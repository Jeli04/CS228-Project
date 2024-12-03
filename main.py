import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
from peft import PeftModel


def load_finetuned_model(base_model_name, adapter_path):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load the base model
    base_model = AutoModelForImageTextToText.from_pretrained(base_model_name)

    # Load the LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # Move model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer

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
    device = next(model.parameters()).device
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Generate the output
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        num_beams=num_beams,
        temperature=temperature,
        early_stopping=True
    )

    # Decode and return the generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    # Define the base model and adapter paths
    base_model_name = "google/paligemma-3b-pt-224"  # actual model name
    adapter_path = "/home/jerryli/CS228-Project/paligemma_vqav2/"  

    # Load the model and tokenizer
    model, tokenizer = load_finetuned_model(base_model_name, adapter_path)

    # Input text
    input_text = "<image> The future of AI is"

    # Generate text
    generated_text = generate_text(model, tokenizer, input_text, max_length=100)
    print("Generated Text:", generated_text)
