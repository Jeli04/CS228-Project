import torch
import torch.nn as nn
import json
# from processing_paligemma import PaliGemmaProcessor
from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration, PaliGemmaConfig
# from transformers import PaliGemmaForConditionalGeneration
from transformers import BitsAndBytesConfig, Trainer, TrainingArguments, GenerationConfig, AutoTokenizer
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
from pathlib import Path
from safetensors.torch import load_file
#from config_utils import PaliGemmaConfig
from transformers import PaliGemmaProcessor
import os
from datetime import datetime

device = "cuda"

def initialize_new_layers(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

def setup(local_weights_path, config):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = PaliGemmaForConditionalGeneration(config, bnb_config)

    # load huggingface weights 
    state_dict = {}
    weights_path = Path(local_weights_path)
    safetensors_files = list(weights_path.glob("*.safetensors"))
    for file in safetensors_files:
        state_dict.update(load_file(file))

    # filter out missing keys
    language_model_prefix = "language_model.model"
    vision_encoder_prefix = "vision_tower.vision_model"

    # Create a new state_dict that maps only the matching keys
    pretrained_language_state_dict = {
        k.replace(language_model_prefix + ".", ""): v
        for k, v in state_dict.items()
        if k.startswith(language_model_prefix + ".")
    }
    pretrained_vision_state_dict = {
        k.replace(vision_encoder_prefix + ".", ""): v
        for k, v in state_dict.items()
        if k.startswith(vision_encoder_prefix + ".")
    }

    # Load the state_dict into the language_model
    model.language_model.model.load_state_dict(pretrained_language_state_dict, strict=False)
    model.vision_tower.vision_model.load_state_dict(pretrained_vision_state_dict, strict=False)

    # Initialize new layers (Just swiglu)
    initialize_new_layers(model)

    # Monkey-patch the config to add a 'get' method
    if not hasattr(model.config, 'get'):
        def config_get(key, default=None):
            return getattr(model.config, key, default)
        model.config.get = config_get

    # Optionally, ensure 'tie_word_embeddings' exists
    if not hasattr(model.config, 'tie_word_embeddings'):
        model.config.tie_word_embeddings = False  # or True, based on your needs

    # Setup LORA config
    lora_config = LoraConfig(
        r=16,  # Rank of the adaptation matrices
        lora_alpha=32,  # Scaling factor
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "out_proj",
        ],  
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"  # Adjust based on your task
    )
    # Inspect module names
    # for name, module in model.named_modules():
    #     print(name)

    model = get_peft_model(model, lora_config)
    # for name, param in model.named_parameters():
    #     if "lora" in name:
    #         print(f"LoRA parameter: {name}, requires_grad: {param.requires_grad}")
    model.print_trainable_parameters()  

    return model

def load_docvqa():
    ds = load_dataset("lmms-lab/DocVQA", "InfographicVQA", split="test")
    print(ds)

def load_vqav2():
    ds = load_dataset('HuggingFaceM4/VQAv2', split="train[:100%]")
    cols_remove = ["question_type", "answers", "answer_type", "image_id", "question_id"]
    ds = ds.remove_columns(cols_remove)

    split_ds = ds.train_test_split(test_size=0.10) # we'll use a very small split for demo
    train_ds = split_ds["test"]
    return train_ds
    
def vqav2_collate_fn(examples): # dictionary of images and text
    # only supports single image for now, if we want multple multple image_token
    texts = [
        f"{' '.join([str(image_token)])} {bos_token} answer {example['question']}"
        for example in examples
    ]
    labels= [example['multiple_choice_answer'] for example in examples]
    images = [example["image"].convert("RGB") for example in examples]
    tokens = processor(text=texts, images=images, suffix=labels,
                        return_tensors="pt", padding="longest",
                        tokenize_newline_separately=False)

    for keys, values in tokens.items():
        values = values.to(torch.bfloat16).to(device)
    return tokens

def generate(prompt: str, local_weights_path, lora_weights_path, model_config, tokenizer, processor):
    """
    Generate a response based on a given prompt using the local weights and LoRA weights.

    Args:
        prompt (str): The input prompt.
        local_weights_path (str): Path to the local model weights.
        lora_weights_path (str): Path to the LoRA adapter weights.
        model_config (PaliGemmaConfig): Model configuration.
        tokenizer (AutoTokenizer): Tokenizer for the model.
        processor (PaliGemmaProcessor): Processor for images and text.

    Returns:
        str: The generated response.
    """
    # Define generation parameters
    generation_config = GenerationConfig(
        max_length=50,  # Maximum length of the generated text
        num_beams=5,  # Number of beams for beam search
        early_stopping=True  # Stop early if all beams converge
    )

    # Tokenize the input prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Dummy image input
    pixel_values = torch.randn(1, 3, 224, 224).to(device)  # Replace with actual images if required
    attention_mask = torch.ones_like(input_ids).to(device)

    # Initialize the model
    model = setup(local_weights_path, model_config)
    model = model.to(device)

    # Load the LoRA weights
    lora_state_dict = load_file(lora_weights_path)
    model.load_state_dict(lora_state_dict, strict=False)

    # Set the generation configuration
    model.generation_config = generation_config

    # Generate text
    generated_ids = model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask
    )

    # Decode the generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_text

def finetune_lora(local_weights_path, model_config, train_ds, collate_fn, training_args):
    # intialize the model and freeze/unfreeze weights
    model = setup(local_weights_path, model_config)

    # intialize the trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_ds ,
        data_collator=collate_fn,
        args=training_args
        )
    
    # train
    print("Begin Finetuning")
    trainer.train()

    # save weights
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_folder = os.path.dirname("/home/jerryli/CS228-Project/paligemma_vqav2/finetuned_weights")  # Parent folder of local weights path
    save_folder = os.path.join(base_folder, f"finetuned_paligemma_{timestamp}")
    os.makedirs(save_folder, exist_ok=True)

    model.save_pretrained(save_folder)
    print(f"Fine-tuned model saved to: {save_folder}")

    # Save the model config
    config_path = os.path.join(save_folder, "model_config.json")
    with open(config_path, "w") as config_file:
        json.dump(model_config.to_dict(), config_file, indent=4)
    print(f"Model configuration saved to: {config_path}")

    # Save the training arguments
    train_args_path = os.path.join(save_folder, "training_args.json")
    training_args_dict = training_args.to_dict()
    with open(train_args_path, "w") as train_args_file:
        json.dump(training_args_dict, train_args_file, indent=4)
    print(f"Training arguments saved to: {train_args_path}")


def load_tokenizer(model_path):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"
    return tokenizer

if __name__ == "__main__":
    # Specify the local path to the model weights
    root = "/home/jerryli/CS228-Project/paligemma-3b-pt-224/models--google--paligemma-3b-pt-224/snapshots/"
    local_weights_path = root + "35e4f46485b4d07967e7e9935bc3786aad50687c"
    model_config_path = root + "35e4f46485b4d07967e7e9935bc3786aad50687c" + "/config.json"
    training_args = TrainingArguments(
            num_train_epochs=6,
            remove_unused_columns=False,
            per_device_train_batch_size=3,
            gradient_accumulation_steps=16,
            warmup_steps=16,
            learning_rate=2e-4, # 2e-5
            weight_decay=1e-8, # more aggresive 
            adam_beta2=0.999,
            logging_steps=100,
            optim="adamw_hf",  
            save_strategy="steps",
            save_steps=1000,
            push_to_hub=True,
            save_total_limit=1,
            output_dir="paligemma_vqav2",
            bf16=True,
            report_to=["tensorboard"],
            dataloader_pin_memory=False
        )

    # load the config
    with open(model_config_path, "r") as f:
        model_config_json = json.load(f)

    # model_config = PaliGemmaConfig(**model_config_json)
    # model_config.copy()
    model_config = PaliGemmaConfig()

    # load the dataset
    train_ds = load_vqav2()

    # finetune 
    num_image_tokens = model_config.vision_config.num_image_tokens
    image_size = model_config.vision_config.image_size
    tokenizer = load_tokenizer(local_weights_path)
    # processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)
    processor = PaliGemmaProcessor.from_pretrained("google/paligemma-3b-pt-224")
    image_token = processor.tokenizer.convert_tokens_to_ids("<image>")
    bos_token = processor.tokenizer.convert_tokens_to_ids("<bos>")
    finetune_lora(local_weights_path, model_config, train_ds, vqav2_collate_fn, training_args)



'''
    Removed MLP weights 
    modified the self attention weights 
    new learnable lambda paramter 

    config changes:
    heads = d_model / d, where d = head_d   // 16,
'''