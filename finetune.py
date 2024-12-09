import torch
import torch.nn as nn
import json
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    GenerationConfig,
    AutoTokenizer,
)
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
from pathlib import Path
from safetensors.torch import load_file
import os
from datetime import datetime
from huggingface_hub import login
from transformers import TrainerCallback
from peft import PeftModel
import numpy as np 
import random

device = "cuda"

os.environ["HF_DATASETS_OFFLINE"] = "1"

# Call back for saving the parameters
class SaveLoRACallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        # Save new parameters every 10 steps
        if state.global_step % 500 == 0:
            save_dir = os.path.join(args.output_dir, f"new_params_step_{state.global_step}")
            os.makedirs(save_dir, exist_ok=True)

            # Filter for new parameters
            new_params = {
                name: param
                for name, param in model.state_dict().items()
                if any(keyword in name for keyword in ["lambda_q1", "lambda_k1", "lambda_q2", "lambda_k2", "subln"])
            }

            # Save only new parameters
            torch.save(new_params, os.path.join(save_dir, "diff_attention_params.pth"))
            print(f"Saved new parameters at step {state.global_step} to {save_dir}")

def initialize_new_layers(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

def setup(local_weights_path, config):
    # Initialize model without BitsAndBytesConfig for now
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        local_weights_path,
        local_files_only=True,
        ignore_mismatched_sizes=True,
        device_map="auto",
    )

    with torch.no_grad():
        for name, param in model.named_parameters():
            if ('lambda_q1' in name or
            'lambda_k1' in name or
            'lambda_q2' in name or
            'lambda_k2' in name or
            'subln' in name):
                print(f"Setting custom values for {name}")
                param.data = nn.Parameter(torch.empty(512 // 2, dtype=torch.bfloat16, device="cuda").normal_(mean=0.0, std=0.1))  # Example: Set random values
                print(param.data)
        # SwigLU initialization logic
            if 'swiglu' in name:
                print(f"Initializing SwigLU weights for {name}")
                if hasattr(param, "weight"):
                    nn.init.xavier_uniform_(param.weight)  # Apply Xavier initialization
                    print(f"Initialized weight for {name}")
                if hasattr(param, "bias") and param.bias is not None:
                    nn.init.zeros_(param.bias)  # Zero-initialize the bias
                    print(f"Initialized bias for {name}")

    # Load huggingface weights 
    state_dict = {}
    weights_path = Path(local_weights_path)
    safetensors_files = list(weights_path.glob("*.safetensors"))
    for file in safetensors_files:
        state_dict.update(load_file(file))

    # Filter out missing keys
    language_model_prefix = "language_model.model"
    vision_encoder_prefix = "vision_tower.vision_model"

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
    # initialize_new_layers(model)

    # Freeze specific parts
    for param in model.vision_tower.parameters():
        param.requires_grad = False 
    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if (
            'lambda_q1' in name or
            'lambda_k1' in name or
            'lambda_q2' in name or
            'lambda_k2' in name or
            'subln' in name or 'swiglu' in name
        ):  
            param.requires_grad = True

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
        r=32,  # Rank of the adaptation matrices
        lora_alpha=64,  # Scaling factor
        target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ], 
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"  # Adjust based on your task
    )

    # Print parameters 
    for name, param in model.named_parameters():
        print(f"{name} requires_grad: {param.requires_grad}")

    model = get_peft_model(model, lora_config)
    for name, param in model.named_parameters():
        if "lora" in name:
            print(f"LoRA parameter: {name}, requires_grad: {param.requires_grad}")
    model.print_trainable_parameters()  


    return model

def load_docvqa():
    ds = load_dataset("lmms-lab/DocVQA", "InfographicVQA", split="test")
    print(ds)

def load_vqav2():
    dataset = load_dataset('HuggingFaceM4/VQAv2', split="train[:100%]")
    dataset = dataset.remove_columns(["question_type", "answers", "answer_type", "image_id", "question_id"])
    split_ds = dataset.train_test_split(test_size=0.15)
    return split_ds["train"], split_ds["test"]

def vqav2_collate_fn(batch): # dictionary of images and text
    texts = ["answer " + sequence["question"] for sequence in batch]
    labels= [sequence['multiple_choice_answer'] for sequence in batch]
    images = [sequence["image"].convert("RGB") for sequence in batch]
    tokens = processor(text=texts, images=images, suffix=labels,
                        return_tensors="pt", padding="longest",
                        tokenize_newline_separately=False)

    # Ensure tensors are moved to the correct device without changing dtype unnecessarily
    tokens = {k: v.to(device) for k, v in tokens.items()}
    return tokens

def load_tokenizer(model_path):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"
    return tokenizer

def load_rng_state(checkpoint_path):
    rng_state_path = os.path.join(checkpoint_path, "rng_state.pth")
    if os.path.isfile(rng_state_path):
        rng_state = torch.load(rng_state_path, map_location="cpu")
        # torch.set_rng_state(rng_state['torch'])
        np.random.set_state(rng_state['numpy'])
        random.setstate(rng_state['python'])
        if 'cuda' in rng_state and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng_state['cuda'])
        print(f"RNG state loaded from {rng_state_path}")
    else:
        print(f"No RNG state found at {rng_state_path}")

def resume_finetuning(base_model_path, adapter_dir, new_weights_path, train_ds, eval_ds, processor, training_args, resume_checkpoint_path=None):
    # 1. Load the base model
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        base_model_path,
        local_files_only=True,
        ignore_mismatched_sizes=True,
        device_map="auto"
    )

    # 2. Load the adapter (LoRA) weights
    model = PeftModel.from_pretrained(base_model, adapter_dir, device_map="auto")

    with torch.no_grad():
        for name, param in model.named_parameters():
            if ('lambda_q1' in name or
            'lambda_k1' in name or
            'lambda_q2' in name or
            'lambda_k2'):
                # hacky solution to set the lambda parameters since resuming will without is random initally
                print(f"Setting custom values for {name}")
                param.data = nn.Parameter(torch.empty(512 // 2, dtype=torch.bfloat16, device="cuda").normal_(mean=0.0, std=0.1))  # Example: Set random values

    # 3. Load the new lambda_q, lambda_k, and subln weights
    if os.path.isfile(new_weights_path):
        # Load the new parameters
        new_params = torch.load(new_weights_path, map_location="cpu")

        missing_keys, unexpected_keys = model.load_state_dict(new_params, strict=False)

        # Free memory
        del new_params
        torch.cuda.empty_cache()
    else:
        print("No new weights file found at:", new_weights_path)

    # Make sure the required parameters are trainable
    for name, param in model.named_parameters():
        if any(x in name for x in ['lambda_q1', 'lambda_k1', 'lambda_q2', 'lambda_k2', 'subln', 'gate_proj', 'lora']):
            param.requires_grad = True
        else:
            param.requires_grad = False

    torch.cuda.empty_cache()

    def resume_collate_fn(batch):
        texts = ["answer " + sequence["question"] for sequence in batch]
        labels = [sequence['multiple_choice_answer'] for sequence in batch]
        images = [sequence["image"].convert("RGB") for sequence in batch]
        tokens = processor(
            text=texts,
            images=images,
            suffix=labels,
            return_tensors="pt",
            padding="longest",
            tokenize_newline_separately=False
        )
        tokens = {k: v.to(model.device) for k, v in tokens.items()}
        return tokens
    
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=resume_collate_fn,
        args=training_args
    )
    trainer.train()

    # 5. Save the re-finetuned model
    save_dir = os.path.join(training_args.output_dir, "resumed_finetuned")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    print("Resumed finetuned model saved to:", save_dir)

def finetune_lora(local_weights_path, model_config, train_ds, eval_ds, collate_fn, training_args):
    # Initialize the model and freeze/unfreeze weights
    model = setup(local_weights_path, model_config)

    # Create timestamped folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_folder = os.path.join("paligemma_vqav2", timestamp)
    os.makedirs(base_folder, exist_ok=True)

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds, 
        data_collator=collate_fn,
        callbacks=[SaveLoRACallback()],
        args=training_args,
        # compute_metrics=compute_metrics  
    )
    
    # Train
    print("Begin Finetuning")
    trainer.train()

    # Save weights
    weights_folder = os.path.join(base_folder, "finetuned_weights")
    os.makedirs(weights_folder, exist_ok=True)
    model.save_pretrained(weights_folder)
    print(f"Fine-tuned model saved to: {weights_folder}")

    # Save the model config
    config_path = os.path.join(base_folder, "model_config.json")
    with open(config_path, "w") as config_file:
        json.dump(model_config, config_file, indent=4)  # No need for .to_dict()
    print(f"Model configuration saved to: {config_path}")

    # Save the training arguments
    train_args_path = os.path.join(base_folder, "training_args.json")
    training_args_dict = training_args.to_dict()
    with open(train_args_path, "w") as train_args_file:
        json.dump(training_args_dict, train_args_file, indent=4)
    print(f"Training arguments saved to: {train_args_path}")

    # Save LoRA config
    lora_config = {
        "rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "target_modules": [
            "q_proj", "o_proj", "k_proj", "v_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }
    lora_config_path = os.path.join(base_folder, "lora_config.json")
    with open(lora_config_path, "w") as lora_file:
        json.dump(lora_config, lora_file, indent=4)
    print(f"LoRA configuration saved to: {lora_config_path}")

    print(f"All outputs saved to folder: {base_folder}")


if __name__ == "__main__":
    # Specify the local path to the model weights
    root = "/home/jerryli/CS228-Project/paligemma-3b-pt-224/"
    local_weights_path = root
    model_config_path = os.path.join(root, "config.json")
    
    training_args = TrainingArguments(
        num_train_epochs=1,
        remove_unused_columns=False,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        learning_rate=4e-4,
        weight_decay=1e-9,
        adam_beta2=0.98,
        logging_steps=100,  # Adjusted for demonstration
        evaluation_strategy="steps", 
        eval_steps=1000, 
        optim="adamw_hf",  
        save_strategy="steps",
        save_steps=500,  # Adjusted for demonstration
        push_to_hub=False,
        # save_total_limit=1,
        output_dir="paligemma_vqav2",  # Base output directory
        bf16=True,  # Ensure hardware supports bfloat16
        # fp16=False,
        report_to=["tensorboard"],
        dataloader_pin_memory=False
    )

    # Load the config
    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    # Load the dataset
    train_ds, eval_ds = load_vqav2()

    # Load tokenizer and processor
    tokenizer = load_tokenizer(local_weights_path)
    processor = PaliGemmaProcessor.from_pretrained(root)

    # Call finetune_lora
    finetune_lora(local_weights_path, model_config, train_ds, eval_ds, vqav2_collate_fn, training_args)
    
    # base_model_path = "/home/jerryli/CS228-Project/paligemma-3b-pt-224" # "google/paligemma-3b-pt-224"  # actual model name
    # adapter_path = "/home/jerryli/CS228-Project/paligemma_vqav2/2024-12-06_12-00-27/checkpoints/checkpoint-12000"  
    # new_weights_path = "/home/jerryli/CS228-Project/paligemma_vqav2/2024-12-06_12-00-27/new_params/new_params_step_12000/diff_attention_params.pth"
    # resume_finetuning(
    #     base_model_path=base_model_path,
    #     adapter_dir=adapter_path,
    #     new_weights_path=new_weights_path,
    #     train_ds=train_ds,
    #     eval_ds=eval_ds,
    #     processor=processor,
    #     training_args=training_args,
    #     resume_checkpoint_path=adapter_path
    # )
    

