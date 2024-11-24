import torch
import torch.nn as nn
import json
from processing_paligemma import PaliGemmaProcessor
from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration
# from transformers import PaliGemmaForConditionalGeneration
from transformers import BitsAndBytesConfig
from transformers import Trainer
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
from pathlib import Path
from safetensors.torch import load_file
from config_utils import PaliGemmaConfig

def initialize_new_layers(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

def setup(local_weights_path, config):
    model = PaliGemmaForConditionalGeneration(config)

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
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type=torch.bfloat16
    )
    lora_config = LoraConfig(
        r=16,  # Rank of the adaptation matrices
        lora_alpha=32,  # Scaling factor
        target_modules=[
            "vision_tower.vision_model.encoder.layers.*.self_attn.q_proj",
            "vision_tower.vision_model.encoder.layers.*.self_attn.k_proj",
            "vision_tower.vision_model.encoder.layers.*.self_attn.v_proj",
            "vision_tower.vision_model.encoder.layers.*.self_attn.out_proj",
            "language_model.model.layers.*.self_attn.q_proj",
            "language_model.model.layers.*.self_attn.k_proj",
            "language_model.model.layers.*.self_attn.v_proj",
            "language_model.model.layers.*.self_attn.o_proj",
        ],  
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"  # Adjust based on your task
    )
    # Inspect module names
    for name, module in model.named_modules():
        print(name)

    model = get_peft_model(model, lora_config)
    for name, param in model.named_parameters():
        if "lora" in name:
            print(f"LoRA parameter: {name}, requires_grad: {param.requires_grad}")
    model.print_trainable_parameters()  
    exit()

    return model

def load_docvqa():
    ds = load_dataset("lmms-lab/DocVQA", "InfographicVQA", split="test")
    print(ds)

def finetune_lora(local_weights_path, model_config, train_ds, collate_fn, training_args):
    model = setup(local_weights_path, model_config)
    print(model)
    exit()
    trainer = Trainer(
        model=model,
        train_dataset=train_ds ,
        data_collator=collate_fn,
        args=training_args
        )

    trainer.train()

if __name__ == "__main__":
    # Specify the local path to the model weights
    local_weights_path = "/home/jerryli/CS228-Project/paligemma-3b-pt-224"
    model_config = "/home/jerryli/CS228-Project/paligemma-3b-pt-224/config.json"
    train_ds =None
    collate_fn = None
    training_args = {}

    with open(model_config, "r") as f:
        model_config = json.load(f)
    model_config = PaliGemmaConfig(model_config)

    finetune_lora(local_weights_path, model_config, train_ds, collate_fn, training_args)

    # load_docvqa()


'''
    Removed MLP weights 
    modified the self attention weights 
    new learnable lambda paramter 

    config changes:
    heads = d_model / d, where d = head_d   // 16,
'''