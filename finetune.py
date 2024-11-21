import torch
from transformers import PaliGemmaForConditionalGeneration
from transformers import BitsAndBytesConfig
from transformers import Trainer
from peft import get_peft_model, LoraConfig
from datasets import load_dataset

def setup(local_model_path):
    model = PaliGemmaForConditionalGeneration.from_pretrained(local_model_path)
    print(model)
    exit()
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type=torch.bfloat16
    )
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    model = PaliGemmaForConditionalGeneration.from_pretrained(local_model_path, quantization_config=bnb_config, device_map={"":0})
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  

    return model

def load_docvqa():
    ds = load_dataset("lmms-lab/DocVQA", "InfographicVQA", split="test")
    print(ds)

def finetune_lora(local_model_path, train_ds, collate_fn, training_args):
    model = setup(local_model_path)
    trainer = Trainer(
        model=model,
        train_dataset=train_ds ,
        data_collator=collate_fn,
        args=training_args
        )

    trainer.train()

if __name__ == "__main__":
    # Specify the local path to the model weights
    local_model_path = "/home/jerryli/CS228-Project/paligemma-3b-pt-224"
    train_ds =None
    collate_fn = None
    training_args = {}

    finetune_lora(local_model_path, train_ds, collate_fn, training_args)

    # load_docvqa()


'''
    Removed MLP weights 
    modified the self attention weights 
    new learnable lambda paramter 
'''