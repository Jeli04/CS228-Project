import os
from datasets import load_dataset, load_from_disk
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig, TrainingArguments, Trainer
import torch
from peft import get_peft_model, LoraConfig
from datetime import datetime

# 1. Load Dataset
ds = load_dataset('HuggingFaceM4/VQAv2', split="train[:50%]")
cols_remove = ["question_type", "answers", "answer_type", "image_id", "question_id"]
ds = ds.remove_columns(cols_remove)
split_ds = ds.train_test_split(test_size=0.20)
train_ds = split_ds["test"]
print(train_ds[0])

# 2. Load Model
model_id = "google/paligemma-3b-pt-224"
processor = PaliGemmaProcessor.from_pretrained(model_id)
device = "cuda"
image_token = processor.tokenizer.convert_tokens_to_ids("<image>")
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

for param in model.vision_tower.parameters():
    param.requires_grad = False 
for param in model.multi_modal_projector.parameters():
    param.requires_grad = False

# Loading Quantised Model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_type=torch.bfloat16
)
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id,
                                                          quantization_config=bnb_config,
                                                          device_map={"": 0})
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3. Fine-tuning the model
def collate_fn(examples):
    texts = ["answer " + example["question"] for example in examples]
    labels = [example['multiple_choice_answer'] for example in examples]
    images = [example["image"].convert("RGB") for example in examples]
    tokens = processor(text=texts, images=images, suffix=labels,
                       return_tensors="pt", padding="longest",
                       tokenize_newline_separately=False)
    tokens = tokens.to(torch.bfloat16).to(device)

    return tokens

args = TrainingArguments(
    num_train_epochs=2,
    remove_unused_columns=False,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=2,
    learning_rate=2e-5,
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=100,
    optim="adamw_hf",
    save_strategy="steps",
    save_steps=1000,
    push_to_hub=True,
    save_total_limit=1,
    output_dir="paligemma_vqav2",
    bf16=True,
    dataloader_pin_memory=False
)

trainer = Trainer(
    model=model,
    train_dataset=train_ds,
    data_collator=collate_fn,
    args=args
)

trainer.train()

# save weights
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
base_folder = os.path.dirname("/home/jerryli/CS228-Project/paligemma_vqav2/finetuned_weights")  # Parent folder of local weights path
save_folder = os.path.join(base_folder, f"finetuned_paligemma_original_{timestamp}")
os.makedirs(save_folder, exist_ok=True)

model.save_pretrained(save_folder)
print(f"Fine-tuned model saved to: {save_folder}")

