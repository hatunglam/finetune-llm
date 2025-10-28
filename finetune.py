import torch    
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)

from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# Model from Huggingface
model_name ="NousResearch/Llama-2-7b-chat-hf"

dataset_name = "mlabonne/guanaco-llama2-1k"

# Define finetuned model
new_model = "llama-2-7b-chat-guanaco"

dataset = load_dataset(dataset_name, split='train')

compute_dtype = getattr(torch, "float16")

# Configure 4-bit quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype, # = float16
    bnb_4bit_use_double_quant=False,
)

# Loading Llama2 model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map={"", 0}
)

model.config.use_cache = False
model.config.pretraining_tp = 1

# Loading tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_size = 'right'

# PEFT parameters
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias='none',
    task_type="CAUSAL_LM",  
)

# Training parameters
training_params = TrainingArguments(
    output_dir= "./result",
    num_train_epochs=1,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    optim="page_adamw_32bit",
    save_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to='tensorboard',
)


# Self supervised model tuning
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field='text',
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

# Train model
if __name__ == "__main__":
    trainer.train()

    trainer.model.save_pretrained(new_model)
    trainer.tokenizer.save_pretrained(new_model)

    # Evaluate with tensorboard

    # from tensorboard import notebook
    # log_dir = "results/runs"
    # notebook.start("--logdir {} --port 4000".format(log_dir))