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


