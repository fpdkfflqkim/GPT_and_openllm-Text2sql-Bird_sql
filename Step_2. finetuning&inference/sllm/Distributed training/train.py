import os
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, AutoTokenizer
import datasets as ds
from datasets import Dataset
from deepspeed.utils import logger as ds_logger
import pandas as pd
from trl import SFTTrainer

from model import load_model
from configs import get_training_arguments
import huggingface_hub

huggingface_hub.login("")
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

if torch.cuda.is_available():
    device = torch.device("cuda")
    n_gpus = torch.cuda.device_count()
    ds_logger.info(f"device: {device}, {n_gpus}")
else:
    device = torch.device("cpu")
    ds_logger.info(f"device: {device}")

model, tokenizer = load_model()
df = pd.read_csv('./data/dev_train_ver2.csv', sep='|')
input_texts = []
for i in range(len(df)):
    row = df.iloc[i,:]
    instruction = row['instruction']
    input_text = row['input']
    output = row['output']

    messages = [
        {'role': 'user', 'content': f"You are a powerful text-to-SQL model.\n\n{instruction}\n\n{input_text}"},
        {'role': 'assistant', 'content': f"{output}"}
    ]

    input_texts.append(messages)
train_df = pd.DataFrame({'messages':input_texts})
# print(train_df)
train_dataset = Dataset.from_pandas(train_df)

model.train()
trainer = SFTTrainer(
    model=model, 
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    packing = False,
    args=get_training_arguments()
)
trainer.train()
ds_logger.info("Complete Train")
trainer.save_model()
ds_logger.info("Save Model")