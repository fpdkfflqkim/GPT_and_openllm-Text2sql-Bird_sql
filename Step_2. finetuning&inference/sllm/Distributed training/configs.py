from transformers import AutoConfig, TrainingArguments, AutoTokenizer
import os
from deepspeed.utils import logger as ds_logger

BASE_MODEL_ID = "./Llama-3.1-8B-Instruct"
TOKENIZER_ID = "./Llama-3.1-8B-Instruct"
CHUNK_SIZE = 8196

def get_model_config():
    """Wrapper for AutoConfig.from_pretrained(). 
    
        return (Config, unused_kwargs)"""
    ds_logger.info("Setting Configuration...")
    
    tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_ID,
    add_bos_token=True,
    add_eos_token=True,
    local_files_only=True
    )
    
    TOKEN_SIZE = tokenizer.vocab_size
    
    config  = AutoConfig.from_pretrained(
        BASE_MODEL_ID,
        return_unused_kwargs=True,
        use_cache=False,
        max_position_embeddings=CHUNK_SIZE,
        vocab_size = TOKEN_SIZE
        )
    return config    


def get_training_arguments():
    """Wrapper for TrainingArguments"""
    
    return TrainingArguments(
        output_dir="./results2",
        deepspeed = "ds_config_2.json",
        gradient_checkpointing=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        weight_decay=0.1, 
        max_grad_norm = 1.0,
        num_train_epochs=10,
        eval_steps=10,
        # warmup_steps=0,
        # learning_rate=0.0005,
        # lr_scheduler_type="cosine",

        logging_dir='./results2/logs',
        logging_steps=20,
        report_to="none", # disable wandb if logged in
        save_strategy="epoch", # deepspeed zero3는 체크포인트 오류 난다.
        overwrite_output_dir=True,  
    )