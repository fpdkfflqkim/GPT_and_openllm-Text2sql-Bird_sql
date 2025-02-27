### model.py

import torch
from transformers import AutoConfig, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM, AutoModel
import os
from deepspeed.utils import logger as ds_logger
import deepspeed

from configs import get_model_config, BASE_MODEL_ID, TOKENIZER_ID

def load_model():
    """return (model, tokenizer)"""
    
    ds_logger.info("Setting Configuration...")
    
    config = get_model_config()

    ds_logger.info("Model Configuration Complete !\n")
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # avoid warning: avoid using tokenizer before the fork
    ds_logger.info("Load Pretrained Tokenizer...")
    
    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID,
        add_bos_token=True,
        add_eos_token=True,
        local_files_only=True,
        pad_token='<pad>'
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})

    ds_logger.info("Loaded Pretrained Tokenizer!\n")
    
    # 모델
    with deepspeed.zero.Init():
        ds_logger.info("Load Model based on config...")
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID,
                                                    config=config,
                                                    ignore_mismatched_sizes=True,
                                                    local_files_only=True 
                                                     )
        ds_logger.info("Loaded Initialized Model based on config!\n")
        
    return model, tokenizer