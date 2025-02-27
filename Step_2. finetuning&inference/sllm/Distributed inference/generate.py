from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForCausalLM, AutoConfig, AutoModel
from accelerate import init_empty_weights, load_checkpoint_and_dispatch,infer_auto_device_map
from accelerate.utils.modeling import compute_module_sizes
import torch
from tqdm import tqdm 
import pandas as pd
import datasets as ds
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

MODEL_ID = "/home/work/train/deepspeed_trainer/Llama-3.1-8B-Instruct" # 로컬에 저장한 모델 경로
TOKENIZER_ID = "/home/work/train/deepspeed_trainer/Llama-3.1-8B-Instruct" # 로컬에 저장한 토크나이저 경로
EVAL_DATA = "/home/work/inferrence/data/minidev_case2.pkl" # 추론 데이터
SAVE_PATH = "./output/base_minidev_case2_ver2_output.csv" # 추론값 저장할 경로

device = torch.device(
    "cuda:0" if torch.cuda.is_available() 
    else "cpu"
    )
print(f"device : {device}")
torch.cuda.memory_allocated()==512

dataset = pd.read_pickle(EVAL_DATA)

# 모델과 토크나이저 로드 및 GPU 분산
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    add_bos_token=True,
    add_eos_token=True,
    local_files_only=True,
    pad_token='<pad>'
)

# 빈 가중치로 모델을 초기화한 후 체크포인트 로드
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, local_files_only=True, 
    device_map="auto", max_memory={0: "15GiB", 1: "15GiB", 2: "15GiB", "cpu": "30GiB"})

# 모델을 여러 GPU로 로드
model = load_checkpoint_and_dispatch(
    model,
    checkpoint=MODEL_ID,
    offload_folder=None, 
    dtype='float16', 
)

torch.cuda.empty_cache()

# 데이터셋에 대해 예측 수행
results = []

for _, item in dataset.iterrows():
    instruction = item['instruction']
    input_text = item['input']

    messages = [
    {'role': 'user', 'content': f"You are a powerful text-to-SQL model.\n\n{instruction}\n\n{input_text}"}
    ]

    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    attention_mask = inputs.ne(tokenizer.pad_token_id)

    # 모델에서 텍스트 생성 수행
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            do_sample=False,
            # top_k=50,
            # top_p=0.95,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    generated_text = generated_text.encode('utf-8', errors='ignore').decode('utf-8')
    results.append({generated_text})
    print(generated_text)
    
# 결과를 DataFrame으로 변환
generated_text_df = pd.DataFrame(results)
results_df = pd.concat([generated_text_df, dataset], axis=1)
clean_results_df = results_df.drop(['instruction', 'input'], axis=1)

# CSV 파일로 저장
clean_results_df.to_csv(SAVE_PATH, sep='|', index=False) # 컴마가 많은 데이터 특성상 sep='|' 필수
print(f"Results saved to {SAVE_PATH}")
