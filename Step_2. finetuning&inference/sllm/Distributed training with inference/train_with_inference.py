import os  # OS 관련 기능을 사용하기 위한 모듈
import torch  # PyTorch 라이브러리
from torch.utils.data import Dataset, DataLoader  # 데이터셋과 데이터 로더를 위한 모듈
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding  # Hugging Face의 모델과 토크나이저
import argparse  # 커맨드라인 인자를 파싱하기 위한 모듈
import pandas as pd  # 데이터 프레임 처리를 위한 pandas 라이브러리
import deepspeed  # DeepSpeed 라이브러리
import matplotlib.pyplot as plt  # 그래프 작성을 위한 matplotlib
import numpy as np  # 수치 연산을 위한 numpy
from deepspeed.ops.adam import DeepSpeedCPUAdam  # DeepSpeed의 Adam 옵티마이저
import json  # JSON 파일 처리를 위한 모듈
import time  # 시간 관련 기능을 위한 모듈
from datetime import timedelta  # 시간 간격을 나타내기 위한 모듈
import torch.distributed as dist  # 분산 훈련을 위한 모듈
import huggingface_hub

huggingface_hub.login("")
# 분산 훈련을 위한 프로세스 그룹 초기화 (NCCL 백엔드 사용)
dist.init_process_group(backend='nccl')

# 커맨드라인 인자 파싱을 위한 ArgumentParser 객체 생성
parser = argparse.ArgumentParser()
parser = deepspeed.add_config_arguments(parser)  # DeepSpeed의 설정 인자 추가
parser.add_argument("--local_rank", type=int, default=-1)  # 로컬 랭크 인자
parser.add_argument("--batch_size", type=int, default=1)  # 배치 크기 인자
parser.add_argument("--num_epochs", type=int, default=20)  # 에폭 수 인자
args = parser.parse_args()  # 인자 파싱

# DeepSpeed 설정 파일 로드
with open('./ds_config_ms.json', 'r') as f:
    ds_config = json.load(f)

# 사용자 정의 데이터셋 클래스 정의
class SQLQueryDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=8192):
        self.dataframe = dataframe  # 데이터프레임 저장
        self.tokenizer = tokenizer  # 토크나이저 저장
        self.max_length = max_length  # 최대 길이 설정

    def __len__(self):
        return len(self.dataframe)  # 데이터셋의 길이 반환

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]  # 데이터프레임에서 인덱스에 해당하는 행 가져오기
        instruction = row['instruction']  # 지시문 가져오기
        input_text = row['input']  # 입력 텍스트 가져오기
        output = row['output']  # 출력 텍스트 가져오기

        # 메시지 포맷팅
        messages = [
            {'role': 'user', 'content': f"You are a powerful text-to-SQL model.\n\n{instruction}\n\n{input_text}"},
            {'role': 'assistant', 'content': output}
        ]

        # 토크나이저를 사용해 입력 텐서 생성
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = inputs.squeeze(0)  # 배치 차원 제거

        # 어텐션 마스크 생성
        # attention_mask = (inputs != self.tokenizer.pad_token_id).long()
        attention_mask = (inputs != self.tokenizer.pad_token_id)

        # SQL 쿼리를 레이블로 사용
        labels = inputs.clone()

        return {
            'input_ids': inputs,  # 입력 ID
            'attention_mask': attention_mask,  # 어텐션 마스크
            'labels': labels  # 레이블
        }

def main():
    deepspeed.init_distributed()  # DeepSpeed 분산 초기화
    
    # local_rank에 맞는 CUDA 디바이스 설정
    torch.cuda.set_device(args.local_rank)  # CUDA 디바이스 설정
    device = torch.device(f"cuda:{args.local_rank}")  # 디바이스 객체 생성

    # 모델과 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained("./Llama-3.1-8B-Instruct", local_files_only=True, add_bos_token=True, add_eos_token=True, pad_token='<pad>')
    model = AutoModelForCausalLM.from_pretrained("./Llama-3.1-8B-Instruct", local_files_only=True, 
    ignore_mismatched_sizes=True, torch_dtype=torch.bfloat16, attn_implementation = "flash_attention_2").to(device)

    model.gradient_checkpointing_enable()  # 그래디언트 체크포인트 활성화

    # 학습 데이터셋 로드
    df = pd.read_csv("./data/dev_train_ver2_last.csv",sep='|')  # 피클 파일에서 데이터프레임 로드
    train_dataset = SQLQueryDataset(df, tokenizer)  # 사용자 정의 데이터셋 생성

    # DeepSpeed 엔진 초기화
    model_engine, _, _, _ = deepspeed.initialize(
        config=ds_config,  # DeepSpeed 설정
        args=args,  # 커맨드라인 인자
        model=model,  # 모델
        model_parameters=model.parameters(),  # 모델 파라미터
        training_data=train_dataset  # 학습 데이터
    )
    
    # 데이터 로더 설정
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)  # 패딩을 위한 데이터 콜레이터
    train_loader = DataLoader(
        train_dataset,  # 학습 데이터셋
        batch_size=args.batch_size,  # 배치 크기
        collate_fn=data_collator  # 데이터 콜레이터
    )

    losses = []  # 손실 값을 저장할 리스트
    start_time = time.time()  # 시작 시간 기록
    total_steps = 0  # 총 스텝 수 초기화
    accumulation_steps = 8  # 손실 누적 스텝 수

    # 추론 데이터 가져오기
    test_data = pd.read_csv('/home/work/inferrence/data/dev_test_ver2_last.csv',sep='|')
    unseen_data = pd.read_csv('/home/work/inferrence/data/dev_unseen_ver2_last.csv',sep='|')
    infer_data = [test_data, unseen_data]
    infer_data_name = ['sllm_dev_case1_ver2_output','sllm_dev_case2_ver2_output']

    # 에폭 루프
    for epoch in range(args.num_epochs):
        model_engine.train()  # 모델을 학습 모드로 설정
        epoch_losses = []  # 에폭 손실 저장 리스트
        for i, batch in enumerate(train_loader):  # 데이터 로더에서 배치 반복
            
            total_steps += 1  # 총 스텝 수 증가
            inputs = batch['input_ids'].to(model_engine.device)  # 입력 ID 텐서
            attention_mask = batch['attention_mask'].to(model_engine.device)  # 어텐션 마스크 텐서
            labels = batch['labels'].to(model_engine.device)  # 레이블 텐서

            # 모델 출력 및 손실 계산
            outputs = model_engine(inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / accumulation_steps  # 손실을 누적 단계 수로 나눔
            
            epoch_losses.append(loss.item() * accumulation_steps)  # 에폭 손실 기록
            model_engine.backward(loss)  # 그래디언트 계산

            # 일정 스텝마다 로그 출력
            if total_steps % 5 == 0 and args.local_rank == 0:
                elapsed_time = time.time() - start_time  # 경과 시간 계산
                avg_time_per_step = elapsed_time / total_steps  # 평균 스텝 시간 계산
                estimated_time_remaining = avg_time_per_step * (len(train_loader) * args.num_epochs - total_steps)  # 남은 시간 추정
                
                print(f"Epoch: {epoch}, Step: {total_steps}, "
                      f"Loss: {loss.item() * accumulation_steps:.4f}, "
                      f"Elapsed Time: {timedelta(seconds=int(elapsed_time))}, "
                      f"Estimated Time Remaining: {timedelta(seconds=int(estimated_time_remaining))}")
                    
            # 누적 단계가 끝나면 스텝 진행
            if (i + 1) % accumulation_steps == 0:
                model_engine.step()  # 모델 업데이트

        # 에폭의 평균 손실 계산 및 기록   
        avg_loss = np.mean(epoch_losses)  # 평균 손실 계산
        losses.append(avg_loss)  # 평균 손실 기록

        print(f"Epoch: {epoch}, Loss: {loss.item()}")  # 에폭 손실 출력

        # 추론
        if epoch==0 or epoch==9 or epoch==14 or epoch==19:
            model_engine.save_checkpoint("./checkpoint", f"dev_checkpoint_deepspeed_epoch_{epoch+1}")  # 체크포인트 저장
            model_engine.eval()
            with torch.no_grad():
                for data, name in zip(infer_data, infer_data_name):
                    results = []
                    for _, row in data.iterrows():
                        instruction = row['instruction']
                        input_text = row['input']

                        messages = [
                        {'role': 'user', 'content': f"You are a powerful text-to-SQL model.\n\n{instruction}\n\n{input_text}"}
                        ]

                        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model_engine.device)

                        # attention_mask = inputs.ne(tokenizer.pad_token_id).long()
                        attention_mask = inputs.ne(tokenizer.pad_token_id)

                        # 모델 추론
                        outputs = model.generate(
                            inputs, 
                            attention_mask=attention_mask,
                            max_new_tokens=1024, 
                            do_sample=False, 
                            # top_k=50, 
                            # top_p=0.9, 
                            num_return_sequences=1,
                            eos_token_id=tokenizer.eos_token_id
                        )

                        generated_text = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
                        generated_text = generated_text.encode('utf-8', errors='ignore').decode('utf-8')
                        results.append({generated_text})

                    # 결과를 DataFrame으로 변환
                    generated_text_df = pd.DataFrame(results)
                    results_df = pd.concat([generated_text_df, data], axis=1)
                    clean_results_df = results_df.drop(['instruction', 'input'], axis=1)
                    
                    # CSV 파일로 저장
                    clean_results_df.to_csv(f'/home/work/inferrence/output/{name}_epoch{epoch+1}.csv', sep='|', index=False)
                    print(f"Results saved to ./output/{name}_epoch{epoch+1}.csv")
    
    # 손실 그래프 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.num_epochs + 1), losses)  # 에폭에 따른 손실 그래프
    plt.title('Training Loss over Epochs')  # 그래프 제목
    plt.xlabel('Epoch')  # x축 레이블
    plt.ylabel('Average Loss')  # y축 레이블
    plt.savefig(f'training_loss_epoch5.png')  # 그래프 저장
    plt.close()  # 그래프 객체 닫기

    # 손실 값을 CSV 파일로 저장
    np.savetxt(f'training_losses_epoch{epoch+1}.csv', losses, delimiter=',')

if __name__ == "__main__":
    main()  # 메인
