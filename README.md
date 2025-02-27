# GPT_and_openllm-Text2sql-Bird_sql
Project

## Purpose
Text2sql을 수행하기 위한 LLM Fine-Tunig 및 성능 확인
  
  - Role-Playing LLM Agent Debation 기반 종목 분석 리포트 생성 서비스**

## Methodology
Text2sql에 맞는 데이터 셋 및 프롬프트 구성 후 LLM Fine-tuning(분산 학습) 수행

  - Model : GPT 4-o, Llama3.1(8B), DeepseekCoder ver2-lite(16B)
  - Dataset : BIRD-SQL dataset

## Result
GPT-4o(API)

  - DB 중 일부 선정 및 학습 후 학습한 DB(case1)와 학습하지 않은 DB(case2)에 대해 평균 약 4.5% 성능 향상

Llama3.1(8B), DeepseekCoder ver2-lite(16B)

  - 학습 후 전반적인 성능 저하 확인
  - 데이터 셋의 품질 재확인 필요 (BIRD-SQL dataset에 오류 존재)
  - GPT와 다르게 오류에 과적합 되는 것을 확인(모델의 기본 성능이 뒷받침되어야 함)
