import sys
import json
import numpy as np
import argparse
import multiprocessing as mp
import time
import pandas as pd
import sqlite3
import re
from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut
from valid_sql import clean_sql_query, extract_sql_query


def clean_sql_query(query):
    # 앞뒤 공백 제거
    query = query.strip()
    # 마지막 세미콜론 제거
    query = re.sub(r';$', '', query)
    # 여러 줄의 공백을 단일 공백으로 대체
    query = re.sub(r'\s+', ' ', query)
    return query

# DB 연결
def connect_db(db_path):
    return sqlite3.connect(db_path)

# SQL 쿼리 실행
def execute_sql(sql, db_path):
    conn = connect_db(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        res = cursor.fetchall()
    except sqlite3.Error as e:
        res = f"Error: {str(e)}"
    finally:
        cursor.close()
    return res

# EX (Execution Match) 계산
def calculate_ex(predicted_res, ground_truth_res):
    return 1 if set(predicted_res) == set(ground_truth_res) else 0

# 쿼리 결과 계산
def process_query(args, timeout=10):
    idx, row, model_output, db_path = args
    ground_truth_sql = extract_sql_query(row['output'])
    predicted_sql = extract_sql_query(model_output)
    start_time = time.time()

    try:
        # 타임아웃을 사용하여 SQL 실행
        predicted_res = func_timeout(timeout, execute_sql, args=(predicted_sql, db_path))
        ground_truth_res = func_timeout(timeout, execute_sql, args=(ground_truth_sql, db_path))
        ex_match = calculate_ex(predicted_res, ground_truth_res)
        print(f"Index {idx} - EX Match: {ex_match}")
    except FunctionTimedOut:
        print(f"Index {idx} - Query execution timed out.")
        ex_match = 0
    except Exception as e:
        print(f"Error in query execution for index {idx}: {str(e)}")
        ex_match = 0

    end_time = time.time()
    execution_time = end_time - start_time

    return {
        'id': idx,
        'golden_query': ground_truth_sql,
        'generated_query': predicted_sql,
        'ex_match': ex_match,
        'execution_time': execution_time
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", type=str, default="./mini_gpt/minidev_case1.pkl", help="테스트 데이터셋 경로")
    parser.add_argument("--model_output_csv", type=str, default="./zzinmak/sllm_mini_dev_case1_ver2_output_epoch3.csv", help="모델 출력 CSV 파일 경로")
    parser.add_argument("--num_cpus", type=int, default=25, help="사용할 CPU 코어 수")
    parser.add_argument("--timeout", type=int, default=30, help="쿼리 실행 타임아웃 시간(초)")
    args = parser.parse_args()

    # 데이터 로드
    df_test = pd.read_pickle(args.test_data_path)
    # ============================== column 이름 확인!!!! =================================
    model_outputs = pd.read_csv(args.model_output_csv,sep='|')['0'].tolist()
    # model_outputs = pd.read_pickle(args.model_output_csv)['result'].tolist()

    # 병렬 처리 설정
    pool = mp.Pool(processes=args.num_cpus)

    # 작업 리스트 생성
    tasks = [
    # ================================ DB path 확인!!!! ===================================
        # (idx, row, model_output, f"C:/Users/user/Desktop/2024-2 didim/train/train_databases/train_databases/{row['db_id']}/{row['db_id']}.sqlite")
        (idx, row, model_output, f"C:/Users/user/Desktop/2024-2 didim/dev_set/dev_20240627/dev_databases/dev_databases/{row['db_id']}/{row['db_id']}.sqlite") 
        for idx, ((_, row), model_output) in enumerate(zip(df_test.iterrows(), model_outputs))
    ]

    # 모든 작업을 병렬로 실행하고 결과를 수집
    results = []
    for task in tasks:
        try:
            result = pool.apply_async(process_query, args=(task, args.timeout))
            results.append(result)
        except Exception as e:
            print(f"Error in processing task {task[0]}: {str(e)}")

    pool.close()
    pool.join()

    # 결과 수집
    completed_results = []
    for result in results:
        try:
            completed_results.append(result.get())
        except Exception as e:
            print(f"Error collecting result: {str(e)}")

    # 결과 정렬 및 저장
    completed_results.sort(key=lambda x: x['id'])
    results_df = pd.DataFrame(completed_results)
    data_name = args.model_output_csv.split('/')[-1].split('.')[0]
    results_df.to_csv(f'./zzinmak/{data_name}_EX.csv', index=False)
    print(f"상세 결과가 './mini_gpt/{data_name}_EX.csv'에 저장되었습니다.")

if __name__ == "__main__":
    main()

# python ex_eval_noDiff.py --test_data_path "./dev_ver2_case1.pkl" --model_output_csv "./sllm_base_dev_ver2_case1_output.csv"