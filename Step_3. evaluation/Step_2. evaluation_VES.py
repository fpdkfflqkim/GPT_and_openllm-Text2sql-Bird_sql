import multiprocessing as mp
from func_timeout import func_timeout, FunctionTimedOut
import time
import math
import numpy as np
import sqlite3
import argparse
import pandas as pd
import re
from valid_sql import clean_sql_query, extract_sql_query

# DB 연결
def connect_db(db_path):
    return sqlite3.connect(db_path)

# SQL 쿼리 실행
def execute_sql(sql, db_path, return_time=False):
    conn = connect_db(db_path)
    start_time = time.time()
    cursor = conn.cursor()
    cursor.execute(sql)
    res = cursor.fetchall()
    conn.close()
    exec_time = time.time() - start_time
    return exec_time if return_time else res

# 이상치 제거 (VES 계산 과정에서 쿼리 실행 시간의 비율을 정제하는 역할)
def clean_abnormal(input_list):
    input_array = np.asarray(input_list)
    mean = np.mean(input_array)
    std = np.std(input_array)
    return [x for x in input_array if mean - 3 * std < x < mean + 3 * std]

# VES (Velocity Execution Score) 계산
def calculate_ves(predicted_sql, ground_truth_sql, db_path, iterate_num, query_timeout):
    try:
        predicted_res = func_timeout(query_timeout, execute_sql, args=(predicted_sql, db_path))
        ground_truth_res = func_timeout(query_timeout, execute_sql, args=(ground_truth_sql, db_path))
    except FunctionTimedOut:
        return 0  # 쿼리 타임아웃 시 VES 점수를 0으로 설정

    if set(predicted_res) != set(ground_truth_res):
        return 0
    
    diff_list = []
    for _ in range(iterate_num):
        try:
            predicted_time = func_timeout(query_timeout, execute_sql, args=(predicted_sql, db_path, True))
            ground_truth_time = func_timeout(query_timeout, execute_sql, args=(ground_truth_sql, db_path, True))
        except FunctionTimedOut:
            return 0  # 개별 쿼리 실행 타임아웃 시 VES 점수를 0으로 설정

        # 0초인 경우 작은 값을 설정해 나눗셈 에러 방지
        predicted_time = max(predicted_time, 1e-10)
        ground_truth_time = max(ground_truth_time, 1e-10)
        
        diff_list.append(ground_truth_time / predicted_time)
    
    time_ratio = np.mean(clean_abnormal(diff_list))
    
    if time_ratio >= 2:
        return 1.25
    elif time_ratio >= 1:
        return 1
    elif time_ratio >= 0.5:
        return 0.75
    elif time_ratio >= 0.25:
        return 0.5
    else:
        return 0.25

# 쿼리 결과 계산
def process_query(args):
    idx, row, model_output, db_path, iterate_num, meta_time_out = args
    ground_truth_sql = extract_sql_query(row['output'])
    predicted_sql = extract_sql_query(model_output)
    
    try:
        ves_score = func_timeout(
            meta_time_out,
            calculate_ves,
            args=(predicted_sql, ground_truth_sql, db_path, iterate_num, meta_time_out)
        )
    except FunctionTimedOut:
        ves_score = 0
    except Exception as e:
        print(f"Error in query execution for index {idx}: {str(e)}")
        ves_score = 0
    
    return {
        'id': idx,
        'golden_query': ground_truth_sql,
        'generated_query': predicted_sql,
        'ves_score': ves_score
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", type=str, default="./zzinmak/minidev_case1.pkl", help="테스트 데이터셋 경로")
    parser.add_argument("--model_output_csv", type=str, default="./zzinmak/sllm_mini_dev_case1_ver2_output_epoch3.csv", help="모델 출력 CSV 파일 경로")
    parser.add_argument("--num_cpus", type=int, default=25, help="사용할 CPU 코어 수")
    parser.add_argument("--meta_time_out", type=float, default=30.0, help="쿼리 실행 타임아웃 시간(초)")
    parser.add_argument("--iterate_num", type=int, default=100, help="VES 계산을 위한 반복 횟수")
    args = parser.parse_args()

    # 데이터 로드
    df_test = pd.read_pickle(args.test_data_path)
    model_outputs = pd.read_csv(args.model_output_csv,sep='|')['0'].tolist()
    # model_outputs = pd.read_pickle(args.model_output_csv)['result'].tolist()

    # 병렬 처리 설정
    pool = mp.Pool(processes=args.num_cpus)
    results = []

    for idx, ((_, row), model_output) in enumerate(zip(df_test.iterrows(), model_outputs)):
        # db_path = f"C:/Users/user/Desktop/2024-2 didim/train/train_databases/train_databases/{row['db_id']}/{row['db_id']}.sqlite"
        db_path = f"C:/Users/user/Desktop/2024-2 didim/dev_set/dev_20240627/dev_databases/dev_databases/{row['db_id']}/{row['db_id']}.sqlite"
        result = pool.apply_async(process_query, [(idx, row, model_output, db_path, args.iterate_num, args.meta_time_out)])
        results.append(result)

    pool.close()
    pool.join()

    results = [r.get() for r in results]
    results.sort(key=lambda x: x['id'])

    # 전체 결과 계산
    total_queries = len(results)
    ves_total = sum(math.sqrt(r['ves_score']) * 100 for r in results)
    ves_score = ves_total / total_queries

    # 결과 출력
    print(f"총 쿼리 수: {total_queries}")
    print(f"VES (Velocity Execution Score): {ves_score:.4f}")

    # 결과를 CSV 파일로 저장
    results_df = pd.DataFrame(results)
    data_name = args.model_output_csv.split('/')[-1].split('.')[0]
    results_df.to_csv(f'./zzinmak/{data_name}_VES.csv', index=False)
    print(f"상세 결과가 './mini_gpt/{data_name}_VES.csv'에 저장되었습니다.")

if __name__ == "__main__":
    main()
    
# python ves_eval_noDiff.py --test_data_path "./train_case2.pkl" --model_output_csv "./train_case2_results.csv"