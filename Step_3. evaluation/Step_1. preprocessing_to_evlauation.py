import re

# 텍스트에서 SQL 쿼리 추출
def extract_sql_query(text):
    # 'SQL:' 및 'assistant' 접두사 제거
    text = re.sub(r'^(SQL|assistant)\s*:', '', text.strip(), flags=re.IGNORECASE)
    text = re.sub(r'^(assistant)\s+', '', text.strip(), flags=re.IGNORECASE)

    # ```sqlite ```로 이루어진 SQL 코드 블록 추출
    sqlite_pattern = r'```sqlite\s*([\s\S]*?)\s*```'
    matches = re.findall(sqlite_pattern, text, re.IGNORECASE)
    if matches:
        return clean_sql_query(matches[-1])  # 마지막 SQL 코드 블록 반환

    # ```sql ```로 이루어진 SQL 코드 블록 추출
    sql_pattern = r'```sql\s*([\s\S]*?)\s*```'
    matches = re.findall(sql_pattern, text, re.IGNORECASE)
    if matches:
        return clean_sql_query(matches[-1])  # 마지막 SQL 코드 블록 반환
    
    # ``` ```로 이루어진 일반 코드 블록 추출
    code_pattern = r'```\s*([\s\S]*?)\s*```'
    matches = re.findall(code_pattern, text)
    if matches:
        return clean_sql_query(matches[-1])  # 마지막 일반 코드 블록 반환
    
    # 코드 블록이 없으면 전체 텍스트를 처리
    return clean_sql_query(text)

def clean_sql_query(query):
    # 앞뒤 공백 제거
    query = query.strip()
    # 줄바꿈과 공백을 정상적으로 처리
    query = re.sub(r'\s*\n\s*', ' ', query)  # 줄바꿈을 단일 공백으로 대체
    # 여러 줄의 공백을 단일 공백으로 대체
    query = re.sub(r'\s{2,}', ' ', query)   # 다중 공백을 단일 공백으로 축소
    # 마지막 세미콜론 제거
    query = re.sub(r';$', '', query)
    # 코드블록 일부 제거
    query = re.sub(r'```sql*', '', query)
    query = re.sub(r'```*', '', query)
    return query
