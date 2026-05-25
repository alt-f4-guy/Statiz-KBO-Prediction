import os
import time
import hmac
import hashlib
import requests
import pandas as pd
from urllib.parse import quote
from tqdm.auto import tqdm
import json

# 1. 경로 설정
ROOT_DIR = os.path.abspath(".")
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

class StatizAPI:
    def __init__(self, api_key, secret):
        self.api_key = api_key
        self.secret = secret
        self.base_url = "https://api.statiz.co.kr/baseballApi"

    def _make_signature(self, method, path, query_str, timestamp):
        payload = f"{method}|{path}|{query_str}|{timestamp}"
        return hmac.new(
            self.secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    def call(self, path, params):
        timestamp = str(int(time.time()))
        safe = "-_.!~*'()"
        sorted_params = {k: str(params[k]) for k in sorted(params.keys())}
        query_str = "&".join(f"{quote(k, safe=safe)}={quote(v, safe=safe)}" for k, v in sorted_params.items())
        signature = self._make_signature("GET", path, query_str, timestamp)
        headers = {"X-API-KEY": self.api_key, "X-TIMESTAMP": timestamp, "X-SIGNATURE": signature, "Accept": "application/json"}
        try:
            response = requests.get(f"{self.base_url}/{path}", params=sorted_params, headers=headers, timeout=30)
            if response.status_code == 200: return response.json()
            elif response.status_code == 429:
                print(f"\n[429] Rate limit. Waiting 60s...")
                time.sleep(60)
                return self.call(path, params)
            return None
        except Exception as e:
            print(f"\n[Error] {e}")
            return None

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("STATIZ_API_KEY")
SECRET  = os.getenv("STATIZ_SECRET")

if not API_KEY or not SECRET:
    raise ValueError("Statiz API Key와 Secret Key가 설정되지 않았습니다. .env 파일을 확인해 주세요.")

api = StatizAPI(API_KEY, SECRET)

def run_roster_collection():
    master_path = os.path.join(RAW_DATA_DIR, "games_master.csv")
    output_file = os.path.join(RAW_DATA_DIR, "rosters.csv")
    
    if not os.path.exists(master_path):
        print(f"[Error] {master_path} 파일이 없습니다.")
        return

    # 1. 수집 대상 날짜/팀 조합 추출
    df_master = pd.read_csv(master_path)
    # 날짜 형식 변환 (YYYY-MM-DD)
    df_master['date_str'] = df_master.apply(lambda x: f"{int(x['year'])}-{int(x['month']):02d}-{int(x['day']):02d}", axis=1)
    
    # 홈/어웨이 팀별로 날짜 조합 생성
    home_tasks = df_master[['date_str', 'homeTeam']].rename(columns={'homeTeam': 't_code'})
    away_tasks = df_master[['date_str', 'awayTeam']].rename(columns={'awayTeam': 't_code'})
    all_tasks = pd.concat([home_tasks, away_tasks]).drop_duplicates().sort_values('date_str')
    
    # 2. 기존 수집 데이터 로드 및 제외
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        # pj_date와 t_code를 기준으로 중복 체크
        df_existing['date_str_temp'] = df_existing['pj_date'].astype(str)
        collected_combinations = set(zip(df_existing['date_str_temp'], df_existing['t_code'].astype(str)))
        
        combos = zip(all_tasks['date_str'].astype(str), all_tasks['t_code'].astype(str))
        mask = [c not in collected_combinations for c in combos]
        target_tasks = all_tasks[mask]
        print(f"기존 {len(collected_combinations)}건 확인, 새로 수집할 대상 {len(target_tasks)}건.")
    else:
        df_existing = pd.DataFrame()
        target_tasks = all_tasks
        print(f"신규 수집 시작 (총 {len(target_tasks)}건)")

    if target_tasks.empty:
        print("모든 데이터가 최신입니다.")
        return

    # 3. API 수집
    all_rosters = []
    batch_size = 50
    
    pbar = tqdm(target_tasks.iterrows(), total=len(target_tasks), desc="로스터 수집 중")
    for i, (_, task) in enumerate(pbar):
        date_str = task['date_str']
        t_code = task['t_code']
        
        res_data = api.call("prediction/playerRoster", {"t_code": t_code, "date": date_str})
        
        if res_data and isinstance(res_data, dict):
            # 숫자 키로 들어오는 선수 데이터만 추출
            for k, v in res_data.items():
                if k.isdigit() and isinstance(v, dict):
                    # pj_date, t_code가 누락된 경우를 대비해 명시적 추가
                    item = v.copy()
                    item['pj_date'] = date_str
                    item['t_code'] = t_code
                    all_rosters.append(item)
        
        # 중간 저장
        if (i + 1) % batch_size == 0 and all_rosters:
            df_new = pd.DataFrame(all_rosters)
            df_final = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates(subset=['pj_date', 'p_no', 't_code'])
            df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
            df_existing = pd.read_csv(output_file)
            all_rosters = []
        
        time.sleep(0.3)

    # 4. 최종 저장
    if all_rosters:
        df_new = pd.DataFrame(all_rosters)
        df_final = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates(subset=['pj_date', 'p_no', 't_code'])
        df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n완료! 로스터 데이터가 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    run_roster_collection()
