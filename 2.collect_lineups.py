### 이 노트북은 `data/raw/games_master.csv`를 기반으로 실제 경기가 완료된 경기들의 선발 라인업 데이터를 수집합니다.

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

print(f"프로젝트 루트: {ROOT_DIR}")
print(f"데이터 저장 경로: {RAW_DATA_DIR}")

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
        
        headers = {
            "X-API-KEY": self.api_key,
            "X-TIMESTAMP": timestamp,
            "X-SIGNATURE": signature,
            "Accept": "application/json"
        }
        
        try:
            response = requests.get(f"{self.base_url}/{path}", params=sorted_params, headers=headers)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("STATIZ_API_KEY")
SECRET  = os.getenv("STATIZ_SECRET")

if not API_KEY or not SECRET:
    raise ValueError("Statiz API Key와 Secret Key가 설정되지 않았습니다. .env 파일을 확인해 주세요.")

api = StatizAPI(API_KEY, SECRET)

def extract_players(data):
    extracted = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and 'p_no' in item:
                extracted.append(item)
            else:
                extracted.extend(extract_players(item))
    elif isinstance(data, dict):
        for k, v in data.items():
            if k in ['result_cd', 'result_msg', 'update_time']: continue
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict) and 'p_no' in item: extracted.append(item)
            elif isinstance(v, dict):
                extracted.extend(extract_players(v))
    return extracted

def run_collection():
    master_path = os.path.join(RAW_DATA_DIR, "games_master.csv")
    output_file = os.path.join(RAW_DATA_DIR, "lineups.csv")
    
    if not os.path.exists(master_path):
        print(f"[Error] {master_path} 파일이 없습니다.")
        return

    df_master = pd.read_csv(master_path)
    df_target = df_master.dropna(subset=['homeScore', 'awayScore'])
    game_ids_master = df_target['s_no'].unique().tolist()
    
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        # games_master.csv에 존재하는 s_no만 유지 (시범경기 등 제거)
        df_existing = df_existing[df_existing['s_no'].isin(game_ids_master)]
        collected_ids = df_existing['s_no'].unique().tolist()
        game_ids = [s_no for s_no in game_ids_master if s_no not in collected_ids]
        print(f"기존 유효 수집 {len(collected_ids)}건 제외, 남은 {len(game_ids)}건 수집 시작.")
    else:
        df_existing = pd.DataFrame()
        game_ids = game_ids_master
        print(f"신규 수집 시작 (대상: {len(game_ids)}건)")

    all_lineups = []
    batch_size = 50
    success_count = 0
    
    pbar = tqdm(game_ids, desc="수집 중")
    for i, s_no in enumerate(pbar):
        res_data = api.call("prediction/gameLineup", {"s_no": s_no})
        players = extract_players(res_data)
        
        if players:
            # 수집된 데이터에 s_no 정보가 없을 수 있으므로 명시적 추가 (추후 필터링 용도)
            for p in players: p['s_no'] = s_no
            all_lineups.extend(players)
            success_count += 1
        
        if (i + 1) % batch_size == 0 and all_lineups:
            df_new = pd.DataFrame(all_lineups)
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
            # 저장 전 최종 필터링 (확실하게 정규시즌만 남김)
            df_final = df_final[df_final['s_no'].isin(game_ids_master)]
            df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
            df_existing = pd.read_csv(output_file)
            all_lineups = []
            pbar.set_postfix({"Success": success_count})
        
        time.sleep(0.3)

    if all_lineups:
        df_new = pd.DataFrame(all_lineups)
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
        df_final = df_final[df_final['s_no'].isin(game_ids_master)]
        df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n완료: 성공 {success_count}개 / 파일: {output_file}")

run_collection()