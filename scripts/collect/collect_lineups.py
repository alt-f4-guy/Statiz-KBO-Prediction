### 이 노트북은 `data/raw/games_master.csv`를 기반으로 실제 경기가 완료된 경기들의 선발 라인업 데이터를 수집합니다.

import time
import pandas as pd
from tqdm.auto import tqdm
from pipeline_config import PROJECT_ROOT, RAW_DATA_DIR, load_api_credentials
from io_utils import atomic_to_csv
from statiz_api import StatizAPI, StatizAPIError, extract_players

# 1. 경로 설정
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"프로젝트 루트: {PROJECT_ROOT}")
print(f"데이터 저장 경로: {RAW_DATA_DIR}")

def run_collection():
    credentials = load_api_credentials()
    api = StatizAPI(credentials.api_key, credentials.secret)
    master_path = RAW_DATA_DIR / "games_master.csv"
    output_file = RAW_DATA_DIR / "lineups.csv"
    
    if not master_path.exists():
        print(f"[Error] {master_path} 파일이 없습니다.")
        return

    df_master = pd.read_csv(master_path)
    df_target = df_master.dropna(subset=['homeScore', 'awayScore'])
    game_ids_master = df_target['s_no'].unique().tolist()
    
    if output_file.exists():
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
        try:
            res_data = api.get("prediction/gameLineup", {"s_no": s_no})
        except StatizAPIError as exc:
            print(f"[경고] s_no={s_no} 라인업 수집 실패: {exc}")
            res_data = None
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
            atomic_to_csv(df_final, output_file)
            df_existing = pd.read_csv(output_file)
            all_lineups = []
            pbar.set_postfix({"Success": success_count})
        
        time.sleep(3.0)

    if all_lineups:
        df_new = pd.DataFrame(all_lineups)
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
        df_final = df_final[df_final['s_no'].isin(game_ids_master)]
        atomic_to_csv(df_final, output_file)
    
    print(f"\n완료: 성공 {success_count}개 / 파일: {output_file}")

if __name__ == "__main__":
    run_collection()
