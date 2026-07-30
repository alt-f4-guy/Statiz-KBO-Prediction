import time
import pandas as pd
from tqdm.auto import tqdm
from pipeline_config import RAW_DATA_DIR, load_api_credentials
from io_utils import atomic_to_csv
from statiz_api import StatizAPI, StatizAPIError

# 1. 경로 설정
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

def run_roster_collection():
    credentials = load_api_credentials()
    api = StatizAPI(credentials.api_key, credentials.secret)
    master_path = RAW_DATA_DIR / "games_master.csv"
    output_file = RAW_DATA_DIR / "rosters.csv"
    
    if not master_path.exists():
        print(f"[Error] {master_path} 파일이 없습니다.")
        return

    # 1. 수집 대상 날짜/팀 조합 추출
    df_master = pd.read_csv(master_path)
    # 날짜 형식 변환 (YYYY-MM-DD)
    df_master["date_str"] = pd.to_datetime(
        df_master[["year", "month", "day"]], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    
    # 홈/어웨이 팀별로 날짜 조합 생성
    home_tasks = df_master[['date_str', 'homeTeam']].rename(columns={'homeTeam': 't_code'})
    away_tasks = df_master[['date_str', 'awayTeam']].rename(columns={'awayTeam': 't_code'})
    all_tasks = pd.concat([home_tasks, away_tasks]).drop_duplicates().sort_values('date_str')
    
    # 2. 기존 수집 데이터 로드 및 제외
    if output_file.exists():
        df_existing = pd.read_csv(output_file)
        # pj_date와 t_code를 기준으로 중복 체크
        df_existing['date_str_temp'] = df_existing['pj_date'].astype(str)
        collected_combinations = set(zip(df_existing['date_str_temp'], df_existing['t_code'].astype(str)))
        
        task_keys = pd.MultiIndex.from_frame(
            all_tasks[["date_str", "t_code"]].astype(str)
        )
        existing_keys = pd.MultiIndex.from_tuples(
            collected_combinations, names=["date_str", "t_code"]
        )
        target_tasks = all_tasks.loc[~task_keys.isin(existing_keys)].copy()
            
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
    
    pbar = tqdm(target_tasks.itertuples(index=False), total=len(target_tasks), desc="로스터 수집 중")
    for i, task in enumerate(pbar):
        date_str = task.date_str
        t_code = task.t_code
        
        try:
            res_data = api.get(
                "prediction/playerRoster",
                {"t_code": t_code, "date": date_str},
            )
        except StatizAPIError as exc:
            print(f"[경고] {date_str} 팀 {t_code} 로스터 수집 실패: {exc}")
            res_data = None
        
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
            atomic_to_csv(df_final, output_file)
            df_existing = pd.read_csv(output_file)
            all_rosters = []
        
        time.sleep(3.0)

    # 4. 최종 저장
    if all_rosters:
        df_new = pd.DataFrame(all_rosters)
        df_final = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates(subset=['pj_date', 'p_no', 't_code'])
        atomic_to_csv(df_final, output_file)
    
    print(f"\n완료! 로스터 데이터가 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    run_roster_collection()
