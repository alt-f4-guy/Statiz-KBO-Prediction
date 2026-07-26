"""주 모델 장애 시 사용하는 시점 누수 없는 최근 10경기 대체 확률."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _team_recent_games(
    history: pd.DataFrame,
    team: int,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    time = pd.to_datetime(history["result_available_datetime"], errors="coerce", utc=True)
    cutoff_utc = pd.to_datetime(cutoff, utc=True)
    completed = history["homeScore"].notna() & history["awayScore"].notna()
    involved = history["homeTeam"].eq(team) | history["awayTeam"].eq(team)
    sort_cols = ["_avail_datetime", "s_no"] if "s_no" in history.columns else ["_avail_datetime"]
    return (
        history.loc[completed & involved & time.lt(cutoff_utc)]
        .assign(_avail_datetime=time.loc[completed & involved & time.lt(cutoff_utc)])
        .sort_values(sort_cols)
        .tail(10)
    )


def _laplace_win_rate(games: pd.DataFrame, team: int) -> float:
    home_win = games["homeScore"].gt(games["awayScore"])
    away_win = games["awayScore"].gt(games["homeScore"])
    team_win = (
        games["homeTeam"].eq(team) & home_win
    ) | (
        games["awayTeam"].eq(team) & away_win
    )
    return float((team_win.sum() + 1) / (len(games) + 2))


def recent_ten_home_probability(
    history: pd.DataFrame,
    *,
    home_team: int,
    away_team: int,
    feature_cutoff_datetime: pd.Timestamp,
    league_home_win_rate: float,
) -> float:
    """두 팀의 최근 승률을 결합해 제한된 홈 승리 확률을 반환한다."""

    required = {
        "game_datetime",
        "result_available_datetime",
        "homeTeam",
        "awayTeam",
        "homeScore",
        "awayScore",
    }
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"대체 모델 입력 열 누락: {sorted(missing)}")

    home_games = _team_recent_games(
        history, int(home_team), feature_cutoff_datetime
    )
    away_games = _team_recent_games(
        history, int(away_team), feature_cutoff_datetime
    )
    if len(home_games) < 5 or len(away_games) < 5:
        return float(np.clip(league_home_win_rate, 0.35, 0.65))

    home_rate = _laplace_win_rate(home_games, int(home_team))
    away_rate = _laplace_win_rate(away_games, int(away_team))
    probability = (home_rate + 1 - away_rate) / 2
    return float(np.clip(probability, 0.35, 0.65))


def backtest_recent_ten(
    games: pd.DataFrame,
) -> pd.DataFrame:
    """경기별 기준시각 직전 최근 10경기로 대체 모델을 순차 백테스트한다."""

    required = {
        "s_no",
        "game_datetime",
        "feature_cutoff_datetime",
        "result_available_datetime",
        "homeTeam",
        "awayTeam",
        "homeScore",
        "awayScore",
    }
    missing = required.difference(games.columns)
    if missing:
        raise ValueError(f"대체 모델 백테스트 열 누락: {sorted(missing)}")

    frame = games.copy()
    frame["game_datetime"] = pd.to_datetime(
        frame["game_datetime"], errors="coerce", utc=True
    )
    frame["feature_cutoff_datetime"] = pd.to_datetime(
        frame["feature_cutoff_datetime"], errors="coerce", utc=True
    )
    frame["result_available_datetime"] = pd.to_datetime(
        frame["result_available_datetime"], errors="coerce", utc=True
    )
    frame = frame.dropna(
        subset=[
            "game_datetime",
            "feature_cutoff_datetime",
            "result_available_datetime",
            "homeScore",
            "awayScore",
        ]
    ).sort_values(["game_datetime", "s_no"])
    home_win = frame["homeScore"].gt(frame["awayScore"])
    away_win = frame["awayScore"].gt(frame["homeScore"])

    common = ["s_no", "result_available_datetime"]
    home_events = frame[common + ["homeTeam"]].rename(
        columns={"homeTeam": "team"}
    )
    home_events["team_win"] = home_win.astype(int).to_numpy()
    away_events = frame[common + ["awayTeam"]].rename(
        columns={"awayTeam": "team"}
    )
    away_events["team_win"] = away_win.astype(int).to_numpy()
    events = pd.concat([home_events, away_events], ignore_index=True).sort_values(
        ["team", "result_available_datetime", "s_no"]
    )
    grouped = events.groupby("team", sort=False)
    events["recent_games"] = (
        grouped["team_win"]
        .rolling(10, min_periods=1)
        .count()
        .reset_index(level=0, drop=True)
    )
    events["recent_wins"] = (
        grouped["team_win"]
        .rolling(10, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    requests_home = frame[
        ["s_no", "feature_cutoff_datetime", "homeTeam"]
    ].rename(columns={"homeTeam": "team"})
    requests_away = frame[
        ["s_no", "feature_cutoff_datetime", "awayTeam"]
    ].rename(columns={"awayTeam": "team"})

    def merge_team_requests(requests: pd.DataFrame) -> pd.DataFrame:
        left = requests.sort_values(["feature_cutoff_datetime", "team"])
        right = events.sort_values(["result_available_datetime", "s_no", "team"])
        return pd.merge_asof(
            left,
            right[["team", "result_available_datetime", "recent_games", "recent_wins"]],
            left_on="feature_cutoff_datetime",
            right_on="result_available_datetime",
            by="team",
            direction="backward",
            allow_exact_matches=False,
        ).sort_values("s_no")

    home_recent = merge_team_requests(requests_home)
    away_recent = merge_team_requests(requests_away)

    # 동일 시각 경기 결과가 리그 사전값에 들어가지 않도록 시각별로 누적한다.
    time_events = (
        frame.assign(home_win=home_win.astype(int))
        .groupby("result_available_datetime", as_index=False)
        .agg(games=("s_no", "size"), home_wins=("home_win", "sum"))
        .sort_values("result_available_datetime")
    )
    time_events["cum_games"] = time_events["games"].cumsum()
    time_events["cum_home_wins"] = time_events["home_wins"].cumsum()
    league = pd.merge_asof(
        frame[["s_no", "feature_cutoff_datetime"]].sort_values(
            "feature_cutoff_datetime"
        ),
        time_events[
            ["result_available_datetime", "cum_games", "cum_home_wins"]
        ].sort_values("result_available_datetime"),
        left_on="feature_cutoff_datetime",
        right_on="result_available_datetime",
        direction="backward",
        allow_exact_matches=False,
    ).sort_values("s_no")
    league_rate = (
        league["cum_home_wins"] / league["cum_games"].replace(0, np.nan)
    ).fillna(0.5)


    home_games = home_recent["recent_games"].fillna(0).to_numpy()
    away_games = away_recent["recent_games"].fillna(0).to_numpy()
    home_rate = (
        home_recent["recent_wins"].fillna(0).to_numpy() + 1
    ) / (home_games + 2)
    away_rate = (
        away_recent["recent_wins"].fillna(0).to_numpy() + 1
    ) / (away_games + 2)
    probability = (home_rate + 1 - away_rate) / 2
    insufficient = (home_games < 5) | (away_games < 5)
    probability[insufficient] = league_rate.to_numpy()[insufficient]
    probability = np.clip(probability, 0.35, 0.65)

    result = frame.sort_values("s_no")[
        ["s_no", "game_datetime", "homeScore", "awayScore"]
    ].copy()
    result["home_win_probability"] = probability
    result["target_home_win"] = result["homeScore"].gt(result["awayScore"]).astype(
        int
    )
    return result.loc[result["homeScore"].ne(result["awayScore"])].reset_index(
        drop=True
    )
