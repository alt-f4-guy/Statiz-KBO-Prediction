"""연도별 야구 지표 상수와 시점 기준 구장 팩터."""

from __future__ import annotations

import numpy as np
import pandas as pd


REFERENCE_RUNS_PER_PA = 0.115
BASE_LINEAR_WEIGHTS = {
    "weight_bb": 0.69,
    "weight_hp": 0.72,
    "weight_1b": 0.88,
    "weight_2b": 1.247,
    "weight_3b": 1.578,
    "weight_hr": 2.031,
}


def calculate_kbo_year_constants(pitching: pd.DataFrame) -> pd.DataFrame:
    """연도별 리그 합계에서 FIP 상수와 득점 환경 보정치를 계산한다."""

    required = {"year", "IP", "ER", "HR", "BB", "HP", "SO"}
    missing = required.difference(pitching.columns)
    if missing:
        raise ValueError(f"연도 상수 계산 열 누락: {sorted(missing)}")

    numeric_columns = ["year", "IP", "ER", "HR", "BB", "HP", "SO"]
    data = pitching[numeric_columns].copy()
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    totals = data.groupby("year", dropna=True).sum(min_count=1)
    valid_ip = totals["IP"].replace(0, np.nan)
    league_era = totals["ER"] * 9 / valid_ip
    fip_component = (
        13 * totals["HR"]
        + 3 * (totals["BB"] + totals["HP"])
        - 2 * totals["SO"]
    ) / valid_ip

    result = pd.DataFrame(
        {
            "year": totals.index.astype(int),
            "league_era": league_era,
            "fip_constant": league_era - fip_component,
            "league_ip": totals["IP"],
        }
    ).reset_index(drop=True)
    return result


def add_batting_environment(
    constants: pd.DataFrame,
    batting: pd.DataFrame,
) -> pd.DataFrame:
    """연도별 득점 환경으로 선형 가중치의 척도를 조정한다."""

    required = {"year", "R", "PA"}
    missing = required.difference(batting.columns)
    if missing:
        raise ValueError(f"타격 환경 계산 열 누락: {sorted(missing)}")
    data = batting[["year", "R", "PA"]].copy()
    for column in ("year", "R", "PA"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    totals = data.groupby("year").sum(min_count=1)
    runs_per_pa = totals["R"] / totals["PA"].replace(0, np.nan)
    scale = (runs_per_pa / REFERENCE_RUNS_PER_PA).clip(0.75, 1.25)

    environment = pd.DataFrame(
        {
            "year": totals.index.astype(int),
            "league_runs_per_pa": runs_per_pa,
        }
    ).reset_index(drop=True)
    for name, base_weight in BASE_LINEAR_WEIGHTS.items():
        environment[name] = environment["year"].map(scale) * base_weight
    return constants.merge(environment, on="year", how="outer")


def calculate_asof_park_factor(
    games: pd.DataFrame,
    *,
    prior_games: float = 50.0,
) -> pd.DataFrame:
    """각 경기보다 먼저 끝난 경기만 사용해 구장 득점 팩터를 계산한다."""

    required = {
        "s_no",
        "game_datetime",
        "s_code",
        "homeScore",
        "awayScore",
    }
    missing = required.difference(games.columns)
    if missing:
        raise ValueError(f"구장 팩터 계산 열 누락: {sorted(missing)}")

    data = games[list(required)].copy()
    data["game_datetime"] = pd.to_datetime(
        data["game_datetime"], errors="coerce", utc=True
    )
    data["total_runs"] = (
        pd.to_numeric(data["homeScore"], errors="coerce")
        + pd.to_numeric(data["awayScore"], errors="coerce")
    )

    time_totals = (
        data.groupby("game_datetime", as_index=False, dropna=False)["total_runs"]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values("game_datetime")
    )
    time_totals["league_runs_before"] = time_totals["sum"].cumsum().shift(1)
    time_totals["league_games_before"] = time_totals["count"].cumsum().shift(1)

    park_totals = (
        data.groupby(["s_code", "game_datetime"], as_index=False, dropna=False)[
            "total_runs"
        ]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values(["s_code", "game_datetime"])
    )
    grouped = park_totals.groupby("s_code", dropna=False)
    park_totals["park_runs_before"] = grouped["sum"].cumsum() - park_totals["sum"]
    park_totals["park_games_before"] = (
        grouped["count"].cumsum() - park_totals["count"]
    )

    data = data.merge(
        time_totals[
            ["game_datetime", "league_runs_before", "league_games_before"]
        ],
        on="game_datetime",
        how="left",
    ).merge(
        park_totals[
            [
                "s_code",
                "game_datetime",
                "park_runs_before",
                "park_games_before",
            ]
        ],
        on=["s_code", "game_datetime"],
        how="left",
    )
    league_average = data["league_runs_before"] / data[
        "league_games_before"
    ].replace(0, np.nan)
    park_average = data["park_runs_before"] / data["park_games_before"].replace(
        0, np.nan
    )
    raw_factor = park_average / league_average.replace(0, np.nan)
    weight = data["park_games_before"] / (
        data["park_games_before"] + prior_games
    )
    data["park_factor"] = (weight * raw_factor + (1 - weight)).fillna(1.0)
    data["park_factor_source"] = np.where(
        data["park_games_before"].gt(0),
        "historical_shrunk",
        "league_prior",
    )
    return data[["s_no", "park_factor", "park_factor_source"]].sort_values(
        "s_no"
    ).reset_index(drop=True)
