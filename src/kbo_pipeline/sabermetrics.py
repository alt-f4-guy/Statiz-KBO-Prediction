"""연도별 야구 지표 상수와 시점 기준 구장 팩터."""

from __future__ import annotations

import numpy as np
import pandas as pd


REFERENCE_RUNS_PER_PA = 0.115
DEFAULT_LEAGUE_ERA = 4.50
DEFAULT_FIP_CONSTANT = 3.10
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
    """기준시각 전에 결과가 가용한 경기만 사용해 구장 득점 팩터를 계산한다."""

    required = {
        "s_no",
        "feature_cutoff_datetime",
        "result_available_datetime",
        "s_code",
        "homeScore",
        "awayScore",
    }
    missing = required.difference(games.columns)
    if missing:
        raise ValueError(f"구장 팩터 계산 열 누락: {sorted(missing)}")

    requests = games[
        ["s_no", "feature_cutoff_datetime", "s_code"]
    ].copy()
    requests["feature_cutoff_datetime"] = pd.to_datetime(
        requests["feature_cutoff_datetime"], errors="coerce", utc=True
    )
    requests["_input_order"] = np.arange(len(requests))

    events = games[
        [
            "s_no",
            "result_available_datetime",
            "s_code",
            "homeScore",
            "awayScore",
        ]
    ].copy()
    events["result_available_datetime"] = pd.to_datetime(
        events["result_available_datetime"], errors="coerce", utc=True
    )
    events["total_runs"] = (
        pd.to_numeric(events["homeScore"], errors="coerce")
        + pd.to_numeric(events["awayScore"], errors="coerce")
    )
    events = events.dropna(
        subset=["result_available_datetime", "total_runs"]
    )

    time_totals = (
        events.groupby("result_available_datetime", as_index=False)["total_runs"]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values("result_available_datetime")
    )
    time_totals["league_runs_before"] = time_totals["sum"].cumsum()
    time_totals["league_games_before"] = time_totals["count"].cumsum()
    league = pd.merge_asof(
        requests.sort_values("feature_cutoff_datetime"),
        time_totals[
            [
                "result_available_datetime",
                "league_runs_before",
                "league_games_before",
            ]
        ],
        left_on="feature_cutoff_datetime",
        right_on="result_available_datetime",
        direction="backward",
        allow_exact_matches=False,
    )

    park_totals = (
        events.groupby(
            ["s_code", "result_available_datetime"],
            as_index=False,
            dropna=False,
        )["total_runs"]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values(["result_available_datetime", "s_code"])
    )
    grouped = park_totals.groupby("s_code", dropna=False)
    park_totals["park_runs_before"] = grouped["sum"].cumsum()
    park_totals["park_games_before"] = grouped["count"].cumsum()
    park = pd.merge_asof(
        requests.sort_values(["feature_cutoff_datetime", "s_code"]),
        park_totals[
            [
                "s_code",
                "result_available_datetime",
                "park_runs_before",
                "park_games_before",
            ]
        ],
        left_on="feature_cutoff_datetime",
        right_on="result_available_datetime",
        by="s_code",
        direction="backward",
        allow_exact_matches=False,
    )

    data = league[
        [
            "s_no",
            "_input_order",
            "league_runs_before",
            "league_games_before",
        ]
    ].merge(
        park[["s_no", "park_runs_before", "park_games_before"]],
        on="s_no",
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
    return (
        data.sort_values("_input_order")[
            ["s_no", "park_factor", "park_factor_source"]
        ]
        .reset_index(drop=True)
    )


def calculate_asof_kbo_constants(
    games: pd.DataFrame,
    pitching: pd.DataFrame,
    batting: pd.DataFrame,
) -> pd.DataFrame:
    """각 경기 기준시각 이전 리그 기록만으로 상수를 계산한다."""

    requests = games[["s_no", "year", "feature_cutoff_datetime"]].copy()
    requests["feature_cutoff_datetime"] = pd.to_datetime(
        requests["feature_cutoff_datetime"], errors="coerce", utc=True
    )
    requests["_input_order"] = np.arange(len(requests))

    pitching_columns = ["IP", "ER", "HR", "BB", "HP", "SO"]
    pitching_events = pitching[
        ["year", "event_datetime", *pitching_columns]
    ].copy()
    pitching_events["event_datetime"] = pd.to_datetime(
        pitching_events["event_datetime"], errors="coerce", utc=True
    )
    pitching_totals = (
        pitching_events.groupby(["year", "event_datetime"], as_index=False)[
            pitching_columns
        ]
        .sum(min_count=1)
        .sort_values(["event_datetime", "year"])
    )
    pitching_totals[pitching_columns] = pitching_totals.groupby(
        "year", sort=False
    )[pitching_columns].cumsum()

    batting_events = batting[["year", "event_datetime", "R", "PA"]].copy()
    batting_events["event_datetime"] = pd.to_datetime(
        batting_events["event_datetime"], errors="coerce", utc=True
    )
    batting_totals = (
        batting_events.groupby(["year", "event_datetime"], as_index=False)[
            ["R", "PA"]
        ]
        .sum(min_count=1)
        .sort_values(["event_datetime", "year"])
    )
    batting_totals[["R", "PA"]] = batting_totals.groupby(
        "year", sort=False
    )[["R", "PA"]].cumsum()

    left = requests.sort_values(["feature_cutoff_datetime", "year"])
    pitching_asof = pd.merge_asof(
        left,
        pitching_totals.sort_values(["event_datetime", "year"]),
        left_on="feature_cutoff_datetime",
        right_on="event_datetime",
        by="year",
        direction="backward",
        allow_exact_matches=False,
    )
    batting_asof = pd.merge_asof(
        left,
        batting_totals.sort_values(["event_datetime", "year"]),
        left_on="feature_cutoff_datetime",
        right_on="event_datetime",
        by="year",
        direction="backward",
        allow_exact_matches=False,
    )

    valid_ip = pitching_asof["IP"].replace(0, np.nan)
    league_era = pitching_asof["ER"] * 9 / valid_ip
    fip_component = (
        13 * pitching_asof["HR"]
        + 3 * (pitching_asof["BB"] + pitching_asof["HP"])
        - 2 * pitching_asof["SO"]
    ) / valid_ip
    runs_per_pa = batting_asof["R"] / batting_asof["PA"].replace(0, np.nan)
    scale = (runs_per_pa / REFERENCE_RUNS_PER_PA).clip(0.75, 1.25)

    result = pitching_asof[["s_no", "year", "_input_order"]].copy()
    result["league_era"] = league_era.to_numpy()
    result["fip_constant"] = (league_era - fip_component).to_numpy()
    result["league_runs_per_pa"] = runs_per_pa.to_numpy()
    for name, base_weight in BASE_LINEAR_WEIGHTS.items():
        result[name] = scale.to_numpy() * base_weight
    return (
        result.sort_values("_input_order")
        .drop(columns="_input_order")
        .reset_index(drop=True)
    )
