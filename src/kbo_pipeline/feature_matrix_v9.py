"""경기 시작 이전 정보만 사용하는 v9 피처 매트릭스 생성기."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from asof_features import mark_bullpen_candidates
from dataset_contract import build_feature_coverage, validate_final_dataset
from game_time import build_game_datetime_reference
from sabermetrics import (
    BASE_LINEAR_WEIGHTS,
    add_batting_environment,
    calculate_asof_kbo_constants,
    calculate_asof_park_factor,
    calculate_kbo_year_constants,
)


PITCHING_COLUMNS = ["IP", "ER", "H", "HR", "BB", "HP", "SO", "NP", "GS"]
BATTING_COLUMNS = [
    "PA",
    "AB",
    "H",
    "1B",
    "2B",
    "3B",
    "HR",
    "TB",
    "BB",
    "HP",
    "SO",
    "SF",
]


def _to_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _merge_asof_by_player_year(
    requests: pd.DataFrame,
    events: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    left = requests.copy()
    right = events[["p_no", "year", "event_datetime", *feature_columns]].copy()
    left["_player_year"] = (
        pd.to_numeric(left["p_no"], errors="coerce").fillna(-1).astype("int64")
        * 10_000
        + pd.to_numeric(left["year"], errors="coerce").fillna(-1).astype("int64")
    )
    right["_player_year"] = (
        pd.to_numeric(right["p_no"], errors="coerce").fillna(-1).astype("int64")
        * 10_000
        + pd.to_numeric(right["year"], errors="coerce").fillna(-1).astype("int64")
    )
    left["feature_cutoff_datetime"] = pd.to_datetime(
        left["feature_cutoff_datetime"], errors="coerce", utc=True
    )
    right["event_datetime"] = pd.to_datetime(
        right["event_datetime"], errors="coerce", utc=True
    )
    left["_input_order"] = np.arange(len(left))
    left = left.sort_values(["feature_cutoff_datetime", "_player_year"])
    right = right.sort_values(["event_datetime", "_player_year"])
    result = pd.merge_asof(
        left,
        right.drop(columns=["p_no", "year"]),
        left_on="feature_cutoff_datetime",
        right_on="event_datetime",
        by="_player_year",
        direction="backward",
        allow_exact_matches=False,
    )
    return (
        result.sort_values("_input_order")
        .drop(
            columns=["_player_year", "_input_order", "event_datetime"],
            errors="ignore",
        )
        .reset_index(drop=True)
    )


def _prepare_games(
    games: pd.DataFrame,
    *,
    include_unscored: bool = False,
) -> pd.DataFrame:
    selected = (
        games.copy()
        if include_unscored
        else games.dropna(subset=["homeScore", "awayScore"]).copy()
    )
    reference = build_game_datetime_reference(selected)
    selected = selected.merge(reference, on="s_no", how="inner")
    selected["game_datetime"] = pd.to_datetime(
        selected["game_datetime"], utc=True
    )
    selected["feature_cutoff_datetime"] = pd.to_datetime(
        selected["feature_cutoff_datetime"], utc=True
    )
    selected["year"] = selected["game_datetime"].dt.year
    selected["game_date_key"] = selected["game_datetime"].dt.strftime("%Y-%m-%d")
    return selected.sort_values(["game_datetime", "s_no"]).reset_index(drop=True)


def _prepare_events(
    day: pd.DataFrame,
    games: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = games[["s_no", "result_available_datetime", "year"]].rename(
        columns={"s_no": "s_no_key", "result_available_datetime": "event_datetime"}
    )
    events = day.copy()
    events["s_no_key"] = pd.to_numeric(events["s_no_key"], errors="coerce")
    events = events.merge(reference, on="s_no_key", how="inner", suffixes=("", "_game"))
    if "year_game" in events.columns:
        events["year"] = events.pop("year_game")

    pitching = _to_numeric(events, PITCHING_COLUMNS)
    pitching = pitching.loc[pitching["IP"].notna()].copy()
    pitching[PITCHING_COLUMNS] = pitching[PITCHING_COLUMNS].fillna(0)
    pitching.sort_values(["p_no", "year", "event_datetime", "s_no_key"], inplace=True)
    pitcher_group = pitching.groupby(["p_no", "year"], sort=False)
    pitching["current_g"] = pitcher_group.cumcount() + 1
    for column in ["GS", "IP", "ER", "H", "HR", "BB", "HP", "SO", "NP"]:
        pitching[f"current_{column.lower()}"] = pitcher_group[column].cumsum()

    relief_flag = pitching["GS"].eq(0).astype(float)
    for column in ["IP", "ER", "H", "HR", "BB", "HP", "SO", "NP"]:
        relief_value = pitching[column] * relief_flag
        pitching[f"_relief_{column.lower()}"] = relief_value
        pitching[f"relief_{column.lower()}"] = relief_value.groupby(
            [pitching["p_no"], pitching["year"]]
        ).cumsum()

    batting = _to_numeric(events, BATTING_COLUMNS)
    batting = batting.loc[batting["PA"].notna()].copy()
    batting[BATTING_COLUMNS] = batting[BATTING_COLUMNS].fillna(0)
    batting.sort_values(["p_no", "year", "event_datetime", "s_no_key"], inplace=True)
    batter_group = batting.groupby(["p_no", "year"], sort=False)
    for column in BATTING_COLUMNS:
        batting[f"current_{column.lower()}"] = batter_group[column].cumsum()
    return pitching.reset_index(drop=True), batting.reset_index(drop=True)


def _build_prior_constants(
    pitching: pd.DataFrame,
    batting: pd.DataFrame,
) -> pd.DataFrame:
    constants = calculate_kbo_year_constants(pitching)
    return add_batting_environment(constants, batting)


def _build_asof_constants(
    games: pd.DataFrame,
    pitching: pd.DataFrame,
    batting: pd.DataFrame,
    prior_constants: pd.DataFrame,
) -> pd.DataFrame:
    current = calculate_asof_kbo_constants(games, pitching, batting)
    previous = prior_constants.copy()
    previous["year"] = previous["year"] + 1
    fallback_columns = [
        "league_era",
        "fip_constant",
        "league_runs_per_pa",
        *BASE_LINEAR_WEIGHTS,
    ]
    current = current.merge(
        previous[["year", *fallback_columns]],
        on="year",
        how="left",
        suffixes=("", "_prior"),
    )
    for column in fallback_columns:
        current[column] = current[column].fillna(current[f"{column}_prior"])
    return current.drop(
        columns=[f"{column}_prior" for column in fallback_columns]
    )


def _prior_pitcher_table(
    season: pd.DataFrame,
    constants: pd.DataFrame,
) -> pd.DataFrame:
    data = _to_numeric(
        season,
        ["p_no", "year", "G", "GS", "IP", "ER", "HR", "BB", "HP", "SO"],
    )
    data = data.dropna(subset=["p_no", "year"]).copy()
    data = data.merge(
        constants[["year", "fip_constant", "league_era"]],
        on="year",
        how="left",
    )
    valid_ip = data["IP"].replace(0, np.nan)
    data["prior_fip"] = (
        13 * data["HR"]
        + 3 * (data["BB"] + data["HP"])
        - 2 * data["SO"]
    ) / valid_ip + data["fip_constant"]
    data.rename(
        columns={
            "G": "prior_g",
            "GS": "prior_gs",
            "IP": "prior_ip",
            "year": "prior_year",
        },
        inplace=True,
    )
    data["year"] = data["prior_year"] + 1
    return data[
        [
            "p_no",
            "year",
            "prior_g",
            "prior_gs",
            "prior_ip",
            "prior_fip",
        ]
    ]


def _pitcher_shrunk_features(
    requests: pd.DataFrame,
    pitching: pd.DataFrame,
    prior_pitcher: pd.DataFrame,
    constants: pd.DataFrame,
) -> pd.DataFrame:
    current_columns = [
        "current_g",
        "current_gs",
        "current_ip",
        "current_er",
        "current_hr",
        "current_bb",
        "current_hp",
        "current_so",
        "relief_ip",
        "relief_er",
        "relief_hr",
        "relief_bb",
        "relief_hp",
        "relief_so",
        "relief_np",
    ]
    result = _merge_asof_by_player_year(requests, pitching, current_columns)
    result = result.merge(prior_pitcher, on=["p_no", "year"], how="left")
    result = result.merge(
        constants[["s_no", "year", "fip_constant", "league_era"]],
        on=["s_no", "year"],
        how="left",
    )
    current_ip = result["current_ip"].fillna(0)
    current_component = (
        13 * result["current_hr"].fillna(0)
        + 3
        * (
            result["current_bb"].fillna(0)
            + result["current_hp"].fillna(0)
        )
        - 2 * result["current_so"].fillna(0)
    ) / current_ip.replace(0, np.nan)
    result["current_fip"] = current_component + result["fip_constant"]

    prior_ip = result["prior_ip"].fillna(0)
    current_weight = current_ip / (current_ip + 20.0)
    prior_weight = (1 - current_weight) * prior_ip / (prior_ip + 40.0)
    league_weight = 1 - current_weight - prior_weight
    result["shrunk_fip"] = (
        current_weight * result["current_fip"].fillna(result["league_era"])
        + prior_weight * result["prior_fip"].fillna(result["league_era"])
        + league_weight * result["league_era"]
    )
    result["feature_source"] = np.select(
        [current_ip.ge(5), prior_ip.gt(0)],
        ["current_season", "prior_season"],
        default="league_prior",
    )
    result["feature_missing"] = current_ip.eq(0) & prior_ip.eq(0)
    return result


def _starter_features(
    games: pd.DataFrame,
    lineups: pd.DataFrame,
    pitching: pd.DataFrame,
    prior_pitcher: pd.DataFrame,
    constants: pd.DataFrame,
) -> pd.DataFrame:
    starters = lineups.copy()
    starters["position"] = pd.to_numeric(starters["position"], errors="coerce")
    starters["p_no"] = pd.to_numeric(starters["p_no"], errors="coerce")
    starters["t_code"] = pd.to_numeric(starters["t_code"], errors="coerce")
    starters = starters.loc[starters["position"].eq(1)].drop_duplicates(
        ["s_no", "t_code"], keep="first"
    )
    team_games = _team_game_rows(games)
    requests = team_games.merge(
        starters[["s_no", "t_code", "p_no"]],
        on=["s_no", "t_code"],
        how="left",
    )
    requests.rename(columns={"p_no": "starter_p_no"}, inplace=True)
    requests["p_no"] = requests["starter_p_no"]
    featured = _pitcher_shrunk_features(
        requests, pitching, prior_pitcher, constants
    )
    return featured[
        [
            "s_no",
            "side",
            "t_code",
            "starter_p_no",
            "shrunk_fip",
            "feature_source",
            "feature_missing",
        ]
    ].rename(
        columns={
            "shrunk_fip": "sp_fip",
            "feature_source": "sp_source",
            "feature_missing": "sp_missing",
        }
    )


def _team_game_rows(games: pd.DataFrame) -> pd.DataFrame:
    common = [
        "s_no",
        "year",
        "game_datetime",
        "feature_cutoff_datetime",
        "game_date_key",
    ]
    home = games[common + ["homeTeam"]].rename(columns={"homeTeam": "t_code"})
    home["side"] = "home"
    away = games[common + ["awayTeam"]].rename(columns={"awayTeam": "t_code"})
    away["side"] = "away"
    return pd.concat([home, away], ignore_index=True)


def _prior_batter_table(
    season: pd.DataFrame,
    constants: pd.DataFrame,
) -> pd.DataFrame:
    data = _to_numeric(
        season,
        [
            "p_no",
            "year",
            "PA",
            "AB",
            "H",
            "1B",
            "2B",
            "3B",
            "HR",
            "TB",
            "BB",
            "HP",
            "SO",
            "SF",
        ],
    )
    data = data.dropna(subset=["p_no", "year"]).merge(
        constants,
        on="year",
        how="left",
    )
    data["prior_pa"] = data["PA"]
    data["prior_obp"] = (
        data["H"] + data["BB"] + data["HP"]
    ) / (data["AB"] + data["BB"] + data["HP"] + data["SF"]).replace(0, np.nan)
    data["prior_slg"] = data["TB"] / data["AB"].replace(0, np.nan)
    data["prior_iso"] = data["prior_slg"] - data["H"] / data["AB"].replace(
        0, np.nan
    )
    data["prior_bb_rate"] = data["BB"] / data["PA"].replace(0, np.nan)
    data["prior_k_rate"] = data["SO"] / data["PA"].replace(0, np.nan)
    singles = data["1B"].fillna(
        data["H"] - data["2B"] - data["3B"] - data["HR"]
    )
    data["prior_linear"] = (
        data["weight_bb"] * data["BB"]
        + data["weight_hp"] * data["HP"]
        + data["weight_1b"] * singles
        + data["weight_2b"] * data["2B"]
        + data["weight_3b"] * data["3B"]
        + data["weight_hr"] * data["HR"]
    ) / data["PA"].replace(0, np.nan)
    data["year"] = data["year"] + 1
    return data[
        [
            "p_no",
            "year",
            "prior_pa",
            "prior_obp",
            "prior_slg",
            "prior_iso",
            "prior_bb_rate",
            "prior_k_rate",
            "prior_linear",
        ]
    ]


def _league_batting_priors(
    season: pd.DataFrame,
    constants: pd.DataFrame,
) -> pd.DataFrame:
    prior = _prior_batter_table(season, constants)
    metrics = [
        "prior_obp",
        "prior_slg",
        "prior_iso",
        "prior_bb_rate",
        "prior_k_rate",
        "prior_linear",
    ]
    weighted = prior.dropna(subset=["prior_pa"]).copy()
    for metric in metrics:
        weighted[f"_{metric}"] = weighted[metric] * weighted["prior_pa"]
    aggregations = {f"_{metric}": "sum" for metric in metrics}
    aggregations["prior_pa"] = "sum"
    totals = weighted.groupby("year", as_index=False).agg(aggregations)
    for metric in metrics:
        totals[f"league_{metric.removeprefix('prior_')}"] = (
            totals.pop(f"_{metric}") / totals["prior_pa"].replace(0, np.nan)
        )
    totals = totals.drop(columns=["prior_pa"])
    result = constants[["year"]].drop_duplicates().merge(
        totals, on="year", how="left"
    )
    fallback = {
        "league_obp": 0.330,
        "league_slg": 0.400,
        "league_iso": 0.130,
        "league_bb_rate": 0.080,
        "league_k_rate": 0.200,
        "league_linear": 0.320,
    }
    for column, value in fallback.items():
        result[column] = result[column].fillna(value)
    return result


def _batter_features(
    games: pd.DataFrame,
    lineups: pd.DataFrame,
    batting: pd.DataFrame,
    season: pd.DataFrame,
    prior_constants: pd.DataFrame,
    asof_constants: pd.DataFrame,
) -> pd.DataFrame:
    lineup = lineups.copy()
    lineup["p_no"] = pd.to_numeric(lineup["p_no"], errors="coerce")
    lineup["t_code"] = pd.to_numeric(lineup["t_code"], errors="coerce")
    lineup["battingOrder"] = pd.to_numeric(
        lineup["battingOrder"], errors="coerce"
    )
    lineup = lineup.loc[
        lineup["battingOrder"].between(1, 9, inclusive="both")
    ].drop_duplicates(["s_no", "t_code", "p_no"])
    requests = _team_game_rows(games).merge(
        lineup[["s_no", "t_code", "p_no"]],
        on=["s_no", "t_code"],
        how="left",
    )
    cumulative_columns = [f"current_{column.lower()}" for column in BATTING_COLUMNS]
    result = _merge_asof_by_player_year(requests, batting, cumulative_columns)
    result = result.merge(
        _prior_batter_table(season, prior_constants),
        on=["p_no", "year"],
        how="left",
    ).merge(_league_batting_priors(season, prior_constants), on="year", how="left")
    result = result.merge(asof_constants, on=["s_no", "year"], how="left", suffixes=("", "_year"))

    pa = result["current_pa"].fillna(0)
    ab = result["current_ab"].fillna(0)
    current_metrics = {
        "obp": (
            result["current_h"].fillna(0)
            + result["current_bb"].fillna(0)
            + result["current_hp"].fillna(0)
        )
        / (
            ab
            + result["current_bb"].fillna(0)
            + result["current_hp"].fillna(0)
            + result["current_sf"].fillna(0)
        ).replace(0, np.nan),
        "slg": result["current_tb"].fillna(0) / ab.replace(0, np.nan),
        "bb_rate": result["current_bb"].fillna(0) / pa.replace(0, np.nan),
        "k_rate": result["current_so"].fillna(0) / pa.replace(0, np.nan),
    }
    current_metrics["iso"] = current_metrics["slg"] - (
        result["current_h"].fillna(0) / ab.replace(0, np.nan)
    )
    current_metrics["linear"] = (
        result["weight_bb"] * result["current_bb"].fillna(0)
        + result["weight_hp"] * result["current_hp"].fillna(0)
        + result["weight_1b"] * result["current_1b"].fillna(0)
        + result["weight_2b"] * result["current_2b"].fillna(0)
        + result["weight_3b"] * result["current_3b"].fillna(0)
        + result["weight_hr"] * result["current_hr"].fillna(0)
    ) / pa.replace(0, np.nan)

    prior_pa = result["prior_pa"].fillna(0)
    current_weight = pa / (pa + 50.0)
    prior_weight = (1 - current_weight) * prior_pa / (prior_pa + 100.0)
    league_weight = 1 - current_weight - prior_weight
    for metric, current_value in current_metrics.items():
        result[f"bat_{metric}"] = (
            current_weight * current_value.fillna(result[f"league_{metric}"])
            + prior_weight
            * result[f"prior_{metric}"].fillna(result[f"league_{metric}"])
            + league_weight * result[f"league_{metric}"]
        )
    result["bat_source"] = np.select(
        [pa.ge(10), prior_pa.gt(0)],
        ["current_season", "prior_season"],
        default="league_prior",
    )
    result["bat_missing"] = pa.eq(0) & prior_pa.eq(0)

    metrics = [
        "bat_obp",
        "bat_slg",
        "bat_iso",
        "bat_bb_rate",
        "bat_k_rate",
        "bat_linear",
    ]
    aggregated = result.groupby(["s_no", "side"], as_index=False).agg(
        **{metric: (metric, "mean") for metric in metrics},
        bat_missing=("bat_missing", "mean"),
        current_count=("bat_source", lambda values: values.eq("current_season").sum()),
        prior_count=("bat_source", lambda values: values.eq("prior_season").sum()),
    )
    aggregated["bat_source"] = np.select(
        [aggregated["current_count"].ge(5), aggregated["prior_count"].gt(0)],
        ["current_season", "prior_season"],
        default="league_prior",
    )
    return aggregated.drop(columns=["current_count", "prior_count"])


def _bounded_kbb(
    strikeouts: pd.Series,
    walks: pd.Series,
) -> pd.Series:
    """볼넷이 0인 경우에도 의미 있는 K/BB 비율을 반환한다."""
    so = pd.to_numeric(strikeouts, errors="coerce").fillna(0)
    bb = pd.to_numeric(walks, errors="coerce").fillna(0)
    ratio = np.select(
        [bb.gt(0), so.gt(0)],
        [so / bb.where(bb.gt(0), 1.0), 10.0],
        default=2.0,
    )
    return pd.Series(ratio, index=so.index, dtype=float).clip(0.25, 10.0)


def _bullpen_features(
    games: pd.DataFrame,
    rosters: pd.DataFrame,
    starters: pd.DataFrame,
    pitching: pd.DataFrame,
    prior_pitcher: pd.DataFrame,
    constants: pd.DataFrame,
) -> pd.DataFrame:
    roster = rosters.copy()
    roster["game_date_key"] = pd.to_datetime(
        roster["pj_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    roster["p_no"] = pd.to_numeric(roster["p_no"], errors="coerce")
    roster["t_code"] = pd.to_numeric(roster["t_code"], errors="coerce")
    requests = _team_game_rows(games).merge(
        roster[["game_date_key", "t_code", "p_no"]],
        on=["game_date_key", "t_code"],
        how="left",
    ).merge(
        starters[["s_no", "side", "starter_p_no"]],
        on=["s_no", "side"],
        how="left",
    )
    featured = _pitcher_shrunk_features(
        requests, pitching, prior_pitcher, constants
    )
    featured["has_pitching_history"] = (
        featured["current_g"].fillna(0).gt(0)
        | featured["prior_g"].fillna(0).gt(0)
    )
    candidates = mark_bullpen_candidates(featured)
    eligible = candidates.loc[candidates["is_bullpen_candidate"]].copy()
    eligible["prior_fip_numerator"] = (
        eligible["prior_fip"] * eligible["prior_ip"]
    )

    grouped = eligible.groupby(["s_no", "side"], as_index=False).agg(
        current_ip=("relief_ip", "sum"),
        current_er=("relief_er", "sum"),
        current_hr=("relief_hr", "sum"),
        current_bb=("relief_bb", "sum"),
        current_hp=("relief_hp", "sum"),
        current_so=("relief_so", "sum"),
        prior_ip=("prior_ip", "sum"),
        prior_fip_numerator=("prior_fip_numerator", "sum"),
        bullpen_candidate_count=("p_no", "nunique"),
    )
    grouped["prior_fip_weighted"] = grouped["prior_fip_numerator"] / grouped[
        "prior_ip"
    ].replace(0, np.nan)
    base = _team_game_rows(games)[["s_no", "side", "year"]]
    grouped = base.merge(grouped, on=["s_no", "side"], how="left").merge(
        constants[["s_no", "year", "fip_constant", "league_era"]],
        on=["s_no", "year"],
        how="left",
    )
    current_ip = grouped["current_ip"].fillna(0)
    current_fip = (
        13 * grouped["current_hr"].fillna(0)
        + 3
        * (
            grouped["current_bb"].fillna(0)
            + grouped["current_hp"].fillna(0)
        )
        - 2 * grouped["current_so"].fillna(0)
    ) / current_ip.replace(0, np.nan) + grouped["fip_constant"]
    current_era = grouped["current_er"].fillna(0) * 9 / current_ip.replace(
        0, np.nan
    )
    prior_ip = grouped["prior_ip"].fillna(0)
    current_weight = current_ip / (current_ip + 40.0)
    prior_weight = (1 - current_weight) * prior_ip / (prior_ip + 80.0)
    league_weight = 1 - current_weight - prior_weight
    prior_fip = grouped["prior_fip_weighted"].fillna(grouped["league_era"])
    grouped["rp_fip"] = (
        current_weight * current_fip.fillna(grouped["league_era"])
        + prior_weight * prior_fip
        + league_weight * grouped["league_era"]
    )
    grouped["rp_era"] = (
        current_weight * current_era.fillna(grouped["league_era"])
        + (1 - current_weight) * grouped["league_era"]
    )
    grouped["rp_k_bb"] = _bounded_kbb(
        grouped["current_so"].fillna(0),
        grouped["current_bb"].fillna(0),
    )
    grouped["rp_source"] = np.select(
        [current_ip.ge(10), prior_ip.gt(0)],
        ["current_season", "prior_season"],
        default="league_prior",
    )
    grouped["rp_missing"] = current_ip.eq(0) & prior_ip.eq(0)
    return grouped[
        [
            "s_no",
            "side",
            "rp_fip",
            "rp_era",
            "rp_k_bb",
            "bullpen_candidate_count",
            "rp_source",
            "rp_missing",
        ]
    ]


def _merge_team_asof(
    requests: pd.DataFrame,
    events: pd.DataFrame,
    cutoff_column: str,
    features: Sequence[str],
) -> pd.DataFrame:
    left = requests.copy()
    right = events[["t_code", "event_datetime", *features]].copy()
    left["_input_order"] = np.arange(len(left))
    left[cutoff_column] = pd.to_datetime(left[cutoff_column], utc=True)
    right["event_datetime"] = pd.to_datetime(right["event_datetime"], utc=True)
    left = left.sort_values([cutoff_column, "t_code"])
    right = right.sort_values(["event_datetime", "t_code"])
    result = pd.merge_asof(
        left,
        right,
        left_on=cutoff_column,
        right_on="event_datetime",
        by="t_code",
        direction="backward",
        allow_exact_matches=False,
    )
    return result.sort_values("_input_order").drop(
        columns=["_input_order", "event_datetime"], errors="ignore"
    )


def _bullpen_fatigue(
    games: pd.DataFrame,
    pitching: pd.DataFrame,
) -> pd.DataFrame:
    relief = pitching.loc[pitching["GS"].eq(0)].copy()
    team_events = (
        relief.groupby(["t_code", "event_datetime"], as_index=False)[["IP", "NP"]]
        .sum()
        .sort_values(["t_code", "event_datetime"])
    )
    grouped = team_events.groupby("t_code", sort=False)
    team_events["cum_ip"] = grouped["IP"].cumsum()
    team_events["cum_np"] = grouped["NP"].cumsum()
    team_events["last10_ip"] = (
        grouped["IP"].rolling(10, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    team_events["last10_np"] = (
        grouped["NP"].rolling(10, min_periods=1).sum().reset_index(level=0, drop=True)
    )

    requests = _team_game_rows(games)
    latest = _merge_team_asof(
        requests,
        team_events,
        "feature_cutoff_datetime",
        ["cum_ip", "cum_np", "last10_ip", "last10_np"],
    )
    for days in (1, 3):
        shifted_column = f"_cutoff_{days}d"
        latest[shifted_column] = (
            latest["feature_cutoff_datetime"] - pd.Timedelta(days=days)
        )
        earlier = _merge_team_asof(
            latest[
                [
                    "s_no",
                    "side",
                    "t_code",
                    shifted_column,
                ]
            ],
            team_events,
            shifted_column,
            ["cum_ip", "cum_np"],
        )
        latest[f"rp_ip_{days}d"] = latest["cum_ip"].fillna(0) - earlier[
            "cum_ip"
        ].fillna(0).to_numpy()
        latest[f"rp_np_{days}d"] = latest["cum_np"].fillna(0) - earlier[
            "cum_np"
        ].fillna(0).to_numpy()
    return latest[
        [
            "s_no",
            "side",
            "rp_ip_1d",
            "rp_np_1d",
            "rp_ip_3d",
            "rp_np_3d",
            "last10_ip",
            "last10_np",
        ]
    ].rename(
        columns={"last10_ip": "rp_ip_last10", "last10_np": "rp_np_last10"}
    )


def _wide_side_features(
    features: pd.DataFrame,
    value_columns: Sequence[str],
) -> pd.DataFrame:
    home = features.loc[features["side"].eq("home"), ["s_no", *value_columns]].rename(
        columns={column: f"home_{column}" for column in value_columns}
    )
    away = features.loc[features["side"].eq("away"), ["s_no", *value_columns]].rename(
        columns={column: f"away_{column}" for column in value_columns}
    )
    return home.merge(away, on="s_no", how="outer")


def build_feature_matrix_v9(
    games: pd.DataFrame,
    lineups: pd.DataFrame,
    rosters: pd.DataFrame,
    day: pd.DataFrame,
    season: pd.DataFrame,
    *,
    include_unscored: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """v9 데이터, 경기별 커버리지, 연도 상수를 함께 반환한다."""

    game_rows = _prepare_games(games, include_unscored=include_unscored)
    pitching, batting = _prepare_events(day, game_rows)
    prior_constants = _build_prior_constants(pitching, batting)
    asof_constants = _build_asof_constants(game_rows, pitching, batting, prior_constants)
    prior_pitcher = _prior_pitcher_table(season, prior_constants)

    starters = _starter_features(
        game_rows,
        lineups,
        pitching,
        prior_pitcher,
        asof_constants,
    )
    batters = _batter_features(
        game_rows,
        lineups,
        batting,
        season,
        prior_constants,
        asof_constants,
    )
    bullpen = _bullpen_features(
        game_rows,
        rosters,
        starters,
        pitching,
        prior_pitcher,
        asof_constants,
    )
    fatigue = _bullpen_fatigue(game_rows, pitching)

    starter_wide = _wide_side_features(
        starters, ["sp_fip", "sp_source", "sp_missing"]
    )
    batter_wide = _wide_side_features(
        batters,
        [
            "bat_obp",
            "bat_slg",
            "bat_iso",
            "bat_bb_rate",
            "bat_k_rate",
            "bat_linear",
            "bat_source",
            "bat_missing",
        ],
    )
    bullpen_wide = _wide_side_features(
        bullpen,
        [
            "rp_fip",
            "rp_era",
            "rp_k_bb",
            "bullpen_candidate_count",
            "rp_source",
            "rp_missing",
        ],
    )
    fatigue_wide = _wide_side_features(
        fatigue,
        [
            "rp_ip_1d",
            "rp_np_1d",
            "rp_ip_3d",
            "rp_np_3d",
            "rp_ip_last10",
            "rp_np_last10",
        ],
    )
    park = calculate_asof_park_factor(game_rows)
    data = (
        game_rows[
            [
                "s_no",
                "game_datetime",
                "year",
                "homeTeam",
                "awayTeam",
                "homeScore",
                "awayScore",
                "feature_cutoff_datetime",
            ]
        ]
        .merge(park, on="s_no", how="left")
        .merge(starter_wide, on="s_no", how="left")
        .merge(batter_wide, on="s_no", how="left")
        .merge(bullpen_wide, on="s_no", how="left")
        .merge(fatigue_wide, on="s_no", how="left")
    )

    # 정의가 있는 사전값만 열별로 보완하고 결측 여부는 별도 열로 보존한다.
    league_era = data["s_no"].map(asof_constants.set_index("s_no")["league_era"])
    for side in ("home", "away"):
        data[f"{side}_sp_missing"] = data[f"{side}_sp_missing"].fillna(True)
        data[f"{side}_rp_missing"] = data[f"{side}_rp_missing"].fillna(True)
        data[f"{side}_bat_missing"] = data[f"{side}_bat_missing"].fillna(1.0)
        data[f"{side}_sp_source"] = data[f"{side}_sp_source"].fillna(
            "league_prior"
        )
        data[f"{side}_rp_source"] = data[f"{side}_rp_source"].fillna(
            "league_prior"
        )
        data[f"{side}_bat_source"] = data[f"{side}_bat_source"].fillna(
            "league_prior"
        )
        data[f"{side}_sp_fip"] = data[f"{side}_sp_fip"].fillna(league_era)
        data[f"{side}_rp_fip"] = data[f"{side}_rp_fip"].fillna(league_era)
        data[f"{side}_rp_era"] = data[f"{side}_rp_era"].fillna(league_era)
        data[f"{side}_rp_k_bb"] = data[f"{side}_rp_k_bb"].fillna(2.0)
        data[f"{side}_bullpen_candidate_count"] = data[
            f"{side}_bullpen_candidate_count"
        ].fillna(0)
        for fatigue_column in (
            "rp_ip_1d",
            "rp_np_1d",
            "rp_ip_3d",
            "rp_np_3d",
            "rp_ip_last10",
            "rp_np_last10",
        ):
            data[f"{side}_{fatigue_column}"] = data[
                f"{side}_{fatigue_column}"
            ].fillna(0)

    paired = [
        "sp_fip",
        "rp_fip",
        "rp_era",
        "rp_k_bb",
        "bat_obp",
        "bat_slg",
        "bat_iso",
        "bat_bb_rate",
        "bat_k_rate",
        "bat_linear",
    ]
    for metric in paired:
        data[f"{metric}_diff"] = data[f"home_{metric}"] - data[f"away_{metric}"]

    if not include_unscored:
        validate_final_dataset(data)
    return data, build_feature_coverage(data), prior_constants
