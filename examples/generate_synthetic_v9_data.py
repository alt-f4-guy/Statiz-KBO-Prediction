"""고정 시드로 공개 가능한 v9 합성 학습 데이터셋을 생성한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
GAMES_PER_SEASON = 36
TEAM_CODES = np.array(["TEAM_01", "TEAM_02", "TEAM_03", "TEAM_04", "TEAM_05"])
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "data"
    / "final_training_set_v9_sample.csv"
)


def _bounded(value: float, lower: float, upper: float) -> float:
    """합성 야구 지표가 현실적인 범위를 벗어나지 않게 제한한다."""

    return float(np.clip(value, lower, upper))


def _team_metrics(
    strength: float,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    """팀 강도와 작은 노이즈로 경기 전 피처 묶음을 만든다."""

    sp_fip = _bounded(4.15 - strength * 0.75 + rng.normal(0, 0.28), 2.30, 6.10)
    bat_obp = _bounded(0.325 + strength * 0.018 + rng.normal(0, 0.010), 0.260, 0.390)
    bat_slg = _bounded(0.385 + strength * 0.030 + rng.normal(0, 0.018), 0.300, 0.520)
    bat_bb_rate = _bounded(0.085 + strength * 0.008 + rng.normal(0, 0.006), 0.045, 0.140)
    bat_k_rate = _bounded(0.205 - strength * 0.010 + rng.normal(0, 0.012), 0.120, 0.290)
    rp_fip = _bounded(4.25 - strength * 0.55 + rng.normal(0, 0.25), 2.50, 6.20)
    rp_era = _bounded(4.40 - strength * 0.60 + rng.normal(0, 0.35), 2.60, 6.80)

    return {
        "sp_fip": sp_fip,
        "bat_obp": bat_obp,
        "bat_slg": bat_slg,
        "bat_iso": bat_slg - bat_obp,
        "bat_bb_rate": bat_bb_rate,
        "bat_k_rate": bat_k_rate,
        "bat_linear": _bounded(
            0.310 + strength * 0.032 + rng.normal(0, 0.014), 0.230, 0.410
        ),
        "rp_fip": rp_fip,
        "rp_era": rp_era,
        "rp_k_bb": _bounded(2.35 + strength * 0.45 + rng.normal(0, 0.30), 1.10, 4.50),
        "bullpen_candidate_count": int(rng.integers(6, 10)),
        "rp_ip_1d": _bounded(rng.normal(2.6, 1.0), 0.0, 6.0),
        "rp_np_1d": _bounded(rng.normal(43, 16), 0.0, 110.0),
        "rp_ip_3d": _bounded(rng.normal(7.5, 2.0), 1.0, 16.0),
        "rp_np_3d": _bounded(rng.normal(122, 31), 15.0, 280.0),
        "rp_ip_last10": _bounded(rng.normal(26, 5), 9.0, 48.0),
        "rp_np_last10": _bounded(rng.normal(415, 75), 120.0, 720.0),
    }


def _side_columns(side: str, metrics: dict[str, float | int]) -> dict[str, object]:
    """한 팀의 경기 전 지표를 최종 v9 열 이름으로 변환한다."""

    values: dict[str, object] = {
        f"{side}_sp_fip": metrics["sp_fip"],
        f"{side}_sp_source": "synthetic",
        f"{side}_sp_missing": False,
        f"{side}_bat_obp": metrics["bat_obp"],
        f"{side}_bat_slg": metrics["bat_slg"],
        f"{side}_bat_iso": metrics["bat_iso"],
        f"{side}_bat_bb_rate": metrics["bat_bb_rate"],
        f"{side}_bat_k_rate": metrics["bat_k_rate"],
        f"{side}_bat_linear": metrics["bat_linear"],
        f"{side}_bat_source": "synthetic",
        f"{side}_bat_missing": False,
        f"{side}_rp_fip": metrics["rp_fip"],
        f"{side}_rp_era": metrics["rp_era"],
        f"{side}_rp_k_bb": metrics["rp_k_bb"],
        f"{side}_bullpen_candidate_count": metrics["bullpen_candidate_count"],
        f"{side}_rp_source": "synthetic",
        f"{side}_rp_missing": False,
    }
    for window in ("1d", "3d", "last10"):
        values[f"{side}_rp_ip_{window}"] = metrics[f"rp_ip_{window}"]
        values[f"{side}_rp_np_{window}"] = metrics[f"rp_np_{window}"]
    return values


def build_synthetic_dataset() -> pd.DataFrame:
    """2023~2026년의 시점 유효·비무승부 합성 v9 경기 데이터를 반환한다."""

    rng = np.random.default_rng(RANDOM_SEED)
    team_strength = dict(
        zip(TEAM_CODES, np.linspace(-0.8, 0.8, len(TEAM_CODES)), strict=True)
    )
    records: list[dict[str, object]] = []

    for year in range(2023, 2027):
        dates = pd.date_range(f"{year}-04-01", periods=GAMES_PER_SEASON, freq="D")
        for game_index, game_date in enumerate(dates):
            home_team = TEAM_CODES[(game_index + year) % len(TEAM_CODES)]
            away_team = TEAM_CODES[(game_index * 2 + year + 1) % len(TEAM_CODES)]
            if home_team == away_team:
                away_team = TEAM_CODES[(game_index + 2) % len(TEAM_CODES)]

            home = _team_metrics(team_strength[home_team], rng)
            away = _team_metrics(team_strength[away_team], rng)
            advantage = (
                0.22
                + (home["bat_linear"] - away["bat_linear"]) * 7.5
                + (away["sp_fip"] - home["sp_fip"]) * 0.35
                + (away["rp_fip"] - home["rp_fip"]) * 0.18
            )
            home_win_probability = 1 / (1 + np.exp(-advantage))
            home_wins = bool(rng.random() < home_win_probability)
            lower_score = int(rng.integers(1, 7))
            score_margin = int(rng.integers(1, 5))
            home_score = lower_score + score_margin if home_wins else lower_score
            away_score = lower_score if home_wins else lower_score + score_margin

            game_datetime = game_date + pd.Timedelta(hours=9)
            cutoff_datetime = game_datetime - pd.Timedelta(hours=3)
            record: dict[str, object] = {
                "s_no": year * 1_000 + game_index + 1,
                "game_datetime": game_datetime.isoformat(),
                "year": year,
                "homeTeam": home_team,
                "awayTeam": away_team,
                "homeScore": home_score,
                "awayScore": away_score,
                "feature_cutoff_datetime": cutoff_datetime.isoformat(),
                "park_factor": _bounded(rng.normal(1.0, 0.025), 0.92, 1.08),
                "park_factor_source": "synthetic",
            }
            record.update(_side_columns("home", home))
            record.update(_side_columns("away", away))
            record.update(
                {
                    "sp_fip_diff": away["sp_fip"] - home["sp_fip"],
                    "rp_fip_diff": away["rp_fip"] - home["rp_fip"],
                    "rp_era_diff": away["rp_era"] - home["rp_era"],
                    "rp_k_bb_diff": home["rp_k_bb"] - away["rp_k_bb"],
                    "bat_obp_diff": home["bat_obp"] - away["bat_obp"],
                    "bat_slg_diff": home["bat_slg"] - away["bat_slg"],
                    "bat_iso_diff": home["bat_iso"] - away["bat_iso"],
                    "bat_bb_rate_diff": home["bat_bb_rate"] - away["bat_bb_rate"],
                    "bat_k_rate_diff": home["bat_k_rate"] - away["bat_k_rate"],
                    "bat_linear_diff": home["bat_linear"] - away["bat_linear"],
                }
            )
            records.append(record)

    return pd.DataFrame.from_records(records)


def main() -> None:
    """명령행 인자를 읽어 합성 데이터셋을 CSV로 저장한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    dataset = build_synthetic_dataset()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"합성 v9 예시 데이터 저장: {args.output} ({len(dataset):,}경기)")


if __name__ == "__main__":
    main()
