import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


PROJECT_ROOT = Path(__file__).resolve().parent
CORE_SOURCE_DIR = PROJECT_ROOT / "src"
CORE_MODULE_DIR = CORE_SOURCE_DIR / "kbo_pipeline"

console = Console()

def run_script(module_name, description):
    # 정적인 구분선 출력
    console.rule(f"[bold white]{module_name}[/bold white]")
    console.print(f"[bold yellow]▶ {description} 시작...[/bold yellow]")
    
    start_time = time.time()
    try:
        # 자식 프로세스가 터미널 UI를 자유롭게 사용할 수 있도록 실행
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            item
            for item in (
                str(CORE_MODULE_DIR),
                str(CORE_SOURCE_DIR),
                existing_pythonpath,
            )
            if item
        )
        subprocess.run(
            [sys.executable, "-m", module_name],
            check=True,
            cwd=PROJECT_ROOT,
            env=environment,
        )
        duration = time.time() - start_time
        console.print(f"[bold green]✔ {module_name} 완료[/bold green] ({duration:.1f}초)\n")
        return True, duration
    except subprocess.CalledProcessError:
        console.print(f"[bold red]✘ {module_name} 실행 실패[/bold red]\n")
        return False, 0

def main():
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]⚾ KBO Daily Prediction Pipeline v1.2[/bold cyan]\n"
        f"[dim]Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
        border_style="cyan",
        padding=(1, 2)
    ))

    phases = [
        ("scripts.collect.collect_schedule", "경기 일정 수집"),
        ("scripts.collect.collect_lineups", "라인업 데이터 업데이트"),
        ("scripts.collect.collect_rosters", "로스터 정보 동기화"),
        ("scripts.collect.collect_player_stats", "선수별 원천 스냅샷 수집"),
        ("scripts.build.process_raw_data", "원천 스냅샷 v2 정형화"),
        ("scripts.build.create_feature_matrix_v9", "시점 기준 v9 피처 생성"),
        ("scripts.model.tune_hyperparameters", "2025년 순차 검증 분류기 튜닝"),
        ("scripts.model.train_classifier", "직접 승패 분류기 학습·보정"),
        ("scripts.model.train_score_models", "득점 분포 비교 모델 학습·보정"),
        ("scripts.model.compare_models", "공통 경기 확률 지표 비교"),
        ("scripts.ops.evaluate_fallback_recent10", "최근 10경기 대체 모델 백테스트"),
        ("scripts.model.backtest", "고정 운영 모델 2026년 평가 재현"),
        ("scripts.ops.predict_2026", "실시간 예측 시스템 가동")
    ]

    summary_table = Table(title="\n[bold]Pipeline Execution Summary[/bold]", show_header=True, header_style="bold magenta")
    summary_table.add_column("Phase", style="dim", width=5)
    summary_table.add_column("Script", style="cyan")
    summary_table.add_column("Description")
    summary_table.add_column("Status", justify="center")
    summary_table.add_column("Duration", justify="right")

    pipeline_start = time.time()
    
    for i, (script, desc) in enumerate(phases):
        success, duration = run_script(script, desc)
        
        status_str = "[bold green]SUCCESS[/bold green]" if success else "[bold red]FAILED[/bold red]"
        dur_str = f"{duration:.1f}s" if success else "-"
        summary_table.add_row(str(i+1), script, desc, status_str, dur_str)
        
        if not success:
            console.print(f"\n[bold red]🛑 {script} 단계에서 중단되었습니다.[/bold red]")
            break
        
        # 마지막 예측 시스템은 무한 루프이므로 요약 테이블을 먼저 보여줄 수 없음
        # 따라서 파이프라인이 중간에 멈췄을 때만 아래 테이블이 나옴
        if script == "scripts.ops.predict_2026":
            return

    # 파이프라인 완료 후 요약 출력 (정상 종료 시)
    console.print(summary_table)
    total_duration = time.time() - pipeline_start
    console.print(Panel(
        f"[bold green]🎉 Pipeline Task Completed![/bold green]\n"
        f"[dim]Total Elapsed: {total_duration/60:.1f} minutes[/dim]",
        border_style="green",
        expand=False
    ))

if __name__ == "__main__":
    main()
