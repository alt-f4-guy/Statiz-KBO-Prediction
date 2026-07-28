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
ENV_PATH = PROJECT_ROOT / "config" / ".env"
REQUIRED_ENV_NAMES = (
    "STATIZ_API_KEY",
    "STATIZ_SECRET",
    "STATIZ_PTT_IDX",
)
DAILY_PHASES: tuple[tuple[str, str], ...] = (
    ("scripts.collect.collect_schedule", "경기 일정 수집"),
    ("scripts.collect.collect_lineups", "라인업 데이터 업데이트"),
    ("scripts.collect.collect_rosters", "로스터 정보 동기화"),
    ("scripts.ops.predict_2026", "실시간 예측 시스템 가동"),
    ("scripts.collect.collect_player_stats", "선수별 원천 스냅샷 수집"),
    ("scripts.build.process_raw_data", "원천 스냅샷 v2 정형화"),
)

console = Console()


def load_runtime_environment(env_path: Path) -> None:
    """환경파일을 읽되 이미 셸에 설정된 값은 덮어쓰지 않는다."""

    if not env_path.is_file():
        raise RuntimeError(f"환경파일을 찾을 수 없습니다: {env_path}")

    for line_number, raw_line in enumerate(
        env_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key:
            raise RuntimeError(
                f"환경파일 {line_number}행 형식이 올바르지 않습니다."
            )
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        os.environ.setdefault(key, value)

    missing = [
        name for name in REQUIRED_ENV_NAMES if not os.getenv(name, "").strip()
    ]
    if missing:
        raise RuntimeError(
            "필수 환경변수가 없습니다: " + ", ".join(missing)
        )


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
    load_runtime_environment(ENV_PATH)
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]⚾ KBO Daily Prediction Pipeline v1.2[/bold cyan]\n"
        f"[dim]Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
        border_style="cyan",
        padding=(1, 2)
    ))

    summary_table = Table(title="\n[bold]Pipeline Execution Summary[/bold]", show_header=True, header_style="bold magenta")
    summary_table.add_column("Phase", style="dim", width=5)
    summary_table.add_column("Script", style="cyan")
    summary_table.add_column("Description")
    summary_table.add_column("Status", justify="center")
    summary_table.add_column("Duration", justify="right")

    pipeline_start = time.time()
    
    for i, (script, desc) in enumerate(DAILY_PHASES):
        success, duration = run_script(script, desc)
        
        status_str = "[bold green]SUCCESS[/bold green]" if success else "[bold red]FAILED[/bold red]"
        dur_str = f"{duration:.1f}s" if success else "-"
        summary_table.add_row(str(i+1), script, desc, status_str, dur_str)
        
        if not success:
            console.print(f"\n[bold red]🛑 {script} 단계에서 중단되었습니다.[/bold red]")
            console.print(summary_table)
            return 1
        
    # 파이프라인 완료 후 요약 출력 (정상 종료 시)
    console.print(summary_table)
    total_duration = time.time() - pipeline_start
    console.print(Panel(
        f"[bold green]🎉 Pipeline Task Completed![/bold green]\n"
        f"[dim]Total Elapsed: {total_duration/60:.1f} minutes[/dim]",
        border_style="green",
        expand=False
    ))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
