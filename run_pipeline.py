import subprocess
import sys
import time
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def run_script(script_name, description):
    # 정적인 구분선 출력
    console.rule(f"[bold white]{script_name}[/bold white]")
    console.print(f"[bold yellow]▶ {description} 시작...[/bold yellow]")
    
    start_time = time.time()
    try:
        # 자식 프로세스가 터미널 UI를 자유롭게 사용할 수 있도록 실행
        result = subprocess.run([sys.executable, script_name], check=True)
        duration = time.time() - start_time
        console.print(f"[bold green]✔ {script_name} 완료[/bold green] ({duration:.1f}초)\n")
        return True, duration
    except subprocess.CalledProcessError:
        console.print(f"[bold red]✘ {script_name} 실행 실패[/bold red]\n")
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
        ("1.collect_schedule.py", "경기 일정 수집"),
        ("2.collect_lineups.py", "라인업 데이터 업데이트"),
        ("3.collect_rosters.py", "로스터 정보 동기화"),
        ("4.collect_player_stats.py", "선수별 세부 스탯 수집"),
        ("5.process_raw_data.py", "원천 데이터 전처리"),
        ("create_feature_matrix_v7.py", "피처 매트릭스 생성"),
        ("predict_2026.py", "실시간 예측 시스템 가동")
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
        if script == "predict_2026.py":
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