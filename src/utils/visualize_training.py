"""
Utility to visualize training progress using TensorBoard.
"""
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from src.config import config
from rich.console import Console


def main():
    console = Console()

    # Find the latest experiment
    log_dir = config.LOG_DIR
    experiments = sorted(log_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)

    if not experiments:
        console.print("[red]No experiments found![/red]")
        return

    latest_experiment = experiments[0]
    console.print(f"[green]Opening TensorBoard for: {latest_experiment.name}[/green]")

    # Launch TensorBoard
    import subprocess
    subprocess.run(["tensorboard", "--logdir", str(log_dir), "--port", "6006"])


if __name__ == "__main__":
    main()