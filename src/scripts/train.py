"""
Main training script for ChronoColor project.
"""
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rich.console import Console
from rich.panel import Panel

from config import config
from data_loading.datasets import get_dataloaders
from training.trainer import ChronoColorTrainer


def main():
    console = Console()

    console.print(Panel.fit(
        "[bold blue]ChronoColor Training[/bold blue]\n"
        "Automatic Colorization of Historical Photographs",
        title="Welcome"
    ))

    # Load data
    console.print("[yellow]Loading data...[/yellow]")
    train_loader, val_loader, test_loader = get_dataloaders()

    console.print(f"[green]Training samples: {len(train_loader.dataset)}")
    console.print(f"[green]Validation samples: {len(val_loader.dataset)}")
    console.print(f"[green]Test samples: {len(test_loader.dataset)}")

    # Create trainer
    trainer = ChronoColorTrainer()

    # Start training
    console.print("[yellow]Starting training...[/yellow]")
    trainer.train(train_loader, val_loader, config.NUM_EPOCHS)

    console.print(Panel.fit(
        "[bold green]Training completed successfully![/bold green]",
        title="Done"
    ))


if __name__ == "__main__":
    main()