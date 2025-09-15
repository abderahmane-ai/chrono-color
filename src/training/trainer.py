import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel

from config import config
from modeling import create_models
from .losses import GANLoss, calculate_psnr


class ChronoColorTrainer:
    """Trainer class for ChronoColor model"""

    def __init__(self):
        self.console = Console()
        self.writer = SummaryWriter(config.LOG_DIR / config.EXPERIMENT_NAME)

        # Create models
        self.generator, self.discriminator = create_models(config.DEVICE)

        # Define loss functions
        self.criterion_gan = GANLoss(gan_mode='vanilla').to(config.DEVICE)
        self.criterion_l1 = nn.L1Loss().to(config.DEVICE)

        # Define optimizers with different learning rates
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.LR_G,
            betas=(config.BETA1, config.BETA2)
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.LR_D,
            betas=(config.BETA1, config.BETA2)
        )

        self.scheduler_G = torch.optim.lr_scheduler.StepLR(
            self.optimizer_G, step_size=50, gamma=0.7
        )
        self.scheduler_D = torch.optim.lr_scheduler.StepLR(
            self.optimizer_D, step_size=50, gamma=0.7
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_psnr = 0

        # Create checkpoints directory
        self.checkpoint_dir = config.CHECKPOINT_DIR / config.EXPERIMENT_NAME
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.console.print(Panel.fit(
            f"[bold green]ChronoColor Training Setup[/bold green]\n"
            f"Experiment: {config.EXPERIMENT_NAME}\n"
            f"Device: {config.DEVICE}\n"
            f"Generator Parameters: {sum(p.numel() for p in self.generator.parameters()):,}\n"
            f"Discriminator Parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}",
            title="Training Configuration"
        ))

    def train(self, train_loader, val_loader, num_epochs):
        """Main training loop"""
        with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=self.console
        ) as progress:
            # Create tasks for progress tracking
            epoch_task = progress.add_task("[red]Epochs...", total=num_epochs)
            batch_task = progress.add_task("[green]Batches...", total=len(train_loader))

            for epoch in range(self.current_epoch, num_epochs):
                # Update current epoch for training logic
                self.current_epoch = epoch
                
                # Update epoch progress
                progress.update(epoch_task, advance=1, description=f"[red]Epoch {epoch + 1}/{num_epochs}")

                # Train for one epoch
                self.train_epoch(train_loader, progress, batch_task)

                # Validate
                val_metrics = self.validate(val_loader)

                # Update learning rates
                self.scheduler_G.step()
                self.scheduler_D.step()

                # Log metrics
                self.log_metrics(epoch, val_metrics)

                # Save checkpoint
                if (epoch + 1) % config.SAVE_INTERVAL == 0:
                    self.save_checkpoint(epoch, val_metrics)

                # Reset batch task for next epoch
                progress.reset(batch_task, description="[green]Batches...", total=len(train_loader))
        
        # Training completed - show final summary
        self.show_training_summary()

    def train_epoch(self, train_loader, progress, batch_task):
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()

        for batch_idx, (L, ab, decades) in enumerate(train_loader):
            # Move data to device
            L = L.to(config.DEVICE)
            ab = ab.to(config.DEVICE)

            # Generate fake images
            fake_ab = self.generator(L)
            
            # Debug: Print value ranges occasionally
            if batch_idx == 0 and self.current_epoch % 5 == 0:
                self.console.print(f"[dim]Debug - L range: [{L.min():.3f}, {L.max():.3f}]")
                self.console.print(f"[dim]Debug - ab target range: [{ab.min():.3f}, {ab.max():.3f}]")
                self.console.print(f"[dim]Debug - ab generated range: [{fake_ab.min():.3f}, {fake_ab.max():.3f}]")
            
            # Train discriminator (skip during warmup period)
            if self.current_epoch >= config.PRETRAIN_EPOCHS:
                self.optimizer_D.zero_grad()

                # Real images
                real_pred = self.discriminator(L, ab)
                loss_D_real = self.criterion_gan(real_pred, True)

                # Fake images
                fake_pred = self.discriminator(L, fake_ab.detach())
                loss_D_fake = self.criterion_gan(fake_pred, False)

                # Combined discriminator loss
                loss_D = (loss_D_real + loss_D_fake) * 0.5
                
                # Check for NaN values
                if torch.isnan(loss_D) or torch.isinf(loss_D):
                    self.console.print(f"[red]Warning: Invalid discriminator loss detected: {loss_D.item()}")
                    continue
                    
                loss_D.backward()
                
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=config.GRAD_CLIP_D)
                
                self.optimizer_D.step()
            else:
                # During warmup, set discriminator loss to 0 for logging
                loss_D = torch.tensor(0.0, device=config.DEVICE)

            # Train generator
            self.optimizer_G.zero_grad()

            # L1 loss (always computed)
            loss_G_l1 = self.criterion_l1(fake_ab, ab) * config.L1_LAMBDA

            # GAN loss (only after warmup period)
            if self.current_epoch >= config.PRETRAIN_EPOCHS:
                fake_pred = self.discriminator(L, fake_ab)
                loss_G_gan = self.criterion_gan(fake_pred, True) * config.ADVERSARIAL_LAMBDA
            else:
                # During warmup, focus only on L1 loss
                loss_G_gan = torch.tensor(0.0, device=config.DEVICE)

            # Combined generator loss
            loss_G = loss_G_gan + loss_G_l1
            
            # Check for NaN values
            if torch.isnan(loss_G) or torch.isinf(loss_G):
                self.console.print(f"[red]Warning: Invalid generator loss detected: {loss_G.item()}")
                continue
                
            loss_G.backward()
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=config.GRAD_CLIP_G)
            
            self.optimizer_G.step()

            # Update progress
            progress.update(batch_task, advance=1,
                            description=f"[green]Batch {batch_idx + 1}/{len(train_loader)} "
                                        f"(D: {loss_D.item():.3f}, G: {loss_G.item():.3f})")

            # Log batch metrics
            if self.global_step % config.LOG_INTERVAL == 0:
                self.writer.add_scalar('Loss/D', loss_D.item(), self.global_step)
                self.writer.add_scalar('Loss/G', loss_G.item(), self.global_step)
                self.writer.add_scalar('Loss/G_GAN', loss_G_gan.item(), self.global_step)
                self.writer.add_scalar('Loss/G_L1', loss_G_l1.item(), self.global_step)

            self.global_step += 1

    def validate(self, val_loader):
        """Validate the model"""
        self.generator.eval()
        self.discriminator.eval()

        total_loss_D = 0
        total_loss_G = 0
        total_psnr = 0
        total_samples = 0

        with torch.no_grad():
            for L, ab, decades in val_loader:
                L = L.to(config.DEVICE)
                ab = ab.to(config.DEVICE)

                # Forward pass
                fake_ab = self.generator(L)
                real_pred = self.discriminator(L, ab)
                fake_pred = self.discriminator(L, fake_ab)

                # Calculate losses
                loss_D_real = self.criterion_gan(real_pred, True)
                loss_D_fake = self.criterion_gan(fake_pred, False)
                loss_D = (loss_D_real + loss_D_fake) * 0.5

                loss_G_gan = self.criterion_gan(fake_pred, True)
                loss_G_l1 = self.criterion_l1(fake_ab, ab) * config.L1_LAMBDA
                loss_G = loss_G_gan + loss_G_l1

                # Calculate PSNR
                psnr = calculate_psnr(fake_ab, ab)

                # Accumulate metrics
                batch_size = L.size(0)
                total_loss_D += loss_D.item() * batch_size
                total_loss_G += loss_G.item() * batch_size
                total_psnr += psnr.item() * batch_size
                total_samples += batch_size

        # Average metrics
        avg_loss_D = total_loss_D / total_samples
        avg_loss_G = total_loss_G / total_samples
        avg_psnr = total_psnr / total_samples

        return {
            'loss_D': avg_loss_D,
            'loss_G': avg_loss_G,
            'psnr': avg_psnr
        }

    def log_metrics(self, epoch, metrics):
        """Log metrics to console and TensorBoard"""
        # Log to TensorBoard
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(f'Val/{metric_name}', metric_value, epoch)

        # Log to console
        table = Table(title=f"Epoch {epoch + 1} Validation Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        for metric_name, metric_value in metrics.items():
            table.add_row(metric_name, f"{metric_value:.4f}")

        self.console.print(table)

        # Update best PSNR
        if metrics['psnr'] > self.best_psnr:
            self.best_psnr = metrics['psnr']
            self.save_checkpoint(epoch, metrics, is_best=True)
            self.console.print(f"[bold green]🏆 New best model! PSNR: {metrics['psnr']:.4f} (epoch {epoch + 1})[/bold green]")

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
            'best_psnr': self.best_psnr,
            'metrics': metrics,
            'config': {k: v for k, v in vars(config).items() if not k.startswith('_')}
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.console.print(f"[green]Best model saved with PSNR: {metrics['psnr']:.4f}")

    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)

        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_psnr = checkpoint['best_psnr']

        self.console.print(f"[green]Loaded checkpoint from epoch {self.current_epoch}")
    
    def show_training_summary(self):
        """Show final training summary"""
        best_model_path = self.checkpoint_dir / "best_model.pth"
        
        summary_panel = Panel.fit(
            f"[bold green]🎉 Training Completed Successfully![/bold green]\n\n"
            f"[cyan]Best PSNR achieved:[/cyan] [bold yellow]{self.best_psnr:.4f}[/bold yellow]\n"
            f"[cyan]Total epochs:[/cyan] {self.current_epoch + 1}\n"
            f"[cyan]Total steps:[/cyan] {self.global_step:,}\n"
            f"[cyan]Best model saved at:[/cyan] {best_model_path}\n\n"
            f"[dim]The best model contains both generator and discriminator weights,\n"
            f"along with optimizer states for potential resumption.[/dim]",
            title="Training Summary"
        )
        
        self.console.print(summary_panel)