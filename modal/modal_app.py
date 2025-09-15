"""
Complete Modal deployment for ChronoColor with source code mounting
"""
import modal

app = modal.App("chronocolor-complete")

# Create the image with dependencies and mount source code
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install([
        "datasets>=4.0.0",
        "matplotlib>=3.10.6",
        "mlcroissant>=1.0.22",
        "numpy>=2.3.3",
        "opencv-python>=4.11.0.86",
        "pillow>=11.3.0",
        "rich>=14.1.0",
        "scikit-image>=0.25.2",
        "sgmllib3k>=1.0.0",
        "tensorboard>=2.20.0",
        "torch>=2.8.0",
        "torchaudio>=2.8.0",
        "torchmetrics>=1.8.2",
        "torchvision>=0.23.0",
        "tqdm>=4.67.1",
        "transforms>=0.2.1",
    ])
    .apt_install(["git", "wget", "curl", "libgl1-mesa-glx", "libglib2.0-0"])
    .run_commands([
        "mkdir -p /app/logs",
        "mkdir -p /app/checkpoints",
        "mkdir -p /app/data/processed",
        "mkdir -p /app/data/raw"
    ])
    .add_local_dir("src", remote_path="/app/src")
)

# Persistent volume for outputs
volume = modal.Volume.from_name("chronocolor-outputs", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/shared": volume},
    timeout=3600,  # 1 hour for data download
    memory=8192,   # 8GB RAM
    cpu=4,
)
def download_and_process_data(force_redownload: bool = False):
    """Download and process the historical color dataset"""
    import os
    import sys
    import subprocess
    import tarfile
    from pathlib import Path
    import shutil
    
    # Set working directory
    os.chdir("/app")
    sys.path.insert(0, "/app/src")
    
    from rich.console import Console
    from rich.panel import Panel
    
    console = Console()
    console.print(Panel.fit(
        "[bold blue]ChronoColor Data Download & Processing[/bold blue]\n"
        "Downloading Historical Color Dataset",
        title="Data Pipeline"
    ))
    
    try:
        # Create data directories in shared volume
        os.makedirs("/shared/data/raw", exist_ok=True)
        os.makedirs("/shared/data/processed", exist_ok=True)
        
        # Check if data already exists and is complete
        if not force_redownload:
            console.print("[yellow]Checking if data already exists...[/yellow]")
            raw_dir = Path("/shared/data/raw")
            total_images = 0
            all_decades_exist = True
            
            for decade in ["1930s", "1940s", "1950s", "1960s", "1970s"]:
                decade_path = raw_dir / decade
                if decade_path.exists():
                    image_count = len([f for f in decade_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
                    total_images += image_count
                    console.print(f"[green]✓ {decade}: {image_count} images[/green]")
                else:
                    console.print(f"[red]❌ Missing {decade} folder[/red]")
                    all_decades_exist = False
            
            if all_decades_exist and total_images > 1000:  # Expect around 1326 images
                console.print(f"[green]✓ Data already exists with {total_images} images. Skipping download.[/green]")
                
                # Check if processed data exists
                processed_dir = Path("/shared/data/processed")
                if processed_dir.exists() and any(processed_dir.iterdir()):
                    console.print("[green]✓ Processed data already exists. Skipping processing.[/green]")
                    return {
                        "status": "success",
                        "message": f"Data already exists with {total_images} images. Skipped download and processing.",
                        "images_found": total_images
                    }
                else:
                    console.print("[yellow]Processed data missing. Will process existing raw data.[/yellow]")
                    # Skip to processing step
                    console.print("[yellow]Processing images...[/yellow]")
                    
                    # Update config paths for image processing
                    from config import config
                    config.RAW_DATA_DIR = Path("/shared/data/raw")
                    config.PROCESSED_DATA_DIR = Path("/shared/data/processed")
                    
                    # Import and run the image processing
                    from scripts.process_images import create_processed_dataset
                    create_processed_dataset()
                    
                    console.print("[green]✓ Image processing completed[/green]")
                    
                    return {
                        "status": "success",
                        "message": f"Used existing data with {total_images} images. Completed processing.",
                        "images_found": total_images
                    }
            else:
                console.print(f"[yellow]Data incomplete or missing (found {total_images} images). Proceeding with download.[/yellow]")
        
        # Download the dataset
        dataset_url = "https://graphics.cs.cmu.edu/projects/historicalColor/HistoricalColor-ECCV2012-DecadeDatabase.tar"
        console.print(f"[yellow]Downloading dataset from: {dataset_url}[/yellow]")
        
        subprocess.run([
            "wget", "--no-check-certificate", "-O", "/shared/data/raw/dataset.tar", dataset_url
        ], check=True)
        
        console.print("[green]✓ Dataset downloaded successfully[/green]")
        
        # Extract the tar file
        console.print("[yellow]Extracting dataset...[/yellow]")
        with tarfile.open("/shared/data/raw/dataset.tar", "r") as tar:
            tar.extractall("/shared/data/raw/")
        
        # Find and extract decade folders from the tar archive
        raw_dir = Path("/shared/data/raw")
        
        # List all extracted contents to debug
        console.print("[yellow]Listing extracted contents:[/yellow]")
        for item in raw_dir.iterdir():
            console.print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
        
        # Navigate to the correct path: HistoricalColor-ECCV2012/data/imgs/decade_database/
        historical_color_path = raw_dir / "HistoricalColor-ECCV2012"
        data_path = historical_color_path / "data"
        imgs_path = data_path / "imgs"
        decade_db_path = imgs_path / "decade_database"
        
        console.print(f"[yellow]Looking for decade database at: {decade_db_path}[/yellow]")
        
        if decade_db_path.exists():
            console.print(f"[green]✓ Found decade database at: {decade_db_path}[/green]")
            
            # List contents of decade_database folder
            console.print("[yellow]Contents of decade_database folder:[/yellow]")
            for item in decade_db_path.iterdir():
                if item.is_dir():
                    image_count = len([f for f in item.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
                    console.print(f"  - {item.name} (dir, {image_count} images)")
                else:
                    console.print(f"  - {item.name} (file)")
            
            # Move decade folders from decade_database to raw directory
            target_dir = Path("/shared/data/raw")
            moved_folders = []
            
            for decade_folder in decade_db_path.iterdir():
                if decade_folder.is_dir() and any(decade in decade_folder.name for decade in ["1930s", "1940s", "1950s", "1960s", "1970s"]):
                    target_path = target_dir / decade_folder.name
                    if target_path.exists():
                        shutil.rmtree(target_path)
                    shutil.move(str(decade_folder), str(target_path))
                    moved_folders.append(decade_folder.name)
                    
                    # Count images in moved folder
                    image_count = len([f for f in target_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
                    console.print(f"[green]✓ Moved {decade_folder.name} to data/raw/ ({image_count} images)[/green]")
            
            console.print(f"[green]✓ Successfully moved {len(moved_folders)} decade folders[/green]")
            
            # Clean up the original database folder
            if moved_folders:
                shutil.rmtree(historical_color_path)
                console.print(f"[green]✓ Cleaned up original database folder[/green]")
        else:
            console.print("[red]❌ Could not find expected dataset structure[/red]")
            console.print(f"[red]Expected: {historical_color_path} and {data_path}[/red]")
            
            # Recursive search for decade folders
            console.print("[yellow]Searching recursively for decade folders...[/yellow]")
            found_decades = []
            
            def search_for_decades(search_path, depth=0):
                if depth > 3:  # Limit recursion depth
                    return
                try:
                    for item in search_path.iterdir():
                        if item.is_dir():
                            if any(decade in item.name for decade in ["1930s", "1940s", "1950s", "1960s", "1970s"]):
                                found_decades.append(item)
                                console.print(f"[green]Found decade folder: {item}[/green]")
                            else:
                                search_for_decades(item, depth + 1)
                except PermissionError:
                    pass
            
            search_for_decades(raw_dir)
            
            # Move any found decade folders
            if found_decades:
                target_dir = Path("/app/data/raw")
                moved_folders = []
                for decade_folder in found_decades:
                    target_path = target_dir / decade_folder.name
                    if target_path.exists():
                        shutil.rmtree(target_path)
                    shutil.move(str(decade_folder), str(target_path))
                    moved_folders.append(decade_folder.name)
                    console.print(f"[green]✓ Moved {decade_folder.name} to data/raw/[/green]")
                console.print(f"[green]✓ Successfully moved {len(moved_folders)} decade folders[/green]")
        
        # Remove the unwanted file from 1940s folder
        console.print("[yellow]Cleaning up 1940s folder...[/yellow]")
        folder_1940s = Path("/shared/data/raw/1940s")
        if folder_1940s.exists():
            for file in folder_1940s.iterdir():
                if file.name.startswith("_"):
                    file.unlink()
                    console.print(f"[green]✓ Removed {file.name} from 1940s folder[/green]")
        
        # Clean up extracted tar file
        os.remove("/shared/data/raw/dataset.tar")
        
        console.print("[green]✓ Dataset extraction and cleanup completed[/green]")
        
        # Verify decade folders exist before processing
        console.print("[yellow]Verifying decade folders...[/yellow]")
        raw_dir = Path("/shared/data/raw")
        found_decades = []
        for decade in ["1930s", "1940s", "1950s", "1960s", "1970s"]:
            decade_path = raw_dir / decade
            if decade_path.exists():
                image_count = len([f for f in decade_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
                found_decades.append(decade)
                console.print(f"[green]✓ Found {decade} with {image_count} images[/green]")
            else:
                console.print(f"[red]❌ Missing {decade} folder[/red]")
        
        if not found_decades:
            raise Exception("No decade folders found! Dataset extraction may have failed.")
        
        # Run the process images script
        console.print("[yellow]Processing images...[/yellow]")
        
        # Update config paths for image processing
        from config import config
        config.RAW_DATA_DIR = Path("/shared/data/raw")
        config.PROCESSED_DATA_DIR = Path("/shared/data/processed")
        
        # Import and run the image processing
        from scripts.process_images import create_processed_dataset
        create_processed_dataset()
        
        console.print("[green]✓ Image processing completed[/green]")
        
        return {
            "status": "success",
            "message": "Data download and processing completed successfully",
            "images_found": total_images
        }
        
    except Exception as e:
        console.print(Panel.fit(
            f"[bold red]❌ Data processing failed: {str(e)}[/bold red]",
            title="Error"
        ))
        raise


@app.function(
    image=image,
    volumes={"/shared": volume},
    timeout=300,  # 5 minutes
    memory=2048,   # 2GB RAM
    cpu=2,
)
def verify_data():
    """Verify that data exists and is accessible"""
    import sys
    import os
    from pathlib import Path
    
    # Set working directory
    os.chdir("/app")
    sys.path.insert(0, "/app/src")
    
    from rich.console import Console
    console = Console()
    
    # Check data availability
    raw_dir = Path("/shared/data/raw")
    total_images = 0
    missing_decades = []
    
    for decade in ["1930s", "1940s", "1950s", "1960s", "1970s"]:
        decade_path = raw_dir / decade
        if decade_path.exists():
            image_count = len([f for f in decade_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            total_images += image_count
            console.print(f"[green]✓ {decade}: {image_count} images[/green]")
        else:
            console.print(f"[red]❌ Missing {decade} folder[/red]")
            missing_decades.append(decade)
    
    if missing_decades:
        raise Exception(f"Missing decade folders: {missing_decades}. Please run data download first.")
    
    if total_images == 0:
        raise Exception(f"No images found in {raw_dir}! Please run data download first.")
    
    console.print(f"[green]✓ Data verification passed: {total_images} images found[/green]")
    
    return {
        "status": "success",
        "total_images": total_images,
        "message": f"Data verification passed with {total_images} images"
    }


@app.function(
    image=image,
    gpu="H100",
    volumes={"/shared": volume},
    timeout=86400,  # 24 hours
    memory=32768,   # 32GB RAM
    cpu=8,
)
def run_training():
    """Run the complete ChronoColor training pipeline"""
    import sys
    import os
    from pathlib import Path
    import shutil
    
    # Set working directory
    os.chdir("/app")
    sys.path.insert(0, "/app/src")
    
    # Copy outputs to persistent volume
    output_dirs = ["/app/logs", "/app/checkpoints"]
    for output_dir in output_dirs:
        if Path(output_dir).exists():
            dest_dir = f"/shared/outputs/{Path(output_dir).name}"
            os.makedirs("/shared/outputs", exist_ok=True)
            if Path(dest_dir).exists():
                shutil.rmtree(dest_dir)
            shutil.copytree(output_dir, dest_dir)
    
    try:
        # Import after setting up paths
        from rich.console import Console
        from rich.panel import Panel
        
        console = Console()
        console.print(Panel.fit(
            "[bold blue]ChronoColor Training on Modal H100[/bold blue]\n"
            "Automatic Colorization of Historical Photographs",
            title="Modal H100 Deployment"
        ))
        
        # Import your modules
        from config import config
        
        # Update paths for Modal environment BEFORE importing other modules
        config.PROJECT_ROOT = Path("/app")
        config.DATA_DIR = Path("/shared/data")
        config.RAW_DATA_DIR = config.DATA_DIR / "raw"
        config.PROCESSED_DATA_DIR = config.DATA_DIR / "processed"
        config.LOG_DIR = Path("/app/logs")
        config.CHECKPOINT_DIR = Path("/app/checkpoints")
        
        # Now import modules that depend on config
        from data_loading.datasets import get_dataloaders
        from training.trainer import ChronoColorTrainer
        
        # Ensure directories exist
        for dir_path in [config.LOG_DIR, config.CHECKPOINT_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        console.print(f"[green]Device: {config.DEVICE}")
        console.print(f"[green]Project root: {config.PROJECT_ROOT}")
        console.print(f"[green]Data directory: {config.DATA_DIR}")
        
        # Verify data exists before loading
        console.print("[yellow]Verifying data availability...[/yellow]")
        raw_dir = config.RAW_DATA_DIR
        total_images = 0
        for decade in ["1930s", "1940s", "1950s", "1960s", "1970s"]:
            decade_path = raw_dir / decade
            if decade_path.exists():
                image_count = len([f for f in decade_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
                total_images += image_count
                console.print(f"[green]✓ {decade}: {image_count} images[/green]")
            else:
                console.print(f"[red]❌ Missing {decade} folder[/red]")
        
        if total_images == 0:
            raise Exception(f"No images found in {raw_dir}! Please check data processing.")
        
        console.print(f"[green]✓ Total images available: {total_images}[/green]")
        
        # Load data with explicit data directory
        console.print("[yellow]Loading datasets...[/yellow]")
        console.print(f"[blue]Using data directory: {config.RAW_DATA_DIR}[/blue]")
        
        # Double-check data exists before creating dataloaders
        if not config.RAW_DATA_DIR.exists():
            raise Exception(f"Data directory does not exist: {config.RAW_DATA_DIR}")
        
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir=config.RAW_DATA_DIR,
            batch_size=config.BATCH_SIZE
        )
        
        console.print(f"[green]✓ Training samples: {len(train_loader.dataset)}")
        console.print(f"[green]✓ Validation samples: {len(val_loader.dataset)}")
        console.print(f"[green]✓ Test samples: {len(test_loader.dataset)}")
        
        # Initialize trainer
        console.print("[yellow]Initializing trainer...[/yellow]")
        trainer = ChronoColorTrainer()
        
        # Start training
        console.print(Panel.fit(
            f"[bold yellow]Starting training for {config.NUM_EPOCHS} epochs[/bold yellow]\n"
            f"Batch size: {config.BATCH_SIZE}\n"
            f"Generator LR: {config.LR_G}, Discriminator LR: {config.LR_D}\n"
            f"Device: {config.DEVICE}",
            title="Training Configuration"
        ))
        
        trainer.train(train_loader, val_loader, config.NUM_EPOCHS)
        
        # Copy final outputs to persistent volume
        os.makedirs("/shared/outputs", exist_ok=True)
        for output_dir in output_dirs:
            if Path(output_dir).exists():
                dest_dir = f"/shared/outputs/{Path(output_dir).name}"
                if Path(dest_dir).exists():
                    shutil.rmtree(dest_dir)
                shutil.copytree(output_dir, dest_dir)
                console.print(f"[green]✓ Copied {output_dir} to {dest_dir}[/green]")
        
        # Ensure the best model is prominently saved
        best_model_path = config.CHECKPOINT_DIR / config.EXPERIMENT_NAME / "best_model.pth"
        if best_model_path.exists():
            # Copy best model to a prominent location in shared storage
            best_model_dest = Path("/shared/outputs/best_model.pth")
            shutil.copy2(best_model_path, best_model_dest)
            console.print(f"[bold green]✓ Best model saved to: {best_model_dest}[/bold green]")
            
            # Also save just the generator weights for easier inference
            checkpoint = torch.load(best_model_path, map_location='cpu')
            generator_only = {
                'generator_state_dict': checkpoint['generator_state_dict'],
                'best_psnr': checkpoint['best_psnr'],
                'epoch': checkpoint['epoch'],
                'config': checkpoint['config']
            }
            generator_dest = Path("/shared/outputs/best_generator.pth")
            torch.save(generator_only, generator_dest)
            console.print(f"[bold green]✓ Best generator weights saved to: {generator_dest}[/bold green]")
            console.print(f"[bold cyan]Best PSNR achieved: {checkpoint['best_psnr']:.4f}[/bold cyan]")
        else:
            console.print("[yellow]Warning: No best model found, this might indicate training issues[/yellow]")
        
        # Commit volume changes
        volume.commit()
        
        console.print(Panel.fit(
            "[bold green]🎉 Training completed successfully![/bold green]\n"
            "Outputs saved to persistent volume",
            title="Success"
        ))
        
        return {
            "status": "success",
            "message": "Training completed successfully",
            "epochs": config.NUM_EPOCHS,
            "experiment": config.EXPERIMENT_NAME
        }
        
    except Exception as e:
        console.print(Panel.fit(
            f"[bold red]❌ Training failed: {str(e)}[/bold red]",
            title="Error"
        ))
        raise


@app.function(
    image=image,
    volumes={"/shared": volume},
)
def get_best_model_info():
    """Get information about the best saved model"""
    import torch
    from pathlib import Path
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    best_model_path = Path("/shared/outputs/best_model.pth")
    best_generator_path = Path("/shared/outputs/best_generator.pth")
    
    if not best_model_path.exists():
        return {
            "status": "error",
            "message": "No best model found. Training may not have completed successfully."
        }
    
    try:
        # Load model info
        checkpoint = torch.load(best_model_path, map_location='cpu')
        
        # Create info table
        table = Table(title="Best Model Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Best PSNR", f"{checkpoint['best_psnr']:.4f}")
        table.add_row("Epoch", str(checkpoint['epoch'] + 1))
        table.add_row("Global Step", str(checkpoint['global_step']))
        table.add_row("Experiment", checkpoint['config'].get('EXPERIMENT_NAME', 'Unknown'))
        
        if 'metrics' in checkpoint:
            for metric, value in checkpoint['metrics'].items():
                table.add_row(f"Final {metric}", f"{value:.4f}")
        
        console.print(table)
        
        return {
            "status": "success",
            "best_psnr": checkpoint['best_psnr'],
            "epoch": checkpoint['epoch'] + 1,
            "global_step": checkpoint['global_step'],
            "experiment_name": checkpoint['config'].get('EXPERIMENT_NAME', 'Unknown'),
            "full_model_path": str(best_model_path),
            "generator_only_path": str(best_generator_path),
            "model_exists": True,
            "generator_only_exists": best_generator_path.exists()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error loading model info: {str(e)}"
        }


@app.function(
    image=image,
    volumes={"/shared": volume},
)
def download_outputs(local_path: str = "./modal_outputs"):
    """Download training outputs from Modal volume to local machine"""
    import shutil
    from pathlib import Path
    
    local_path = Path(local_path)
    local_path.mkdir(exist_ok=True)
    
    # Copy outputs from volume
    output_dirs = ["/shared/outputs/logs", "/shared/outputs/checkpoints"]
    for output_dir in output_dirs:
        if Path(output_dir).exists():
            dest_dir = local_path / Path(output_dir).name
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.copytree(output_dir, dest_dir)
    
    # Copy the best model files
    best_files = ["/shared/outputs/best_model.pth", "/shared/outputs/best_generator.pth"]
    for best_file in best_files:
        if Path(best_file).exists():
            dest_file = local_path / Path(best_file).name
            shutil.copy2(best_file, dest_file)
    
    return f"Outputs downloaded to {local_path}"


@app.function(
    image=image,
    volumes={"/shared": volume},
)
def download_best_model(local_path: str = "./best_model"):
    """Download only the best model files"""
    import shutil
    from pathlib import Path
    
    local_path = Path(local_path)
    local_path.mkdir(exist_ok=True)
    
    # Copy the best model files
    best_files = [
        ("/shared/outputs/best_model.pth", "Complete model with all training state"),
        ("/shared/outputs/best_generator.pth", "Generator weights only for inference")
    ]
    
    downloaded_files = []
    for src_path, description in best_files:
        if Path(src_path).exists():
            dest_file = local_path / Path(src_path).name
            shutil.copy2(src_path, dest_file)
            downloaded_files.append(f"{dest_file.name}: {description}")
    
    if not downloaded_files:
        return "No best model files found. Training may not have completed successfully."
    
    return f"Best model files downloaded to {local_path}:\n" + "\n".join(downloaded_files)


@app.local_entrypoint()
def main(
    download: bool = False, 
    skip_data: bool = False, 
    force_redownload: bool = False,
    model_info: bool = False,
    download_model: bool = False
):
    """
    Main entrypoint for ChronoColor training on Modal
    
    Args:
        download: If True, download outputs after training
        skip_data: If True, skip data download and processing (assumes data is already processed)
        force_redownload: If True, force redownload even if data exists
        model_info: If True, only show information about the best saved model
        download_model: If True, only download the best model files
    """
    # Handle info-only requests
    if model_info:
        print("📊 Getting best model information...")
        model_info_result = get_best_model_info.remote()
        if model_info_result["status"] == "success":
            print(f"🏆 Best model achieved PSNR: {model_info_result['best_psnr']:.4f} at epoch {model_info_result['epoch']}")
            print(f"📁 Full model: {model_info_result['full_model_path']}")
            print(f"🧠 Generator only: {model_info_result['generator_only_path']}")
        else:
            print(f"⚠️ {model_info_result['message']}")
        return
    
    if download_model:
        print("📥 Downloading best model...")
        download_result = download_best_model.remote()
        print(f"✅ {download_result}")
        return
    
    print("🚀 Starting ChronoColor pipeline on Modal H100...")
    
    try:
        # Download and process data (unless skipped)
        if not skip_data:
            print("📥 Downloading and processing dataset...")
            data_result = download_and_process_data.remote(force_redownload=force_redownload)
            print(f"✅ {data_result}")
        else:
            print("⏭️ Skipping data download and processing...")
        
        # Verify data exists before training
        print("🔍 Verifying data availability...")
        verify_result = verify_data.remote()
        print(f"✅ {verify_result}")
        
        # Run training
        print("🏋️ Starting training...")
        result = run_training.remote()
        print(f"✅ {result}")
        
        # Get best model information
        print("📊 Getting best model information...")
        model_info = get_best_model_info.remote()
        if model_info["status"] == "success":
            print(f"🏆 Best model achieved PSNR: {model_info['best_psnr']:.4f} at epoch {model_info['epoch']}")
            print(f"📁 Model saved in Modal storage at: {model_info['full_model_path']}")
        else:
            print(f"⚠️ {model_info['message']}")
        
        # Optionally download outputs
        if download:
            print("📥 Downloading outputs...")
            download_result = download_outputs.remote()
            print(f"✅ {download_result}")
            
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()