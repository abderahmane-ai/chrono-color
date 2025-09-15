#!/usr/bin/env python3
"""
Modal script to test the best ChronoColor model on grayscale images
and download the colored results
"""
import modal
from pathlib import Path
import io
import base64

app = modal.App("chronocolor-inference")

# Create the image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install([
        "torch>=2.8.0",
        "torchvision>=0.23.0",
        "pillow>=11.3.0",
        "numpy>=2.3.3",
        "opencv-python>=4.11.0.86",
        "scikit-image>=0.25.2",
        "matplotlib>=3.10.6",
        "rich>=14.1.0",
    ])
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0"])
    .add_local_dir("src", remote_path="/app/src")
)

# Use the same volume as training
volume = modal.Volume.from_name("chronocolor-outputs", create_if_missing=False)

@app.function(
    image=image,
    volumes={"/shared": volume},
    gpu="T4",  # T4 is sufficient for inference
    timeout=600,
    memory=8192,
)
def load_model_and_test():
    """Load the best model and prepare for inference"""
    import torch
    import sys
    import os
    from pathlib import Path
    
    # Set up paths
    os.chdir("/app")
    sys.path.insert(0, "/app/src")
    
    from rich.console import Console
    from rich.panel import Panel
    
    console = Console()
    console.print(Panel.fit(
        "[bold blue]ChronoColor Model Loading[/bold blue]\n"
        "Loading best trained model for inference",
        title="Model Inference"
    ))
    
    # Check if model exists
    best_model_path = Path("/shared/outputs/best_model.pth")
    best_generator_path = Path("/shared/outputs/best_generator.pth")
    
    if not best_model_path.exists() and not best_generator_path.exists():
        return {
            "status": "error",
            "message": "No trained model found. Please run training first."
        }
    
    # Import model components
    from modeling.generator import Generator, init_weights_normal
    from config import config
    
    # Create generator
    generator = Generator(in_channels=1, out_channels=2)
    generator.apply(init_weights_normal)
    generator.to(config.DEVICE)
    
    # Load model weights with proper PyTorch 2.6+ handling
    import pathlib
    torch.serialization.add_safe_globals([pathlib.PosixPath])
    
    if best_generator_path.exists():
        console.print(f"[green]Loading generator-only weights from: {best_generator_path}[/green]")
        try:
            checkpoint = torch.load(best_generator_path, map_location=config.DEVICE, weights_only=True)
        except Exception as e:
            console.print(f"[yellow]Falling back to weights_only=False due to: {e}[/yellow]")
            checkpoint = torch.load(best_generator_path, map_location=config.DEVICE, weights_only=False)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        best_psnr = checkpoint.get('best_psnr', 'Unknown')
        epoch = checkpoint.get('epoch', 'Unknown')
    else:
        console.print(f"[green]Loading full model from: {best_model_path}[/green]")
        try:
            checkpoint = torch.load(best_model_path, map_location=config.DEVICE, weights_only=True)
        except Exception as e:
            console.print(f"[yellow]Falling back to weights_only=False due to: {e}[/yellow]")
            checkpoint = torch.load(best_model_path, map_location=config.DEVICE, weights_only=False)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        best_psnr = checkpoint.get('best_psnr', 'Unknown')
        epoch = checkpoint.get('epoch', 'Unknown')
    
    generator.eval()
    
    console.print(f"[bold een]✓ Model loaded successfully![/bold green]")
    console.print(f"[cyan]Best PSNR: {best_psnr}[/cyan]")
    console.print(f"[cyan]Epoch: {epoch}[/cyan]")
    console.print(f"[cyan]Device: {config.DEVICE}[/cyan]")
    
    return {
        "status": "success",
        "message": "Model loaded successfully",
        "best_psnr": best_psnr,
        "epoch": epoch,
        "device": str(config.DEVICE)
    }

@app.function(
    image=image,
    volumes={"/shared": volume},
    gpu="T4",
    timeout=600,
    memory=8192,
)
def colorize_image(image_data: bytes, image_name: str = "test_image"):
    """Colorize a single grayscale image"""
    import torch
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    import cv2
    import sys
    import os
    import io
    import base64
    from pathlib import Path
    
    # Set up paths
    os.chdir("/app")
    sys.path.insert(0, "/app/src")
    
    from modeling.generator import Generator, init_weights_normal
    from config import config
    from rich.console import Console
    
    console = Console()
    
    try:
        # Load model
        generator = Generator(in_channels=1, out_channels=2)
        generator.apply(init_weights_normal)
        generator.to(config.DEVICE)
        
        # Load weights with proper PyTorch 2.6+ handling
        import pathlib
        torch.serialization.add_safe_globals([pathlib.PosixPath])
        
        best_generator_path = Path("/shared/outputs/best_generator.pth")
        best_model_path = Path("/shared/outputs/best_model.pth")
        
        if best_generator_path.exists():
            try:
                checkpoint = torch.load(best_generator_path, map_location=config.DEVICE, weights_only=True)
            except Exception as e:
                console.print(f"[yellow]Falling back to weights_only=False due to: {e}[/yellow]")
                checkpoint = torch.load(best_generator_path, map_location=config.DEVICE, weights_only=False)
        elif best_model_path.exists():
            try:
                checkpoint = torch.load(best_model_path, map_location=config.DEVICE, weights_only=True)
            except Exception as e:
                console.print(f"[yellow]Falling back to weights_only=False due to: {e}[/yellow]")
                checkpoint = torch.load(best_model_path, map_location=config.DEVICE, weights_only=False)
        else:
            return {"status": "error", "message": "No model found"}
        
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.eval()
        
        # Load and preprocess image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to 256x256 as requested
        target_size = (256, 256)
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy and normalize
        img_array = np.array(image).astype(np.float32)
        img_array = (img_array / 255.0) * 2.0 - 1.0  # Normalize to [-1, 1]
        
        # Convert to tensor and add batch dimension
        L_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(config.DEVICE)
        
        console.print(f"[yellow]Processing image: {image_name}[/yellow]")
        console.print(f"[dim]Input shape: {L_tensor.shape}[/dim]")
        console.print(f"[dim]Input range: [{L_tensor.min():.3f}, {L_tensor.max():.3f}][/dim]")
        
        # Generate colorization
        with torch.no_grad():
            fake_ab = generator(L_tensor)
        
        console.print(f"[dim]Output shape: {fake_ab.shape}[/dim]")
        console.print(f"[dim]Output range: [{fake_ab.min():.3f}, {fake_ab.max():.3f}][/dim]")
        
        # Convert back to numpy
        L_np = L_tensor.squeeze().cpu().numpy()
        ab_np = fake_ab.squeeze().cpu().numpy()
        
        # Denormalize
        L_np = (L_np + 1.0) / 2.0 * 100.0  # L channel: [0, 100]
        ab_np = ab_np * 128.0  # ab channels: [-128, 128]
        
        # Combine L and ab channels
        lab_image = np.zeros((256, 256, 3), dtype=np.float32)
        lab_image[:, :, 0] = L_np
        lab_image[:, :, 1] = ab_np[0]
        lab_image[:, :, 2] = ab_np[1]
        
        # Convert LAB to RGB - OpenCV expects specific LAB ranges
        # L: [0, 100] -> [0, 255], A: [-128, 127] -> [0, 255], B: [-128, 127] -> [0, 255]
        lab_for_cv2 = lab_image.copy()
        lab_for_cv2[:, :, 0] = np.clip(lab_for_cv2[:, :, 0] * 255.0 / 100.0, 0, 255)  # L channel
        lab_for_cv2[:, :, 1] = np.clip(lab_for_cv2[:, :, 1] + 128, 0, 255)  # A channel
        lab_for_cv2[:, :, 2] = np.clip(lab_for_cv2[:, :, 2] + 128, 0, 255)  # B channel
        
        lab_image_uint8 = lab_for_cv2.astype(np.uint8)
        rgb_image = cv2.cvtColor(lab_image_uint8, cv2.COLOR_LAB2RGB)
        
        # Convert to PIL Image
        colored_image = Image.fromarray(rgb_image)
        
        # Save colored image to bytes
        colored_buffer = io.BytesIO()
        colored_image.save(colored_buffer, format='PNG')
        colored_bytes = colored_buffer.getvalue()
        
        # Also create a side-by-side comparison
        # Convert grayscale to RGB properly
        grayscale_normalized = ((img_array + 1.0) / 2.0 * 255.0).astype(np.uint8)
        grayscale_rgb = Image.fromarray(np.stack([grayscale_normalized] * 3, axis=-1))
        
        # Create side-by-side comparison
        comparison_width = 256 * 2
        comparison_height = 256
        comparison = Image.new('RGB', (comparison_width, comparison_height))
        comparison.paste(grayscale_rgb, (0, 0))
        comparison.paste(colored_image, (256, 0))
        
        # Save comparison to bytes
        comparison_buffer = io.BytesIO()
        comparison.save(comparison_buffer, format='PNG')
        comparison_bytes = comparison_buffer.getvalue()
        
        console.print(f"[bold green]✓ Successfully colorized {image_name}[/bold green]")
        
        return {
            "status": "success",
            "message": f"Successfully colorized {image_name}",
            "colored_image": base64.b64encode(colored_bytes).decode('utf-8'),
            "comparison_image": base64.b64encode(comparison_bytes).decode('utf-8'),
            "image_name": image_name
        }
        
    except Exception as e:
        console.print(f"[red]Error processing {image_name}: {str(e)}[/red]")
        return {
            "status": "error",
            "message": f"Error processing {image_name}: {str(e)}"
        }

@app.function(
    image=image,
    volumes={"/shared": volume},
    gpu="T4",
    timeout=1800,  # 30 minutes for batch processing
    memory=8192,
)
def colorize_batch(image_files: dict):
    """Colorize multiple images in batch"""
    import sys
    import os
    from pathlib import Path
    
    # Set up paths
    os.chdir("/app")
    sys.path.insert(0, "/app/src")
    
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TextColumn
    
    console = Console()
    console.print(f"[yellow]Processing batch of {len(image_files)} images...[/yellow]")
    
    results = {}
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("Colorizing images...", total=len(image_files))
        
        for image_name, image_data in image_files.items():
            result = colorize_image.local(image_data, image_name)
            results[image_name] = result
            progress.advance(task)
    
    successful = sum(1 for r in results.values() if r["status"] == "success")
    console.print(f"[bold green]✓ Batch processing complete: {successful}/{len(image_files)} successful[/bold green]")
    
    return results

@app.function(
    image=image,
    volumes={"/shared": volume},
    timeout=300,
)
def save_results_to_volume(results: dict, batch_name: str = "inference_results"):
    """Save inference results to the Modal volume for download"""
    import json
    import base64
    from pathlib import Path
    import os
    
    # Create results directory in volume
    results_dir = Path(f"/shared/inference_results/{batch_name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    for image_name, result in results.items():
        if result["status"] == "success":
            # Save colored image
            colored_data = base64.b64decode(result["colored_image"])
            colored_path = results_dir / f"{Path(image_name).stem}_colored.png"
            with open(colored_path, "wb") as f:
                f.write(colored_data)
            saved_files.append(str(colored_path))
            
            # Save comparison image
            comparison_data = base64.b64decode(result["comparison_image"])
            comparison_path = results_dir / f"{Path(image_name).stem}_comparison.png"
            with open(comparison_path, "wb") as f:
                f.write(comparison_data)
            saved_files.append(str(comparison_path))
    
    # Save results metadata
    metadata = {
        "batch_name": batch_name,
        "total_images": len(results),
        "successful": sum(1 for r in results.values() if r["status"] == "success"),
        "failed": sum(1 for r in results.values() if r["status"] == "error"),
        "saved_files": saved_files,
        "results": results
    }
    
    metadata_path = results_dir / "results_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return {
        "status": "success",
        "message": f"Results saved to {results_dir}",
        "saved_files": len(saved_files),
        "metadata_path": str(metadata_path)
    }

@app.function(
    image=image,
    volumes={"/shared": volume},
    timeout=300,
)
def download_inference_results(batch_name: str = "inference_results"):
    """Download inference results from Modal volume"""
    import shutil
    import zipfile
    from pathlib import Path
    import tempfile
    
    results_dir = Path(f"/shared/inference_results/{batch_name}")
    
    if not results_dir.exists():
        return {
            "status": "error",
            "message": f"No results found for batch: {batch_name}"
        }
    
    # Create a zip file with all results
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
        with zipfile.ZipFile(tmp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in results_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(results_dir)
                    zipf.write(file_path, arcname)
        
        # Read zip file content
        with open(tmp_zip.name, 'rb') as f:
            zip_content = f.read()
    
    return {
        "status": "success",
        "zip_content": zip_content,
        "batch_name": batch_name,
        "file_count": len(list(results_dir.rglob('*')))
    }

@app.local_entrypoint()
def main(
    image_path: str = None,
    batch_dir: str = None,
    download_results: bool = False,
    batch_name: str = "inference_results",
    test_model: bool = False
):
    """
    Main entrypoint for ChronoColor inference
    
    Args:
        image_path: Path to a single image to colorize
        batch_dir: Directory containing multiple images to colorize
        download_results: Download results from previous inference
        batch_name: Name for the batch (used for organizing results)
        test_model: Just test if the model loads correctly
    """
    from pathlib import Path
    import base64
    
    print("🎨 ChronoColor Inference on Modal")
    
    # Test model loading
    if test_model:
        print("🔍 Testing model loading...")
        result = load_model_and_test.remote()
        if result["status"] == "success":
            print(f"✅ Model loaded successfully!")
            print(f"   Best PSNR: {result['best_psnr']}")
            print(f"   Epoch: {result['epoch']}")
            print(f"   Device: {result['device']}")
        else:
            print(f"❌ {result['message']}")
        return
    
    # Download previous results
    if download_results:
        print(f"📥 Downloading results for batch: {batch_name}")
        result = download_inference_results.remote(batch_name)
        if result["status"] == "success":
            output_path = Path(f"{batch_name}_results.zip")
            with open(output_path, "wb") as f:
                f.write(result["zip_content"])
            print(f"✅ Results downloaded to: {output_path}")
            print(f"   Files in archive: {result['file_count']}")
        else:
            print(f"❌ {result['message']}")
        return
    
    # Process single image
    if image_path:
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            return
        
        print(f"🎨 Colorizing single image: {image_path.name}")
        
        # Read image
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Process image
        result = colorize_image.remote(image_data, image_path.name)
        
        if result["status"] == "success":
            # Save results locally
            output_dir = Path("colorized_output")
            output_dir.mkdir(exist_ok=True)
            
            # Save colored image
            colored_data = base64.b64decode(result["colored_image"])
            colored_path = output_dir / f"{image_path.stem}_colored.png"
            with open(colored_path, "wb") as f:
                f.write(colored_data)
            
            # Save comparison
            comparison_data = base64.b64decode(result["comparison_image"])
            comparison_path = output_dir / f"{image_path.stem}_comparison.png"
            with open(comparison_path, "wb") as f:
                f.write(comparison_data)
            
            print(f"✅ Colorization complete!")
            print(f"   Colored image: {colored_path}")
            print(f"   Comparison: {comparison_path}")
        else:
            print(f"❌ {result['message']}")
        return
    
    # Process batch of images
    if batch_dir:
        batch_path = Path(batch_dir)
        if not batch_path.exists():
            print(f"❌ Directory not found: {batch_path}")
            return
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = {}
        
        for ext in image_extensions:
            for img_path in batch_path.glob(f"*{ext}"):
                with open(img_path, "rb") as f:
                    image_files[img_path.name] = f.read()
            for img_path in batch_path.glob(f"*{ext.upper()}"):
                with open(img_path, "rb") as f:
                    image_files[img_path.name] = f.read()
        
        if not image_files:
            print(f"❌ No image files found in: {batch_path}")
            return
        
        print(f"🎨 Processing batch of {len(image_files)} images from: {batch_path}")
        
        # Process batch
        results = colorize_batch.remote(image_files)
        
        # Save results to volume
        save_result = save_results_to_volume.remote(results, batch_name)
        print(f"✅ {save_result['message']}")
        print(f"   Saved files: {save_result['saved_files']}")
        
        # Also save locally
        output_dir = Path(f"colorized_batch_{batch_name}")
        output_dir.mkdir(exist_ok=True)
        
        successful = 0
        for image_name, result in results.items():
            if result["status"] == "success":
                # Save colored image
                colored_data = base64.b64decode(result["colored_image"])
                colored_path = output_dir / f"{Path(image_name).stem}_colored.png"
                with open(colored_path, "wb") as f:
                    f.write(colored_data)
                
                # Save comparison
                comparison_data = base64.b64decode(result["comparison_image"])
                comparison_path = output_dir / f"{Path(image_name).stem}_comparison.png"
                with open(comparison_path, "wb") as f:
                    f.write(comparison_data)
                
                successful += 1
        
        print(f"✅ Batch processing complete!")
        print(f"   Successful: {successful}/{len(image_files)}")
        print(f"   Local output: {output_dir}")
        print(f"   Modal storage: Use --download-results --batch-name {batch_name} to download later")
        return
    
    # No arguments provided
    print("❌ Please provide either --image-path, --batch-dir, or --download-results")
    print("\nUsage examples:")
    print("  # Test model loading:")
    print("  python modal_inference.py --test-model")
    print("  # Colorize single image:")
    print("  python modal_inference.py --image-path path/to/image.jpg")
    print("  # Colorize batch of images:")
    print("  python modal_inference.py --batch-dir path/to/images/ --batch-name my_batch")
    print("  # Download previous results:")
    print("  python modal_inference.py --download-results --batch-name my_batch")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ChronoColor Inference on Modal")
    parser.add_argument("--image-path", type=str, help="Path to single image to colorize")
    parser.add_argument("--batch-dir", type=str, help="Directory containing images to colorize")
    parser.add_argument("--download-results", action="store_true", help="Download previous results")
    parser.add_argument("--batch-name", type=str, default="inference_results", help="Batch name for organizing results")
    parser.add_argument("--test-model", action="store_true", help="Test model loading")
    
    args = parser.parse_args()
    
    main(
        image_path=args.image_path,
        batch_dir=args.batch_dir,
        download_results=args.download_results,
        batch_name=args.batch_name,
        test_model=args.test_model
    )