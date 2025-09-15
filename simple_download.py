#!/usr/bin/env python3
"""
Simple script to download the best model from Modal volume to local directory
"""
import modal
from pathlib import Path
import os

app = modal.App("simple-download")

# Use the same volume as the training app
volume = modal.Volume.from_name("chronocolor-outputs", create_if_missing=False)

@app.function(
    volumes={"/shared": volume},
    timeout=300,
)
def get_model_bytes():
    """Read the model file and return its bytes"""
    model_path = Path("/shared/outputs/best_model.pth")
    if not model_path.exists():
        return None
    
    with open(model_path, "rb") as f:
        return f.read()

@app.local_entrypoint()
def main():
    print("📥 Downloading best_model.pth from Modal...")
    
    # Get the model bytes from Modal
    model_bytes = get_model_bytes.remote()
    
    if model_bytes is None:
        print("❌ Model file not found in Modal volume")
        return
    
    # Create local directory
    os.makedirs("downloaded_models", exist_ok=True)
    
    # Write the model file locally
    local_path = Path("downloaded_models/best_model.pth")
    with open(local_path, "wb") as f:
        f.write(model_bytes)
    
    print(f"✅ Model downloaded successfully to: {local_path}")
    print(f"📊 File size: {len(model_bytes) / (1024*1024):.1f} MB")

if __name__ == "__main__":
    main()