from pathlib import Path
import torch
from datetime import datetime


class Config:
    """Central configuration class for the ChronoColor project."""

    # Project Paths
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    LOG_DIR = PROJECT_ROOT / "logs"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

    # List of decades in your dataset
    DECADES = ["1930s", "1940s", "1950s", "1960s", "1970s"]

    # Create directories if they don't exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Data Parameters
    IMAGE_SIZE = (256, 256)
    BATCH_SIZE = 16
    NUM_WORKERS = 4

    # Model Architecture Parameters
    GEN_FILTERS = 64  # Base number of filters for generator
    DISC_FILTERS = 64  # Base number of filters for discriminator
    LATENT_DIM = 100  # Dimension of noise vector
    DROPOUT_RATE = 0.5  # Dropout rate for generator

    # Training Parameters
    LR_G = 0.0002  # Generator learning rate
    LR_D = 0.0002  # Discriminator learning rate (balanced)
    BETA1 = 0.5
    BETA2 = 0.999
    NUM_EPOCHS = 100
    L1_LAMBDA = 10.0  # Weight for L1 loss in the generator (reduced from 100)
    ADVERSARIAL_LAMBDA = 1.0  # Weight for adversarial loss
    
    # Gradient clipping
    GRAD_CLIP_G = 5.0  # Generator gradient clipping (more lenient)
    GRAD_CLIP_D = 5.0  # Discriminator gradient clipping (more lenient)

    # Training schedule
    PRETRAIN_EPOCHS = 5  # Number of epochs to pretrain generator (reduced warmup)
    SAVE_INTERVAL = 5  # Save checkpoint every N epochs
    SAMPLE_INTERVAL = 100  # Generate sample images every N batches
    LOG_INTERVAL = 10  # Log metrics every N batches

    # Device (GPU/CPU)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Experiment tracking
    EXPERIMENT_NAME = f"chronocolor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


config = Config()