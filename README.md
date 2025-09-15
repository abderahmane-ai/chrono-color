# 🎨 ChronoColor: AI-Powered Historical Photo Colorization

> *Bringing the past to life, one pixel at a time*

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org)
[![Modal](https://img.shields.io/badge/Modal-Cloud%20Training-purple.svg)](https://modal.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

ChronoColor is a cutting-edge deep learning project that automatically colorizes historical black and white photographs using a sophisticated GAN (Generative Adversarial Network) architecture. Transform decades of monochrome memories into vibrant, lifelike images with the power of AI.

## ✨ Features

- 🧠 **Advanced U-Net Generator**: Sophisticated encoder-decoder architecture with skip connections
- ⚔️ **Adversarial Training**: PatchGAN discriminator for realistic colorization
- 📊 **Multi-Decade Dataset**: Trained on historical photos from 1930s-1970s
- ☁️ **Cloud-Powered Training**: Leverages Modal's H100 GPUs for lightning-fast training
- 🎯 **High-Quality Results**: Optimized for PSNR and perceptual quality
- 📈 **Rich Monitoring**: TensorBoard integration with comprehensive metrics
- 🚀 **Easy Inference**: Simple API for colorizing your own photos

## 🏗️ Architecture

ChronoColor employs a sophisticated **conditional GAN** architecture:

- **Generator**: U-Net with skip connections (L → ab color space)
- **Discriminator**: PatchGAN for realistic texture generation
- **Loss Function**: Combined adversarial + L1 loss for optimal results
- **Color Space**: LAB color space for perceptually uniform colorization

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU (recommended)
- Modal account (for cloud training)

### Installation

```bash
# Clone the repository
git clone https://github.com/abderahmane-ai/chrono-color.git
cd chrono-color

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### 🎨 Colorize Your First Photo

```python
from chronocolor import ChronoColorModel

# Load the trained model
model = ChronoColorModel.from_pretrained("best_model.pth")

# Colorize an image
colored_image = model.colorize("path/to/your/bw_photo.jpg")
colored_image.save("colorized_result.png")
```

## 📊 Dataset

ChronoColor is trained on the **Historical Color Database** containing:

- **1,325 images** across 5 decades (1930s-1970s)
- **265 images per decade** for balanced training
- **256×256 resolution** for optimal performance
- **Automatic preprocessing** with LAB color space conversion

### Data Distribution
```
1930s: 265 images  ████████████████████
1940s: 265 images  ████████████████████
1950s: 265 images  ████████████████████
1960s: 265 images  ████████████████████
1970s: 265 images  ████████████████████
```

## ☁️ Cloud Training with Modal

**⚠️ Security Notice**: The Modal deployment code was entirely AI-generated for rapid H100 training and may contain security vulnerabilities. Use with caution in production environments.

### Train on H100 GPUs

```bash
# Full training pipeline
modal run modal/modal_app.py

# Skip data download (if already processed)
modal run modal/modal_app.py --skip-data

# Download trained model
python simple_download.py
```

### Inference on Cloud

```bash
# Test model loading
modal run modal/modal_inference.py --test-model

# Colorize single image
modal run modal/modal_inference.py --image-path photo.jpg

# Batch processing
modal run modal/modal_inference.py --batch-dir ./photos --batch-name my_batch
```

## 🏋️ Local Training

```bash
# Process dataset
python src/scripts/process_images.py

# Start training
python src/scripts/train.py

# Monitor with TensorBoard
tensorboard --logdir logs/
```

## 📈 Model Performance

Our best model achieves:
- **PSNR**: 26.5+ dB on validation set
- **Training Time**: ~10 minutes on H100 GPU
- **Model Size**: ~664MB (generator only)
- **Inference Speed**: ~2s per 256×256 image

## 🛠️ Project Structure

```
chronocolor/
├── 📁 src/
│   ├── 🧠 modeling/          # Neural network architectures
│   ├── 🏋️ training/          # Training loops and losses
│   ├── 📊 data_loading/      # Dataset and transforms
│   ├── 🔧 utils/             # Utilities and visualization
│   └── 📜 scripts/           # Training and processing scripts
├── ☁️ modal/                 # Cloud deployment (AI-generated)
├── 📓 notebooks/             # Jupyter exploration notebooks
├── 📊 data/                  # Dataset storage
└── 🎯 simple_download.py     # Model download utility
```

## 🎛️ Configuration

Key hyperparameters in `src/config.py`:

```python
# Model Architecture
IMAGE_SIZE = (256, 256)
GEN_FILTERS = 64
DISC_FILTERS = 64

# Training Parameters
LR_G = 0.0002          # Generator learning rate
LR_D = 0.0002          # Discriminator learning rate
L1_LAMBDA = 10.0       # L1 loss weight
NUM_EPOCHS = 100       # Training epochs
BATCH_SIZE = 16        # Batch size
```

## 🔬 Technical Details

### Color Space Conversion
- **Input**: Grayscale (L channel) images
- **Output**: ab color channels in LAB space
- **Final**: LAB → RGB conversion for display

### Loss Function
```
Total Loss = λ₁ × L1_Loss + λ₂ × Adversarial_Loss
```

### Data Augmentation
- Random horizontal flips
- Normalization to [-1, 1] range
- Consistent L/ab channel processing

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Historical Color Database**: Carnegie Mellon University Graphics Lab
- **Modal**: For providing H100 GPU access for rapid training
- **PyTorch Team**: For the excellent deep learning framework
- **U-Net Architecture**: Ronneberger et al. (2015)
- **PatchGAN**: Isola et al. (2017)

## 📚 References

- Ronneberger, O., et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI 2015.
- Isola, P., et al. "Image-to-Image Translation with Conditional Adversarial Networks." CVPR 2017.
- Zhang, R., et al. "Colorful Image Colorization." ECCV 2016.

## 🐛 Known Issues

- Modal deployment code is AI-generated and may contain vulnerabilities
- Training requires significant GPU memory (8GB+ recommended)
- Color accuracy may vary for certain historical periods

## 🔮 Future Work

- [ ] Support for higher resolution images (512×512, 1024×1024)
- [ ] Interactive web interface for easy colorization
- [ ] Fine-tuning for specific historical periods
- [ ] Integration with historical photo databases
- [ ] Mobile app development

---

<div align="center">

**Made with ❤️ and lots of ☕**

*Transform your black and white memories into colorful stories*

[🌟 Star this repo](https://github.com/yourusername/chronocolor) • [🐛 Report Bug](https://github.com/yourusername/chronocolor/issues) • [💡 Request Feature](https://github.com/yourusername/chronocolor/issues)

</div>