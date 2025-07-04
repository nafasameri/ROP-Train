# Retinopathy of Prematurity (ROP) Classification

A comprehensive machine learning project for automated classification of Retinopathy of Prematurity (ROP) in retinal images using deep learning techniques. This repository implements multiple state-of-the-art CNN architectures with advanced image preprocessing methods to achieve accurate binary classification of ROP severity.

## 🎯 Project Overview

Retinopathy of Prematurity (ROP) is a potentially blinding eye disorder that primarily affects premature infants. Early detection and accurate classification are crucial for preventing vision loss. This project develops automated classification systems to assist healthcare professionals in ROP diagnosis.

### Key Features

- **Multiple CNN Architectures**: Implementation of 6 different deep learning models
- **Advanced Preprocessing**: CLAHE and AMSR enhancement techniques
- **Comprehensive Evaluation**: Detailed performance metrics and visualizations
- **Binary Classification**: Normal vs Plus disease classification
- **Transfer Learning**: Pretrained models fine-tuned for medical imaging

## 🏗️ Architecture

### Implemented Models

| Model | Framework | Key Features |
|-------|-----------|--------------|
| **ResNet18** | PyTorch | Residual connections, efficient training |
| **ResNet50** | PyTorch/Keras | Deeper architecture, best performance |
| **VGG16** | PyTorch | Simple architecture, reliable baseline |
| **VGG19** | PyTorch | Deeper VGG variant |
| **DenseNet121** | PyTorch | Dense connections, parameter efficiency |
| **MobileNet** | TensorFlow/Keras | Lightweight, mobile-optimized |

### Preprocessing Techniques

1. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
   - Enhances local contrast in retinal images
   - Reduces noise while preserving important features
   - Applied to LAB color space L-channel

2. **AMSR (Automated Multi-Scale Retinex)**
   - Multi-scale retinex enhancement
   - Brightness and contrast adjustment
   - Gaussian blur filtering with multiple sigma values

3. **Circular Masking**
   - Removes background artifacts
   - Focuses on retinal region of interest
   - Applied to all preprocessed images

## 📊 Results

### Performance Summary

| Model | Preprocessing | Accuracy | Sensitivity | Specificity | F1 Score | AUC |
|-------|---------------|----------|-------------|-------------|----------|-----|
| **ResNet50** | CLAHE | **90.88%** | 87.83% | 93.22% | 89.34% | 0.91 |
| **DenseNet121** | CLAHE + Crop | 88.54% | 88.82% | 89.18% | 87.41% | 0.89 |
| **MobileNet** | CLAHE | 86.17% | 79.72% | 91.14% | 83.39% | 0.85 |
| **DenseNet121** | CLAHE | 85.00% | 82.43% | 86.97% | 82.71% | 0.85 |
| **ResNet18** | CLAHE + Crop | 81.47% | 75.00% | 86.45% | 77.89% | 0.81 |
| **VGG16** | CLAHE | 80.88% | 79.05% | 82.29% | 78.26% | 0.81 |
| **VGG19** | CLAHE | 73.23% | 96.62% | 55.20% | 75.86% | 0.76 |

*ResNet50 with CLAHE preprocessing achieved the best overall performance*

## 🛠️ Installation

### Prerequisites

```bash
# Python 3.7+
# CUDA-capable GPU (recommended)
```

### Dependencies

```bash
# Core ML frameworks
pip install torch torchvision
pip install tensorflow keras

# Image processing
pip install opencv-python
pip install pillow

# Scientific computing
pip install numpy pandas
pip install scikit-learn
pip install matplotlib

# Additional utilities
pip install openpyxl  # For Excel export
```

### Quick Install

```bash
git clone <repository-url>
cd rop-classification
pip install -r requirements.txt
```

## 🚀 Usage

### Dataset Structure

Organize your dataset as follows:
```
dataset/
├── 0/          # Normal cases
│   ├── image1.jpg
│   └── image2.jpg
└── 1/          # Plus disease cases
    ├── image1.jpg
    └── image2.jpg
```

### Training Models

#### PyTorch Models (ResNet, VGG, DenseNet)

```bash
# Train ResNet50
python resnet_50_keras.py

# Train VGG19
python vgg19_torch.py

# Train DenseNet121
python densenet121_torch.py
```

#### TensorFlow Models

```bash
# Train MobileNet
python mobilenet.py
```

### Preprocessing Only

```bash
# Apply CLAHE enhancement
from clahe import perproccessing
perproccessing('input_image.jpg', 'output_image.jpg')

# Apply AMSR enhancement
from amsr import perproccessing
perproccessing('input_image.jpg', 'output_image.jpg')
```

### Evaluation

Each model script automatically generates:
- Performance metrics (accuracy, sensitivity, specificity, etc.)
- Confusion matrices
- ROC curves
- Training loss plots
- Classification reports

Results are saved in model-specific directories and compiled in `results.html`.

## 📁 Repository Structure

```
├── README.md                          # This file
├── results.html                       # Comprehensive results comparison
├── .gitignore                        # Git ignore file
│
├── Model Implementations/
│   ├── vgg16_torch.py                # VGG16 PyTorch implementation
│   ├── vgg19_torch.py                # VGG19 PyTorch implementation
│   ├── resnet_18_torch.py            # ResNet18 PyTorch implementation
│   ├── resnet_50_keras.py            # ResNet50 Keras implementation
│   ├── densenet121_torch.py          # DenseNet121 PyTorch implementation
│   └── mobilenet.py                  # MobileNet TensorFlow implementation
│
├── Preprocessing/
│   ├── clahe.py                      # CLAHE enhancement
│   ├── amsr.py                       # AMSR enhancement
│   └── mask.py                       # Circular masking utility
│
└── Results/                          # Experimental results
    ├── resnet50 200 clahe/           # ResNet50 results
    ├── densenet121 clahe 200/        # DenseNet121 results
    ├── mobilenet 200 clahe/          # MobileNet results
    └── [other model results]/
```

## 🔬 Technical Details

### Model Architecture Modifications

- **Transfer Learning**: All models use ImageNet pretrained weights
- **Fine-tuning**: Selective layer unfreezing for medical domain adaptation
- **Regularization**: Dropout, batch normalization, and weight decay
- **Data Augmentation**: Random rotation, flipping, cropping, and color jittering

### Training Configuration

```python
# Hyperparameters
batch_size = 32
learning_rate = 1e-4 to 1e-5
num_epochs = 200
optimizer = Adam
loss_function = CrossEntropyLoss
```

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Sensitivity (Recall)**: True positive rate
- **Specificity**: True negative rate  
- **Precision**: Positive predictive value
- **F1 Score**: Harmonic mean of precision and recall
- **Jaccard Index**: Intersection over union
- **AUC**: Area under ROC curve
- **Matthews Correlation Coefficient**: Balanced metric for binary classification

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{rop-classification,
  title={Retinopathy of Prematurity Classification using Deep Learning},
  author={[Your Name]},
  year={2024},
  howpublished={\url{https://github.com/[username]/rop-classification}}
}
```

## 🆘 Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Contact: [your-email@domain.com]

## ⚠️ Medical Disclaimer

This software is for research purposes only and should not be used for clinical diagnosis without proper validation and regulatory approval. Always consult qualified healthcare professionals for medical decisions.

---

**Keywords**: Retinopathy of Prematurity, Medical Image Analysis, Deep Learning, Computer Vision, Transfer Learning, PyTorch, TensorFlow, CLAHE, Medical AI