# Image Colorization using Convolutional U-Net in PyTorch

An end-to-end deep learning project that automatically converts black-and-white (grayscale) images into natural, full-color photographs using a **Convolutional U-Net** and the **CIE Lab** color space.

---

## Table of Contents
- [Overview](#overview)
- [How It Works](#how-it-works)
- [Model Architecture](#model-architecture)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Color-Weighted Loss Function](#color-weighted-loss-function)
- [Training Details & Results](#training-details--results)
- [Inference & Visual Comparison](#inference--visual-comparison)
- [Installation & Usage](#installation--usage)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)

---

## Overview

Image colorization is the task of hallucinating plausible color channels for a single-channel grayscale input image. 

Traditional RGB-based models often struggle because predicting 3 color channels ($R, G, B$) from luminance can result in unnatural artifacts. This project instead operates in the **CIE Lab** color space:
- **Input ($L$ channel)**: Represents Lightness (grayscale).
- **Target ($a$ and $b$ channels)**: Represent color opponents (green–red and blue–yellow).

By predicting only the $a$ and $b$ chrominance channels and combining them with the original $L$ channel, the model preserves 100% of the input image's original sharpness, textures, and luminance.

---

## How It Works

```
Grayscale Image (L Channel) 
           │
           ▼
┌─────────────────────────┐
│   Colorization U-Net    │ ─── Predicts a & b channels
└─────────────────────────┘
           │
           ▼
    Predicted (a, b)
           │
           ▼
   Combine with Original L
   (Lab Color Reconstruction)
           │
           ▼
  Convert Lab ──▶ RGB Output (Full Color Image)
```

---

## Model Architecture

The model uses a **U-Net** architecture with skip connections. Skip connections allow high-resolution feature maps from early encoder layers to be copied directly to corresponding decoder layers, helping retain fine edge and texture details.

* **Basic Block (`DoubleConv`)**: Two consecutive $3 \times 3$ convolutions, each followed by `ReLU` activation and padding to preserve spatial dimensions.
* **Encoder**:
  * Block 1: $1 \rightarrow 32$ channels ($128 \times 128$) $\rightarrow$ MaxPool ($64 \times 64$)
  * Block 2: $32 \rightarrow 64$ channels ($64 \times 64$) $\rightarrow$ MaxPool ($32 \times 32$)
  * Block 3: $64 \rightarrow 128$ channels ($32 \times 32$) $\rightarrow$ MaxPool ($16 \times 16$)
* **Bottleneck**:
  * $128 \rightarrow 256$ channels ($16 \times 16$)
* **Decoder**:
  * Transpose Conv ($2 \times 2$) + Skip Connection from Block 3 $\rightarrow 256 \rightarrow 128$ channels ($32 \times 32$)
  * Transpose Conv ($2 \times 2$) + Skip Connection from Block 2 $\rightarrow 128 \rightarrow 64$ channels ($64 \times 64$)
  * Transpose Conv ($2 \times 2$) + Skip Connection from Block 1 $\rightarrow 64 \rightarrow 32$ channels ($128 \times 128$)
* **Output Layer**:
  * $1 \times 1$ Convolution: $32 \rightarrow 2$ channels (predicted $\hat{a}$ and $\hat{b}$).

---

## Dataset & Preprocessing

- **Dataset**: [Fashionpedia](https://fashionpedia.info/) training dataset (~45,600 images).
- **Subset**: 30,000 images sampled with a fixed random seed (`seed=42`).
- **Resolution**: Resized to $128 \times 128$ pixels using bicubic resampling.
- **Normalization**:
  - $L$ channel scaled from $[0, 100]$ to $[-1.0, 1.0]$: `(L / 50.0) - 1.0`
  - $a$ and $b$ channels scaled by dividing by $128.0$: `ab / 128.0`
- **Train / Validation Split**:
  - **Training set**: 24,000 images (80%)
  - **Validation set**: 6,000 images (20%)

---

## Color-Weighted Loss Function

Standard Mean Squared Error (MSE) loss often produces washed-out, brownish, or sepia colors. This happens because the model plays it safe: predicting dull gray/brown minimizes average mathematical error across varied objects.

To solve this problem, a **chroma-weighted MSE loss** is implemented:

$$\text{chroma} = \sqrt{a^2 + b^2}$$
$$\text{weight} = 1.0 + 2.0 \cdot \min\left(1.0, \frac{\text{chroma}}{0.35}\right)$$
$$\text{Loss} = \frac{\sum (\text{weight} \cdot (\hat{y} - y)^2)}{\sum \text{weight}}$$

Pixels with higher color saturation (vibrant colors) receive up to **$3\times$ higher loss penalty**, forcing the neural network to learn and predict vivid colors instead of defaulting to dull grays.

---

## Training Details & Results

- **Framework**: PyTorch
- **Hardware**: NVIDIA Tesla T4 GPU (Google Colab)
- **Optimizer**: Adam ($\text{learning rate} = 0.001$)
- **Batch Size**: 16 (1,500 train batches, 375 val batches per epoch)
- **Epochs**: Configured for 20 epochs (~6.3 minutes per epoch)
- **Checkpoints**: Saved periodically to Google Drive with automated resumption support.

### Sample Loss History

| Epoch | Train MSE | Validation MSE | Time per Epoch |
| :---: | :---: | :---: | :---: |
| 01 | 0.01578 | 0.01597 | 6.5 min |
| 05 | 0.01354 | 0.01314 | 6.4 min |
| 10 | 0.01260 | 0.01312 | 6.2 min |
| 15 | 0.01171 | 0.01241 | 6.3 min |
| 17 | 0.01149 | 0.01236 | 6.3 min |

---

## Inference & Visual Comparison

The notebook provides two evaluation methods:
1. **Validation Comparison**: Generates a $3 \times 3$ grid comparing:
   - Input Grayscale ($L$ only)
   - Model Colorized Prediction ($\hat{a}, \hat{b} + L$)
   - Ground Truth Original Photo
2. **Interactive Custom Image Upload**:
   - Upload any personal image via a Colab upload button.
   - The notebook converts it to grayscale, passes it through the trained network, and displays the colorized output side-by-side with the original input.

---

## Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/image-colorization-unet.git
cd image-colorization-unet
```

### 2. Install Dependencies
Make sure you have Python 3.8+ installed. Install the required libraries:
```bash
pip install torch torchvision numpy scikit-image pillow matplotlib
```

### 3. Run the Notebook
Open `CNN.ipynb` in **Jupyter Lab**, **VS Code**, or **Google Colab** (recommended for GPU support):
```bash
jupyter notebook CNN.ipynb
```

> **Note:** If running in Google Colab, enable GPU acceleration under **Runtime > Change runtime type > Hardware accelerator > GPU (T4)**.

---

## Project Structure

```text
├── CNN.ipynb          # Complete training, evaluation, and inference notebook
└── README.md          # Project documentation
```

---

## Future Improvements

- [ ] Train on higher image resolutions ($256 \times 256$ or $512 \times 512$).
- [ ] Add an adversarial discriminator loss (cGAN / Pix2Pix architecture) for sharper textures.
- [ ] Add a lightweight web interface using Gradio or Streamlit for quick demonstrations.

---

## Acknowledgments
- Dataset provided by [Fashionpedia](https://fashionpedia.info/).
- Architecture inspired by the original [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) (Ronneberger et al., 2015).
