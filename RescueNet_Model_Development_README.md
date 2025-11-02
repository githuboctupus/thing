# RescueNet Segmentation Project — Full Development History (BWSI → Oct 2025)

This document details the complete development of **RescueNet**, an aerial-imagery segmentation system designed to detect and classify post-disaster features such as damaged buildings, water, roads, and vehicles.  
It traces the project’s evolution from an initial classroom prototype at the **MIT Beaver Works Summer Institute (BWSI)** to a mature, research-grade model developed independently over the following months.

---

## Background: How the Model Learns

To make this documentation easier to follow, here’s a quick overview of key machine learning terms used throughout.

- **Neural Network:** A system of mathematical “neurons” that learn patterns from examples. Each neuron performs a small computation, and millions of them combine to make predictions (similar to how the brain processes signals).

- **Training:** The process of showing the model many input–output pairs (for example, satellite images and their labeled masks). The model slowly adjusts its internal parameters to minimize mistakes.

- **Loss Function:** A mathematical way of measuring how wrong the model’s predictions are.  
  - A smaller *loss* means better performance.  
  - Different loss functions highlight different types of mistakes—for example, Cross-Entropy focuses on classification accuracy, while Lovasz improves boundary precision.

- **Optimizer (AdamW):** A method that tells the model how much to adjust itself after each mistake.  
  - The **learning rate** (often written as `lr`) controls how large these adjustments are. Too high, and training becomes unstable; too low, and it learns too slowly.

- **Scheduler:** Adjusts the learning rate during training. A **cosine schedule**, for example, starts high and gradually lowers the learning rate for smoother convergence.

- **Backbone:** The base network (like **ResNet-50**) that extracts general image features such as edges and textures before the segmentation layers classify each pixel.

- **Segmentation Model (DeepLabV3+):** A specialized neural network that assigns a label to every pixel in an image—turning satellite images into detailed color-coded maps.

- **Batch Size:** The number of image samples processed at once. Larger batches are more stable but need more GPU memory.

- **Augmentation:** Randomly altering input images (flipping, rotating, adding blur) to make the model robust against variations like lighting or camera angle.

- **mIoU (Mean Intersection-over-Union):** A standard metric for segmentation. It measures how much the predicted regions overlap with the true labeled regions.  
  - 1.0 means perfect overlap; 0.7 (70%) means strong agreement.

These ideas form the foundation for every design choice described below.

---

## Overview

**Objective:** Classify every pixel in post-disaster satellite images into 11 semantic categories, including water, roads, buildings (by damage level), vehicles, trees, and pools.  
This process, called **semantic segmentation**, helps convert raw satellite data into meaningful disaster-assessment maps.

**Dataset:** RescueNet (~22 GB ZIP, includes aerial images and color-coded masks)

**Final Model Performance**
- Validation mIoU: **0.724** (≈ 72.4% mean overlap accuracy)  
- Pixel Accuracy: **≈ 90%**

**Core Architecture:** DeepLabV3+ (ResNet-50 backbone)  
**Loss Functions:** Cross-Entropy, Lovasz, Focal CE, and OHEM  
**Optimizer:** AdamW (learning rate 1e-4, weight decay 1e-4)  
**Scheduler:** Cosine learning rate with warm-up  
**Sampler:** Rarity-aware (selects images with rare objects more often)  
**Precision:** AMP (Automatic Mixed Precision, uses 16- and 32-bit math for efficiency)  
**EMA:** Exponential Moving Average (stabilizes weights over time)  
**TTA:** Test-Time Augmentation (averages predictions from scaled and flipped inputs)

---

## Version 0 — BWSI Prototype (July 2025)

**Environment:** Google Colab Free Tier (T4/P100 GPU, 12 GB VRAM)

The first version was created during the Beaver Works Summer Institute’s *Remote Sensing for Disaster Response* course.  
The model was based on **DeepLabV3+** with a **ResNet-50** backbone pretrained on ImageNet.  
It attempted to label each pixel as one of 11 classes but lacked many stabilizing techniques that would come later.

**Configuration**
- Manual dataset unzip and loading  
- Basic augmentations (flips, color jitter)  
- Cross-Entropy loss (no class weights)  
- AdamW optimizer (lr 1e-4, wd 1e-4)  
- Batch 4, crop 512×512  

**Limitations**
- No RGB-to-ID mapping → color mismatch errors in masks  
- Rare classes underrepresented  
- No mixed precision or EMA  
- Frequent GPU memory errors  

**Result**
- Validation mIoU: 0.52–0.55  
- Pixel Accuracy: ≈ 83%  
- Detected large structures only (water, major buildings)

---

*(Subsequent sections continue as in your documentation — Versions 1–10, cumulative lessons, and future work — now with technical context established above.)*
