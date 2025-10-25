# RescueNet Segmentation – Full Development Timeline

This README documents every stage of the RescueNet segmentation model's development — from the initial BWSI baseline through the ten major revisions leading up to October 10, 2025. Each version includes the **date**, **main features**, **changes from previous version**, and **outcomes**.

---

## 📘 Overview
- **Goal:** Classify each pixel in post-disaster aerial imagery into one of 11 classes (water, buildings by damage level, vehicles, roads, etc.).  
- **Dataset:** RescueNet (~22 GB).  
- **Metric:** Mean Intersection-over-Union (mIoU) — overlap ratio between predicted vs. ground truth regions.  
- **Final performance:** 0.724 validation mIoU (~90% pixel accuracy).

---

## 🧩 Model Version Timeline

### 🧠 Version 0 — BWSI Prototype (July 2025)
- **Environment:** Free Colab (T4/P100), no Pro GPU.
- **Model:** DeepLabV3+ (ResNet-50), CE loss only.
- **Mask handling:** Direct RGB; no color lookup table.
- **Augmentations:** Simple flips/jitter; batch size 4.
- **Outcome:** mIoU ≈ 0.52–0.55; struggled on rare classes.
- **Lessons:** Needed proper RGB→ID mapping, weighted losses, and AMP.

---

### 🧠 Version 1 — Dataset Pipeline Rebuild (Aug 10, 2025)
- Added **Dropbox→Drive automation** for 22 GB dataset.
- Implemented **256³ RGB→ID LUT** and `IGNORE_INDEX=255`.
- Introduced **PadIfNeededConst** to maintain perfect mask alignment.
- Verified mask pairing by stem name.
- **No training yet** — focused on reproducibility foundation.

---

### ⚙️ Version 2 — First Stable Training Pipeline (Aug 17, 2025)
- Added **DeepLabV3+ (ResNet-50)** and **PSPNet (EffB3)** branches.
- Weighted CE + Lovasz + Tversky losses.
- Cosine LR scheduler, 1-epoch warm-up, AMP (fp16), and EMA (0.999).
- Batch = 32, crop 768×768, grad accumulation ×2.
- **Results:** mIoU ≈ 0.68–0.70.
- **Issues:** Underfit on rare classes, imbalance persisted.

---

### ⚙️ Version 3 — Rarity-Aware Sampling + Hybrid Loss (Aug 24, 2025)
- Introduced **rarity-aware sampler v2**: presence + area rarity (γ=1.1, λ=0.9).
- Upgraded loss mix: `0.35CE + 0.30Lovasz + 0.20FocalCE + 0.15OHEM`.
- Gradient clipping (1.0), AMP(bf16), channels_last memory format.
- **Results:** mIoU ↑ to 0.714, acc ≈ 89%.
- **Breakthrough:** Major boost for rare classes (e.g., road-blocked).

---

### 🧮 Version 4 — Augmentation Refinement (Early Sept 2025)
- Streamlined Albumentations: Longest/SmallestMaxSize → PadIfNeededConst → CropNonEmptyMaskIfExists.
- Tuned **CoarseDropout** and **MotionBlur**.
- Single normalization step (ImageNet mean/std).
- **Outcome:** More stable convergence, fewer NaNs.

---

### 📊 Version 5 — Evaluation & Visualization Suite (Mid Sept 2025)
- Added **per-class IoU**, **FG/BG IoU**, and **confusion matrix** metrics.
- Implemented **TTA** (scales [0.85, 1.0, 1.15] + flip averaging).
- Introduced `demo_predict_and_show()` for visual diagnostics.
- **Result:** Full eval suite; validated generalization visually.

---

### 💾 Version 6 — Resource Management + Stability (Late Sept 2025)
- Added **VRAM/RAM monitors** via `psutil`.
- **Safe checkpoint loading** with `weights_only=False` fallback.
- Implemented `torch.serialization.add_safe_globals()` fix.
- **Result:** Zero Colab crashes; reproducible multi-session runs.

---

### ⚙️ Version 7 — PSPNet Branch Tuning (Early Oct 2025)
- PSPNet (EfficientNet-B3) fine-tuning with Tversky + ReduceLROnPlateau.
- Batch 16, 1024×1024 crops, channels_last optimization.
- **Result:** Smoother textures, slightly lower mIoU (~0.67).

---

### 📂 Version 8 — Checkpoint Management (Oct 8, 2025)
- Versioned model saves (`v13_deeplabv3p_ema_e{epoch}.pth`).
- Maintained Drive sync for every “best” checkpoint.
- **Result:** Consistent rollback capability and tracking.

---

### 🔍 Version 9 — Safe Inference + Visualization Cells (Oct 9, 2025)
- Isolated **inference-only** pipeline with exact val transforms.
- Sanity-checked batch shapes, verified reproducibility.
- Integrated optional post-processing (remove small components).

---

### 🏁 Version 10 — Final Training & Integration (Oct 10, 2025)
- Unified DeepLabV3+ with rarity-aware sampler, hybrid loss, EMA, and TTA.
- Harmonized all Albumentations versions and transforms.
- Final evaluation metrics and confusion matrix generation.
- **Best model:** DeepLabV3+ (ResNet-50, EMA)  
  - **val mIoU = 0.724**, **pixel accuracy ≈ 90%**  
  - Water/buildings IoU ≈ 0.82; vehicles/roads ≈ 0.45–0.55.

---

## 📈 Summary Table

| Version | Date | Major Additions | Result |
|----------|------|------------------|---------|
| 0 | Jul 2025 | BWSI baseline (CE only) | 0.52 mIoU |
| 1 | Aug 10 | Data pipeline rebuild (LUT, PadIfNeededConst) | – |
| 2 | Aug 17 | Weighted CE, Lovasz, EMA, AMP | 0.68–0.70 |
| 3 | Aug 24 | Rarity sampler + hybrid loss | **0.714** |
| 4 | Sep | Transform cleanup | Stable |
| 5 | Sep | Eval + TTA + Viz suite | Diagnostic ready |
| 6 | Sep | Memory stability | Robust training |
| 7 | Oct | PSPNet EffB3 tuning | 0.67 |
| 8 | Oct 8 | Checkpointing discipline | Reproducible |
| 9 | Oct 9 | Safe inference cells | Consistent viz |
| 10 | Oct 10 | Final training | **0.724 / 90% acc** |

---

**Final takeaway:**  
The project evolved from a lightweight educational baseline into a fully optimized, research-grade segmentation pipeline with intelligent sampling, multi-loss synergy, and production-level stability.
