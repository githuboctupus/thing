# 🛰️ RescueNet Segmentation Project — Full Development History (BWSI → Oct 2025)

This README documents the complete evolution of my **RescueNet aerial-imagery segmentation model**, from the first BWSI prototype to the tenth (and final) independent version in October 2025.  
Each version lists what was **added**, **removed**, and **kept**, along with technical explanations and results (mIoU = mean Intersection-over-Union, higher = better).

---

## 🌍 Overview
**Goal:** Classify every pixel in post-disaster satellite images into 11 semantic classes  
(water, roads, buildings by damage level, vehicles, trees, pools, etc.).

**Dataset:** RescueNet (~22 GB ZIP, color masks)

**Final Performance:**  
- **Validation mIoU = 0.724** (≈ 72.4 % mean overlap accuracy)  
- **Pixel Accuracy ≈ 90 %**

**Core architecture:** `DeepLabV3+ (ResNet-50)`  
**Loss family:** CE + Lovasz + FocalCE + OHEM  
**Optimizer:** AdamW (1e-4, wd 1e-4)  
**Scheduler:** Cosine LR with warm-up  
**Sampler:** Rarity-aware (presence + area rarity)  
**Precision:** AMP (bf16 preferred)  
**EMA:** 0.999 decay  
**TTA:** multi-scale [0.85, 1.0, 1.15] + horizontal flip  

---

## 🧩 Version 0 — BWSI Prototype (July 2025)

**Environment:** Free Colab (T4/P100, 12 GB VRAM)

### Added / Kept
- Dataset unzip → `/content/`
- `DeepLabV3Plus(resnet50, imagenet)`  
- Basic augmentations (H/V flips, color jitter)
- CE loss (no weights)
- AdamW lr 1e-4 / wd 1e-4
- Batch 4, crop 512×512

### Missing / Limitations
- No RGB→ID LUT → mask color mismatch noise  
- No weighted loss → rare classes ignored  
- No AMP, EMA, or LR schedule → unstable training  
- Limited context from small crops

### Result
- **val mIoU ≈ 0.52 – 0.55**
- **acc ≈ 83 %**
- Learned large structures (water/buildings) only  
- Frequent NaNs / disconnections due to VRAM limit

---

## 🧠 Version 1 — Dataset Pipeline Rebuild (Aug 10 2025)

### Added
- Automated Dropbox → Drive download and cache (22 GB)  
- Full **256³ RGB→Class ID LUT** for exact mapping  
- `IGNORE_INDEX = 255` for invalid pixels  
- `PadIfNeededConst` to pad images and masks together  
- File pairing by filename stem + warnings for mismatch  

### Kept
- DeepLabV3+ baseline architecture  
- Basic augmentations framework (Albumentations)

### Removed
- Raw RGB mask reads (replaced with LUT conversion)  
- Manual unzip each session

### Outcome
- **Reproducible dataset pipeline**
- No training yet (laid foundation)

---

## ⚙️ Version 2 — First Stable Training (Aug 17 2025)

### Added
- **Two branches:** DLv3+ (ResNet-50) & PSPNet (EfficientNet-B3)  
- **Loss mix:** Weighted CE (1/√freq) + Lovasz + Tversky (α 0.6, β 0.4)  
- **Cosine LR schedule** + 1 epoch warm-up  
- **AMP (fp16)** with GradScaler  
- **EMA (0.999)** shadow weights  
- Batch 32 (grad accum ×2), crop 768×768  
- **Validation loop + confusion matrix prototype**

### Kept
- LUT mask conversion and PadIfNeededConst  

### Removed
- Plain random sampler (no class balance)

### Result
- **mIoU ≈ 0.68 – 0.70**
- Smoother training, stable losses  
- Still biased toward frequent classes  

---

## ⚙️ Version 3 — Rarity-Aware Sampler + Hybrid Loss (Aug 24 2025)

### Added
- **Sampler v2:** rarity = presenceᵞ × areaˡᵃᵐᵇᵈᵃ  
  - γ = 1.1, λ = 0.9, clamp [0.25, 6]  
- **Hybrid Loss v2:**  
  - 0.35 CE (weighted)  
  - 0.30 Lovasz  
  - 0.20 Focal CE (per-class γ)  
  - 0.15 OHEM (top 20 % hard pixels)  
- **Gradient clip = 1.0**  
- **AMP (bf16)** + `channels_last` format  

### Kept
- EMA, Cosine LR, Weighted CE  

### Removed
- Tversky for DLv3+ (mainly kept in PSP)  
- Over-aggressive transforms causing mask drift  

### Result
- **mIoU = 0.714**, **acc ≈ 89 %**  
- Rare classes (+10–12 IoU pts)  
- Stable validation curves  

---

## 🧮 Version 4 — Augmentation & Normalization Refinement (≈ Aug 30 2025)

### Added
- Unified transform sequence:  
  `LongestMaxSize → SmallestMaxSize → PadIfNeededConst → CropNonEmptyMaskIfExists → Normalize`  
- Tuned **CoarseDropout** and **MotionBlur**  
- Enforced **single Normalize** call (ImageNet mean/std)  

### Kept
- Hybrid loss and rarity sampler  

### Removed
- Duplicate resize transforms  
- Random rotation with mask mismatch  

### Result
- Fewer NaNs, cleaner edges, stable epoch-to-epoch IoU  

---

## 📊 Version 5 — Evaluation & Visualization Suite (≈ Sep 9 2025)

### Added
- **Per-class IoU, accuracy, FG/BG IoU**  
- **Confusion matrix heatmap**  
- **TTA:** scales [0.85, 1.0, 1.15] + flip average  
- `demo_predict_and_show()` (overlay input/pred/GT)  
- Optional post-filter `remove_small_components()`  

### Kept
- EMA weights, rarity sampler  

### Removed
- Outdated metric scripts  

### Result
- Visual and numeric evaluation aligned  
- Confirmed model generalization  

---

## 💾 Version 6 — Resource Management & Stability (Late Sept 2025)

### Added
- `psutil` RAM/VRAM logging  
- `torch.cuda.mem_get_info()` tracker  
- `torch.serialization.add_safe_globals()` fix for safe loads  
- Fallback `torch.load(..., weights_only=False)`  

### Kept
- Eval suite and EMA  

### Removed
- Excess debug printing  
- Old checkpoint paths causing drive conflicts  

### Result
- No more Colab OOM/crashes  
- Multi-session reproducibility  

---

## ⚙️ Version 7 — PSPNet Branch Fine-Tuning (Early Oct 2025)

### Added
- PSPNet (EfficientNet-B3) branch with `channels_last`  
- Loss = 0.6 CE + 0.4 Tversky  
- Scheduler = ReduceLROnPlateau  
- Batch 16, crop 1024×1024  

### Kept
- DLv3+ mainline for comparison  

### Removed
- Extra EMA tracking to save VRAM  

### Result
- mIoU ≈ 0.67 (softer textures, less edge precision)  
- Retained for potential ensemble use  

---

## 📂 Version 8 — Checkpoint Management (Oct 8 2025)

### Added
- Versioned file names: `v13_deeplabv3p_ema_e{epoch}.pth`  
- Drive sync after each “best” save  
- Timestamped log headers  

### Kept
- EMA and TTA evaluation  

### Removed
- Old naming (“bestprev”) files  

### Result
- Easy rollback and checkpoint diff tracking  

---

## 🔍 Version 9 — Safe Inference and Visualization (Oct 9 2025)

### Added
- Dedicated inference-only cells  
- Exact val transforms for post-training prediction  
- Shape assertions and warnings  
- Optional small-component cleanup  

### Kept
- Finalized DLv3+ architecture  

### Removed
- Randomized augmentations (deterministic inference)  

### Result
- Fully reproducible evaluation outputs  
- Clean overlay visuals for reports  

---

## 🏁 Version 10 — Final Training & Integration (Oct 10 2025)

### Added
- Unified pipeline (DLv3+ + rarity sampler + hybrid loss + EMA + TTA)  
- Harmonized Albumentations API calls with try/except fallbacks  
- Integrated confusion matrix export and class IoU summary  
- GPU utilization monitor and save-to-Drive automation  

### Kept
- Best model config from v3–v9  

### Removed
- Experimental PSPNet branch from mainline  
- Duplicate eval cells  

### Final Results
| Metric | Value |
|---------|-------|
| **Validation mIoU** | **0.724** |
| **Pixel Accuracy** | **≈ 90 %** |
| **Strong classes** | Water & Buildings (no/medium damage ≈ 0.82 IoU) |
| **Weaker classes** | Vehicles/Road-blocked ≈ 0.45–0.55 IoU |

---

## 🧠 Cumulative Lessons

1. **Sampler engineering > architecture swaps.**  
   Balancing data sampling had the largest impact (+0.04 mIoU).
2. **EMA and mixed precision** stabilized metrics and reduced VRAM use.  
3. **Lovasz + FocalCE** improved edges and rare regions jointly.  
4. **Visualization and confusion matrix tooling** made errors intuitive to spot.  
5. **Versioned checkpoints and LUT-based dataset handling** enabled true reproducibility.

---

## 📈 Summary Table

| Ver | Date | Major Additions | Removed / Changed | mIoU / Acc |
|------|------|-----------------|-------------------|------------|
| 0 | Jul 2025 | Baseline DLv3+ CE only | – | 0.52 / 83 % |
| 1 | Aug 10 | LUT + PadIfNeededConst | Raw RGB masks | – |
| 2 | Aug 17 | Weighted CE + Lovasz + EMA | Random sampler | 0.68–0.70 |
| 3 | Aug 24 | Rarity sampler + Hybrid Loss v2 | Tversky (DLv3+) | 0.714 / 89 % |
| 4 | Aug 30 | Transform cleanup | Duplicate resize | Stable |
| 5 | Sep 9 | Eval suite + TTA | Old metrics | Diag ready |
| 6 | Late Sep | Safe load + mem monitor | Debug spam | Robust |
| 7 | Oct 2 | PSPNet EffB3 branch | Extra EMA | 0.67 |
| 8 | Oct 8 | Checkpoint discipline | Legacy names | Stable |
| 9 | Oct 9 | Safe inference cells | Random augs | Clean viz |
| 10 | Oct 10 | Final integration | PSP branch | **0.724 / 90 %** |

---

## 🔮 Future Work
- Try SegFormer B2/B4 encoders for real-time inference.  
- Semi-supervised consistency training on unlabeled imagery.  
- Model distillation to lightweight student network.  
- Add boundary-aware loss (SoftDice / Boundary IoU).  
- Port to PyTorch Lightning for multi-GPU experiments.

---

**Final Note:**  
Across ten versions, the project matured from a fragile educational demo to a research-grade segmentation system with balanced sampling, robust training loops, and reliable evaluation infrastructure.

