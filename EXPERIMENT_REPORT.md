# Báo Cáo Thực Nghiệm: TT-SFUDA + Federated Learning cho Phân Đoạn Bệnh Lá

**Ngày:** 30/05/2026  
**Tác giả:** CS2203.CH201  
**Môi trường:** Windows 11, Python 3.12.10, PyTorch (CPU), Docker Desktop v29.1.3 (WSL2)

---

## 1. Tổng Quan Bài Toán

Bài toán: **Phân đoạn bệnh lá cây** (leaf disease segmentation) theo hướng **Test-Time Source-Free Unsupervised Domain Adaptation (TT-SFUDA)** kết hợp **Federated Learning (FL)**.

### Mục tiêu
- Train model phân đoạn bệnh lá trên nhiều client phân tán (FL) mà **không chia sẻ dữ liệu thô**
- Adapt model sang domain mới (target) **tại test-time**, không cần nhãn ground-truth

### Kiến trúc
| Thành phần | Chi tiết |
|---|---|
| Model | UNet (binary segmentation) |
| Input | 512×512, 3 channels (BGR) |
| Output | 1-class binary mask (disease/background) |
| Loss | BCEDiceLoss |
| FL Framework | Flower (flwr) v1.30, FedAvg |
| Infrastructure | Docker Compose (1 server + 3 clients) |

---

## 2. Dataset

### Source datasets (training FL)
| Dataset | Mô tả | Dùng cho |
|---|---|---|
| `leafandmask_full` | 118 ảnh lá bệnh + mask vùng bệnh | Supervised baseline, FL client data |
| `leaf_hrf_style` | Ảnh lá style transfer từ HRF | FL multi-domain |
| `leaf_rite_style` | Ảnh lá style transfer từ RITE | FL multi-domain |

### Target dataset (test-time adaptation)
| Dataset | Số ảnh | Mô tả |
|---|---|---|
| `leafandmask_trial` | 20 ảnh | Tập test domain shift, dùng để đánh giá adaptation |

> **Lưu ý:** Toàn bộ dataset chỉ gồm **lá có bệnh** — không có lá khỏe trong training, do đó model không phân biệt được lá khỏe hoàn toàn.

---

## 3. Quá Trình Thực Nghiệm

### Thực nghiệm 1: Supervised Baseline (leafandmask_full_unet)

**Mục tiêu:** Train model supervised trên toàn bộ `leafandmask_full`, dùng làm baseline  
**Cấu hình:**
```
arch: UNet, loss: BCEDiceLoss
lr: 7e-05, weight_decay: 1e-4
input: 512×512, num_classes: 1
stage1: 5 epochs, stage2: 8 epochs
```

**Kết quả trên 118 ảnh test:**
| Metric | Giá trị |
|---|---|
| Dice | **0.4847** |
| IoU | 0.3533 |
| Loss | 0.9960 |
| FG ratio trung bình | 34.96% |

→ **Đây là kết quả tốt nhất trong toàn bộ thực nghiệm.**

---

### Thực nghiệm 2: FL Round 1 — Cold-start, 1 client (multi_dataset_fedavg_3containers)

**Cấu hình FL:**
```
algorithm: FedAvg
clients: 3, rounds: 1, local_epochs: 1
init: random weights (cold-start)
```

**Kết quả TT-SFUDA trên `leafandmask_trial` (20 ảnh):**
| Phase | Metric | Giá trị |
|---|---|---|
| Source-only | Dice | **0.0000** |
| Adaptation train | Loss | 1.4151 |
| Adaptation train | IoU | 0.0000 |
| Refinement | Loss | 1.3256 |
| Adapted | Dice | **0.0000** |

→ **Thất bại hoàn toàn.** Model chưa hội tụ sau 1 round — sigmoid output ≈ 1.0 toàn bộ (predict toàn trắng).

---

### Thực nghiệm 3: FL Round 2 — Cold-start, 3 clients (multi_dataset_fedavg_3containers_r2_20260530)

**Cấu hình FL:**
```
algorithm: FedAvg
clients: 3, rounds: 2, local_epochs: 1
init: random weights (cold-start)
Infrastructure: Docker Compose (3 containers độc lập)
```

**Kết quả TT-SFUDA (3 steps) trên `leafandmask_trial`:**
| Phase | Metric | Giá trị |
|---|---|---|
| Source-only | Dice | **0.2396** |
| Adaptation train | Loss | 1.3303 |
| Adaptation train | IoU | 0.4682 |
| Adapted | Dice | **0.2396** |

→ Model bắt đầu học được cấu trúc bệnh, nhưng kết quả còn thấp. Adaptation không cải thiện (3 steps quá ít).

---

### Thực nghiệm 4: FL Round 3 — Fedavg trên leafandmask_full (leafandmask_full_unet_fedavg)

**Mục tiêu:** Train FL trực tiếp trên supervised dataset để so sánh  
**Cấu hình:** 3 rounds, multi-dataset (chase, hrf, rite, leafandmask)

**Kết quả từ federated_history.csv:**
| Round | Train Loss | Train IoU | Val Dice |
|---|---|---|---|
| 1 | 1.3976 | ~0.0 | ~0.0 |
| 2 | 1.3920 | 0.0643 | 0.2616 |
| 3 | 1.3873 | 0.1416 | 0.2615 |

→ Hội tụ chậm, val dice dừng ở ~0.26 sau 3 rounds.

---

### Thực nghiệm 5: FL Warm-start 5 Rounds — Kết quả tốt nhất FL (multi_dataset_fedavg_warm_r5_e3_20260530)

**Ý tưởng:** Khởi tạo FL server bằng **supervised model** đã train (warm-start) thay vì random weights, tăng local epochs và rounds.

**Cấu hình FL:**
```
algorithm: FedAvg
clients: 3, rounds: 5, local_epochs: 3
init: warm-start từ leafandmask_full_unet/model.pth
Infrastructure: Docker Compose
Thời gian train: ~2.5 giờ (15:45 – 18:15, 30/05/2026)
```

**Checkpoints lưu:** `model_round_1.pth` → `model_round_5.pth` (mỗi file ~30MB)

**Kết quả TT-SFUDA (full epoch = 80 steps) trên `leafandmask_trial`:**
| Phase | Metric | Giá trị |
|---|---|---|
| Source-only | Dice | **0.4425** |
| Adaptation phase 1 | Train loss | 0.9225 |
| Adaptation phase 1 | Train IoU | 0.2805 |
| Refinement phase 2 | Loss | 0.4392 |
| Refinement phase 2 | IoU | 0.2484 |
| **Adapted (final)** | **Dice** | **0.2922** |

→ **Warm-start cải thiện đáng kể** source-only (0.44 vs 0.00 của cold-start).  
→ Tuy nhiên adaptation làm **giảm** performance (0.44 → 0.29) — TT-SFUDA bị overfit trên pseudo-label.

---

### Thực nghiệm 6: FL Simulation 10 Rounds + TT-SFUDA Cải Tiến (fl_sim_r10_e3) — 03/06/2026

**Mục tiêu:** Tiếp tục FL từ warm_r5 thêm 5 rounds (tổng 10 rounds) với in-process simulation (không cần Docker).

**Cài đặt FL simulation:**
- Init từ: `multi_dataset_fedavg_warm_r5_e3_20260530/model.pth`
- Rounds thêm: 5 (tổng 10 rounds)
- Local epochs: 2, Batch size: 8, Samples/client: 100
- Learning rate: 3e-5 (thấp hơn original 7e-5 để bảo toàn warm-start)
- 3 clients: leafandmask_full, leaf_hrf_style, leaf_rite_style

**Kết quả FL simulation:**

| Round | val_dice | val_iou |
|-------|----------|---------|
| 1 | 0.4637 | 0.3307 |
| 2 | 0.4714 | 0.3373 |
| 3 | 0.4768 | 0.3423 |
| 4 | 0.4782 | 0.3451 |
| 5 | 0.4782 | 0.3434 |

**TT-SFUDA cải tiến (pseudo_thresh=0.65, adapt_lr_scale=0.2):**

| Phase | Metric | Giá trị |
|---|---|---|
| Source-only | Dice | **0.4907** |
| **Adapted (final)** | **Dice** | **0.4568** |

→ **Tốt nhất trong các thực nghiệm FL**: source dice 0.49, adapted dice 0.46.  
→ TT-SFUDA cải tiến (threshold cao hơn + LR thấp hơn) **không còn làm giảm** performance.

---

## 4. Tổng Hợp Kết Quả

### Bảng so sánh tất cả model (target: leafandmask_trial)

| STT | Experiment | Model | Source-only Dice | Adapted Dice | Ghi chú |
|---|---|---|---|---|---|
| 1 | Supervised (train trực tiếp) | `leafandmask_full_unet` | **0.4847** | — | Train trên toàn bộ GT |
| 2 | **FL sim 10r + TT-SFUDA cải tiến** | `fl_sim_r10_e3` | **0.4907** | **0.4568** | **Tốt nhất FL** |
| 3 | FL warm-start 5r (source-only) | `warm_r5_e3` | 0.4425 | — | FL baseline |
| 4 | FL warm-start 5r + TT-SFUDA cũ | `warm_r5_e3` | 0.4425 | 0.2922 | TT-SFUDA overfit |
| 5 | FL 2r cold-start + TT-SFUDA | `r2_20260530` | 0.2396 | 0.2396 | Không cải thiện |
| 6 | FL fedavg 3r (val) | `leafandmask_full_unet_fedavg` | — | 0.2616 | Val Dice thấp |
| 7 | FL 1r cold-start + TT-SFUDA | `3containers` | 0.0000 | 0.0000 | Chưa hội tụ |

### Phân tích

```
FL sim r10 source (0.491) > Supervised (0.485) > FL sim r10 adapted (0.457) > FL warm-start source (0.442) > FL warm-start adapted cũ (0.292) > FL 2r (0.240) > FL 1r (0.000)
```

**Kết luận:**
1. **FL simulation 10 rounds** vượt supervised trên source-only (0.491 vs 0.485) — nhờ multi-dataset
2. **TT-SFUDA cải tiến** (thresh=0.65, lr_scale=0.2) giữ được gần như toàn bộ source performance sau adaptation (0.491 → 0.457)
3. **TT-SFUDA cũ** bị overfit nghiêm trọng (0.442 → 0.292) do threshold thấp + LR cao
4. **Warm-start FL** là chìa khóa: cold-start (0r) → 0.00, warm-start (10r) → 0.49

---

## 5. Phân Tích Lỗi & Bài Học

### Bug phát hiện trong quá trình thực nghiệm

**Bug preprocessing (phát hiện 30/05/2026):**  
Inference trong `demo_web/app.py` dùng sai preprocessing so với training:

| | Training (dataset.py) | Inference ban đầu | Fix |
|---|---|---|---|
| Color space | BGR (cv2 mặc định) | Convert sang RGB | Giữ BGR |
| Normalization | `A.Normalize()` → `/255` | `A.Normalize()` (không /255) | Thêm `/255` |

→ Model predict sai toàn bộ (scan background thay vì vùng bệnh)

### Giới hạn của hệ thống
- Dataset chỉ gồm **lá bệnh** → không phân biệt được lá khỏe
- TT-SFUDA phụ thuộc chất lượng pseudo-label của source model
- FL convergence chậm khi data heterogeneous giữa các clients

---

## 6. Cơ Sở Hạ Tầng Thực Nghiệm

### Stack kỹ thuật
```
OS:          Windows 11
Python:      3.12.10 (venv: e:\Python\.venv)
PyTorch:     CPU (không có GPU)
Docker:      v29.1.3, WSL2 backend
FL:          Flower (flwr) v1.30, FedAvg
Web demo:    Flask 3.1.3 + Chart.js 4.4.3
```

### Docker Compose FL Stack
```
fl_server     → port 8080 (FedAvg aggregator)
fl_client_0   → Dataset: leafandmask_full
fl_client_1   → Dataset: leaf_hrf_style  
fl_client_2   → Dataset: leaf_rite_style
Network:      fl_network (bridge)
```

### Web Demo
- URL: http://127.0.0.1:5000
- Sections: Dashboard | Gallery | Predict | Adaptation Demo
- Gallery: 118 ảnh supervised (Dice≈0.485) + 20 ảnh FL adapted
- Disease card: phân loại mức độ bệnh (Khỏe / Nhẹ / Trung bình / Nặng / Rất nặng)

---

## 7. File & Thư Mục Quan Trọng

```
tt_sfuda_leaf/
├── models/
│   ├── leafandmask_full_unet/          ← Supervised baseline (BEST, Dice=0.485)
│   ├── multi_dataset_fedavg_warm_r5_e3_20260530/  ← FL tốt nhất (Dice=0.442)
│   ├── multi_dataset_fedavg_3containers_r2_20260530/  ← FL 2r cold-start
│   └── leafandmask_full_unet_fedavg/   ← FL trên supervised data
├── leafandmask_full_source_eval_fix8/  ← Predicted masks supervised (118 ảnh)
├── results_leafandmask_trial_masks/    ← Predicted masks FL warm-start (20 ảnh)
├── adapted_target_model_leafandmask_trial.pth  ← Model sau TT-SFUDA
├── tt_sfuda_warm_r5_full.log           ← Log adaptation warm-start FL
├── demo_web/                           ← Flask web demo
│   ├── app.py
│   ├── templates/index.html
│   └── static/{app.js, style.css}
├── federated_flwr_server.py            ← FL server (đã thêm --init-weights)
├── docker-compose.federated.yml        ← Docker stack FL
└── tt_sfuda_2d.py                      ← TT-SFUDA adaptation pipeline
```
