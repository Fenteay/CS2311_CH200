# CS2311_CH200 — Leaf Disease Segmentation

Phân đoạn vùng bệnh trên lá cây sử dụng **Học Liên Kết (FedAvg)** và **Thích Nghi Miền Tại Thời Điểm Kiểm Tra Không Cần Dữ Liệu Nguồn (TT-SFUDA)**.

## Kết quả tốt nhất

| Stage | Model | Dice | IoU |
|---|---|---|---|
| Stage 1 — Supervised | ResUNet (ResNet34 pretrained) | **0.8036** | 0.6916 |
| Stage 1 + TTA | ResUNet + 4-flip TTA | **0.8261** | 0.7104 |
| Stage 2 — FL | ResUNet FedAvg (10 rounds, 3 clients) | **0.8047** | — |
| Stage 3 — TT-SFUDA | Adapted trên leafandmask_trial | **0.7888** | 0.6710 |

---

## Yêu cầu hệ thống

- Python 3.10+
- CUDA 12.1+ (GPU NVIDIA, khuyến nghị ≥4GB VRAM)
- Windows 10/11 hoặc Linux

---

## Cài đặt thư viện

```bash
# PyTorch với CUDA 12.1
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Segmentation backbone (ResNet encoder + UNet decoder)
pip install segmentation-models-pytorch==0.5.0

# Vision encoder pretrained
pip install timm==1.0.26

# Augmentation
pip install albumentations==2.0.8

# Utilities
pip install tqdm pyyaml numpy opencv-python Pillow scikit-learn scipy

# Web demo (tuỳ chọn)
pip install flask
```

Hoặc cài tất cả một lần:

```bash
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install segmentation-models-pytorch==0.5.0 timm==1.0.26 albumentations==2.0.8 tqdm pyyaml numpy opencv-python Pillow scikit-learn scipy flask
```

---

## Cấu trúc thư mục

```
CS2311_CH200/
├── archs.py                  # Định nghĩa kiến trúc (UNet, ResUNet, ...)
├── dataset.py                # Dataset loader
├── losses.py                 # BCEDiceLoss, WeightedBCEDiceLoss, ...
├── metrics.py                # iou_score, dice_score
├── utils.py                  # AverageMeter, ...
├── train_source.py           # Stage 1: Huấn luyện giám sát
├── eval_tta.py               # Đánh giá với TTA
├── fl_simulate.py            # Stage 2: Mô phỏng Federated Learning
├── federated_flwr_common.py  # FedAvg logic
├── tt_sfuda_2d.py            # Stage 3: TT-SFUDA adaptation
├── models/                   # Chứa model weights đã train
│   ├── leafandmask_full_resunet/model.pth   # Kết quả Stage 1
│   └── fl_resunet_r10_e2/model.pth          # Kết quả Stage 2
├── inputs/inputs/            # Dữ liệu đầu vào (xem mục Dữ liệu bên dưới)
└── demo_web/app.py           # Web demo Flask
```

---

## Chuẩn bị dữ liệu

Cấu trúc thư mục dữ liệu:

```
inputs/inputs/
├── leafandmask_full/       # Tập dữ liệu nguồn + Client 1 FL
│   ├── train/
│   │   ├── images/         # 470 ảnh .jpg
│   │   └── masks/0/        # 470 masks .png
│   └── test/
│       ├── images/         # 118 ảnh .jpg
│       └── masks/0/
├── leaf_hrf_style/         # Client 2 FL (màu HRF)
│   └── train/images+masks/ # 470 ảnh
├── leaf_rite_style/        # Client 3 FL (màu RITE)
│   └── train/images+masks/ # 470 ảnh
└── leafandmask_trial/      # Miền đích TT-SFUDA
    ├── train/images/       # 80 ảnh không nhãn
    └── test/images+masks/  # 20 ảnh để đánh giá
```

---

## Stage 1 — Huấn luyện Giám sát (ResUNet)

Huấn luyện ResUNet với ResNet34 encoder pretrained trên `leafandmask_full`:

```bash
cd e:\Python\CS2311_CH200

python train_source.py \
  --dataset leafandmask_full \
  --arch ResUNet \
  --loss BCEDiceLoss \
  --lr 3e-4 \
  --batch_size 8 \
  --epochs 50 \
  --img-ext .jpg \
  --img-size 512
```

Kết quả lưu tại: `models/leafandmask_full_resunet/model.pth`  
**Dice = 0.8036, IoU = 0.6916**

### Đánh giá với TTA (Test-Time Augmentation)

```bash
python eval_tta.py \
  --model models/leafandmask_full_resunet/model.pth \
  --arch ResUNet \
  --dataset leafandmask_full \
  --img-ext .jpg \
  --tta
```

**Dice = 0.8261, IoU = 0.7104**

---

## Stage 2 — Học Liên Kết (Federated Learning)

FedAvg 10 vòng × 2 epoch cục bộ × 3 client, khởi tạo từ model Stage 1:

```bash
python fl_simulate.py \
  --init-weights models/leafandmask_full_resunet/model.pth \
  --arch ResUNet \
  --rounds 10 \
  --local-epochs 2 \
  --num-clients 3 \
  --datasets leafandmask_full,leaf_hrf_style,leaf_rite_style \
  --img-ext .jpg \
  --lr 1.5e-5 \
  --batch-size 8 \
  --output-dir models/fl_resunet_r10_e2
```

Kết quả lưu tại: `models/fl_resunet_r10_e2/model.pth`  
**Val Dice = 0.8047** (FL avg 3 domains: 0.7925)

---

## Stage 3 — TT-SFUDA Adaptation

Thích nghi mô hình FL sang miền đích `leafandmask_trial` không cần nhãn:

```bash
python tt_sfuda_2d.py \
  --source fl_resunet_r10_e2 \
  --target leafandmask_trial \
  --pseudo-thresh 0.65 \
  --adapt-lr-scale 0.05 \
  --stage1 15 \
  --stage2 15 \
  --const-loss-weight 0.001
```

| Tham số | Ý nghĩa | Giá trị tốt nhất |
|---|---|---|
| `--source` | Tên thư mục model nguồn trong `models/` | `fl_resunet_r10_e2` |
| `--target` | Tên dataset đích trong `inputs/inputs/` | `leafandmask_trial` |
| `--pseudo-thresh` | Ngưỡng tin cậy pseudo-label (τ) | `0.65` |
| `--adapt-lr-scale` | Learning rate = base_lr × scale | `0.05` |
| `--stage1` | Số epoch Stage I (pseudo-label adaptation) | `15` |
| `--stage2` | Số epoch Stage II (EMA refinement) | `15` |
| `--const-loss-weight` | Trọng số consistency loss (λ_c) | `0.001` |

Kết quả lưu tại: `adapted_target_model_leafandmask_trial.pth`  
Masks dự đoán: `results_leafandmask_trial_masks/`  
**Adapted Dice = 0.7888**

---

## Web Demo (tuỳ chọn)

```bash
cd e:\Python\CS2311_CH200\demo_web
python app.py
```

Mở trình duyệt tại `http://localhost:5000`

---

## Lưu ý

- `--const-loss-weight 0.001` quan trọng: giá trị cao hơn (mặc định 1.0) gây **catastrophic collapse** (Dice → 0.14)
- FL simulation chạy in-process (không cần Docker), tuần tự từng client
- TT-SFUDA dùng unlabeled train set (`inputs/inputs/<target>/train/`) và evaluate trên test set
