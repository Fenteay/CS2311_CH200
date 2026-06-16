"""
TT-SFUDA + Federated Learning — Web Demo
Endpoints:
  GET  /                     → main dashboard
  GET  /api/models            → list available source models
  GET  /api/fl_metrics        → FL training history (all runs)
  POST /api/predict           → predict segmentation on uploaded image
  POST /api/adapt_and_predict → adapt on uploaded image then predict
"""

import os
import io
import sys
import json
import yaml
import base64
import traceback
from pathlib import Path
from glob import glob

import numpy as np
import cv2
from PIL import Image

# Flask
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Torch — loaded lazily to avoid slow startup on first request
import torch
import albumentations as A

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent   # tt_sfuda_leaf/
sys.path.insert(0, str(ROOT))

import archs
import losses
from metrics import iou_score
from utils import AverageMeter

# ── app setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_DIR   = ROOT / "models"
ADAPTED_DIR  = ROOT
UPLOAD_DIR   = ROOT / "demo_web" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Gallery config: target_name → paths ──────────────────────────────────────
GALLERY_CONFIG = {
    "leafandmask_full": {
        "img_dir":     ROOT / "inputs" / "inputs" / "leafandmask_full" / "test" / "images",
        "gt_dir":      ROOT / "inputs" / "inputs" / "leafandmask_full" / "test" / "masks" / "0",
        "pred_dir":    ROOT / "results" / "leafandmask_full_unet_source_eval",
        "pred_prefix": "leafandmask_full__",   # train_source saves as <prefix>__<id>.jpg
        "pred_ext":    ".jpg",
        "compare_dir": None,
        "metrics_file":ROOT / "results" / "leafandmask_full_unet_source_eval" / "metrics.txt",
        "label": "leafandmask_full (Supervised UNet, Dice=0.5879)",
    },
    "leafandmask_trial": {
        "img_dir":     ROOT / "inputs" / "inputs" / "leafandmask_trial" / "test" / "images",
        "gt_dir":      ROOT / "inputs" / "inputs" / "leafandmask_trial" / "test" / "masks" / "0",
        "pred_dir":    ROOT / "results_leafandmask_trial_masks",
        "pred_prefix": "",
        "pred_ext":    ".png",
        "compare_dir": None,
        "metrics_file":None,
        "label": "leafandmask_trial (FL hot-start 10r + TT-SFUDA, Source=0.5732, Adapted=0.5539)",
    },
}

# ── helpers ───────────────────────────────────────────────────────────────────

def _load_model(arch: str, num_classes: int, input_channels: int,
                deep_supervision: bool, weights_path: str):
    model = archs.__dict__[arch](num_classes, input_channels, deep_supervision)
    state = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _preprocess(img_bgr: np.ndarray, h: int = 512, w: int = 512) -> torch.Tensor:
    # Must match dataset.py training pipeline exactly:
    # 1) keep BGR (cv2 default, same as training)
    # 2) A.Normalize() — then divide by 255 again (dataset.py line: img /= 255)
    transform = A.Compose([
        A.Resize(h, w),
        A.Normalize(),
    ])
    aug = transform(image=img_bgr)["image"]   # BGR, same as training
    arr = aug.astype("float32") / 255.0       # matches dataset.py post-processing
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor


def _predict_tensor(model, tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        out = model(tensor.to(device))
        if isinstance(out, (list, tuple)):
            out = out[0]
        prob = torch.sigmoid(out).squeeze().cpu().numpy()
    mask = (prob >= 0.5).astype(np.uint8) * 255
    return mask, prob


def _ndarray_to_b64png(arr: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", arr)
    return base64.b64encode(buf.tobytes()).decode()


def _overlay(img_bgr: np.ndarray, mask: np.ndarray,
             color=(0, 255, 0), alpha: float = 0.45) -> np.ndarray:
    overlay = img_bgr.copy()
    green_layer = np.zeros_like(img_bgr)
    green_layer[mask > 0] = color
    cv2.addWeighted(green_layer, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay


def _list_source_models():
    models = []
    for p in sorted(MODELS_DIR.iterdir()):
        if not p.is_dir():
            continue
        model_pth = p / "model.pth"
        if not model_pth.exists():
            continue
        configs = list(p.glob("config_*.yml"))

        # Determine model type
        is_fl = any(k in p.name for k in ["fedavg", "federated", "containers", "fl_", "_fl_"])
        if is_fl:
            # Try to read rounds from config federated block
            rounds = _get_rounds_from_config(configs[0]) if configs else None
            mtype  = f"FL ({rounds}r)" if rounds else "FL"
        else:
            mtype = "Supervised"

        models.append({
            "name":    p.name,
            "type":    mtype,
            "configs": [c.stem.replace("config_", "") for c in configs],
            "rounds":  _get_rounds_from_config(configs[0]) if configs and is_fl else None,
        })

    # Sort: supervised first, then FL by rounds desc
    models.sort(key=lambda m: (0 if m["type"] == "Supervised" else 1,
                                -(m["rounds"] or 0)))
    return models


def _get_rounds_from_config(config_path: Path):
    try:
        with open(config_path, encoding="utf-8-sig") as f:
            cfg = yaml.safe_load(f)
        fed = cfg.get("federated", {})
        return fed.get("rounds")
    except Exception:
        return None


def _collect_fl_metrics():
    """Scan all model dirs and build timeline of FL training metrics."""
    runs = []
    for p in sorted(MODELS_DIR.iterdir()):
        if not p.is_dir():
            continue
        model_pth = p / "model.pth"
        if not model_pth.exists():
            continue
        # try to read metrics from a federated_history.csv or configs
        run_info = {"name": p.name, "rounds": [], "source_only_dice": None, "adapted_dice": None}

        hist_csv = p / "federated_history.csv"
        if hist_csv.exists():
            import csv
            with open(hist_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    run_info["rounds"].append({
                        "round":      int(row.get("round", 0)),
                        "train_loss": float(row.get("train_loss", 0)),
                        "train_iou":  float(row.get("train_iou", 0)),
                    })

        # configs carry best_dice sometimes
        for cfg_path in p.glob("config_*.yml"):
            try:
                with open(cfg_path, encoding="utf-8-sig") as f:
                    cfg = yaml.safe_load(f)
                fed = cfg.get("federated", {})
                if "best_dice" in fed:
                    run_info["adapted_dice"] = fed["best_dice"]
            except Exception:
                pass

        runs.append(run_info)
    return runs


def _read_adaptation_log(log_path: Path):
    """Parse a tt_sfuda log file and return structured results."""
    if not log_path.exists():
        return None
    # Auto-detect encoding: UTF-16 LE/BE (BOM) or UTF-8
    raw = log_path.read_bytes()
    if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
        text = raw.decode("utf-16", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
    result = {"source_only_dice": None, "adapted_dice": None,
              "train_loss": None, "train_iou": None,
              "refine_loss": None, "refine_iou": None}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Source_only dice:"):
            result["source_only_dice"] = float(line.split(":")[1])
        elif line.startswith("Adapted target model dice:"):
            result["adapted_dice"] = float(line.split(":")[1])
        elif line.startswith("train_loss"):
            parts = line.split()
            result["train_loss"] = float(parts[1])
            result["train_iou"]  = float(parts[4])
        elif line.startswith("refine_loss"):
            parts = line.split()
            result["refine_loss"] = float(parts[1])
            result["refine_iou"]  = float(parts[4])
    return result


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


@app.route("/api/models")
def api_models():
    return jsonify(_list_source_models())


# ── Hardcoded experiment results (CS2311_CH200) ────────────────────────
# Stage 1: supervised train_source.py → dice=0.5879
# Stage 2: fl_warm_r5_e2 (5 rounds warm-start) → round dices below
# Stage 2b: fl_hot_r10_e2 (5 more rounds hot-start) → source_dice=0.5732
# Stage 3: TT-SFUDA on leafandmask_trial → adapted_dice=0.5539
EXPERIMENT_RESULTS = {
    "supervised": {
        "source_only_dice": 0.5879,
        "iou": 0.4453,
        "loss": 0.6215,
    },
    "fl_warm_r5": {
        "rounds": [
            {"round": 1, "val_dice": 0.5359, "val_iou": 0.3953},
            {"round": 2, "val_dice": 0.5391, "val_iou": 0.4018},
            {"round": 3, "val_dice": 0.5408, "val_iou": 0.4012},
            {"round": 4, "val_dice": 0.5476, "val_iou": 0.4102},
            {"round": 5, "val_dice": 0.5466, "val_iou": 0.4053},
        ]
    },
    "best_adaptation": {
        "model":            "fl_hot_r10_e2",
        "source_only_dice": 0.5732,
        "adapted_dice":     0.5539,
        "train_iou":        0.3249,
        "refine_iou":       0.3309,
        "train_loss":       0.5700,
        "refine_loss":      0.4889,
    },
}


@app.route("/api/fl_metrics")
def api_fl_metrics():
    metrics = _collect_fl_metrics()
    return jsonify({
        "fl_runs":            metrics,
        "adaptation_results": {"best": EXPERIMENT_RESULTS["best_adaptation"]},
        "experiment_results": EXPERIMENT_RESULTS,
    })


@app.route("/api/gallery_targets")
def api_gallery_targets():
    return jsonify([{"value": k, "label": v["label"]} for k, v in GALLERY_CONFIG.items()])


@app.route("/api/test_gallery")
def api_test_gallery():
    """Return test images with GT masks and pre-computed predicted masks."""
    try:
        target = request.args.get("target", "leafandmask_full")
        cfg = GALLERY_CONFIG.get(target)
        if not cfg:
            return jsonify({"error": f"Unknown target: {target}"}), 400

        img_dir     = cfg["img_dir"]
        gt_dir      = cfg["gt_dir"]
        pred_dir    = cfg["pred_dir"]
        compare_dir = cfg["compare_dir"]
        metrics_file= cfg["metrics_file"]

        # Read metrics.txt if available
        metrics_txt = None
        if metrics_file and Path(metrics_file).exists():
            metrics_txt = Path(metrics_file).read_text(encoding="utf-8").strip()

        pred_prefix = cfg.get("pred_prefix", "")
        pred_ext    = cfg.get("pred_ext", ".png")

        items = []
        img_ids = sorted([p.stem for p in img_dir.glob("*.jpg")])

        for img_id in img_ids:
            img_path     = img_dir  / f"{img_id}.jpg"
            gt_path      = gt_dir   / f"{img_id}.png"
            pred_path    = pred_dir / f"{pred_prefix}{img_id}{pred_ext}"
            compare_path = (compare_dir / f"{img_id}_compare.png") if compare_dir else None

            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            h0, w0 = img.shape[:2]
            scale = min(320 / w0, 320 / h0, 1.0)
            tw, th = int(w0 * scale), int(h0 * scale)
            thumb = cv2.resize(img, (tw, th))

            gt_b64, pred_b64, compare_b64, dice = "", "", "", None

            if gt_path.exists():
                gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                gt_r = cv2.resize(gt, (tw, th))
                gt_vis = np.zeros((th, tw), dtype=np.uint8)
                gt_vis[gt_r > 0] = 255
                gt_b64 = _ndarray_to_b64png(gt_vis)

            if pred_path.exists():
                pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
                pred_r = cv2.resize(pred, (tw, th))
                pred_b64 = _ndarray_to_b64png(pred_r)

                if gt_path.exists():
                    gt_full   = cv2.imread(str(gt_path),   cv2.IMREAD_GRAYSCALE)
                    pred_full = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
                    pred_rs   = cv2.resize(pred_full, (gt_full.shape[1], gt_full.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)
                    gt_bin   = (gt_full  > 0).astype(np.float32)
                    pred_bin = (pred_rs  > 0).astype(np.float32)
                    inter    = (gt_bin * pred_bin).sum()
                    denom    = gt_bin.sum() + pred_bin.sum()
                    dice = round(float(2 * inter / (denom + 1e-6)), 4) if denom > 0 else 0.0

            # Compare image (pre-rendered side-by-side)
            if compare_path and compare_path.exists():
                cmp_img = cv2.imread(str(compare_path))
                # resize width to max 640
                ch, cw = cmp_img.shape[:2]
                cscale = min(640 / cw, 1.0)
                cmp_img = cv2.resize(cmp_img, (int(cw * cscale), int(ch * cscale)))
                compare_b64 = _ndarray_to_b64png(cmp_img)

            items.append({
                "id":          img_id,
                "img_b64":     _ndarray_to_b64png(thumb),
                "gt_b64":      gt_b64,
                "pred_b64":    pred_b64,
                "compare_b64": compare_b64,
                "dice":        dice,
            })

        return jsonify({
            "items":        items,
            "target":       target,
            "metrics_txt":  metrics_txt,
            "has_compare":  compare_dir is not None,
        })
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Expect: multipart/form-data with:
      - image   (file)
      - source  (str) model folder name
    Returns: JSON with base64-encoded mask and overlay PNG
    """
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        source = request.form.get("source", "multi_dataset_fedavg_3containers_r2_20260530")
        model_dir = MODELS_DIR / source
        weights = model_dir / "model.pth"
        if not weights.exists():
            return jsonify({"error": f"Model not found: {source}"}), 404

        # Load config
        cfg_path = list(model_dir.glob("config_*.yml"))[0]
        with open(cfg_path, encoding="utf-8-sig") as f:
            cfg = yaml.safe_load(f)
        cfg.setdefault("num_workers", 0)

        # Read image
        file_bytes = np.frombuffer(request.files["image"].read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Cannot decode image"}), 400

        orig_h, orig_w = img.shape[:2]

        model = _load_model(
            cfg["arch"], cfg["num_classes"], cfg["input_channels"],
            cfg["deep_supervision"], str(weights)
        )

        tensor = _preprocess(img, cfg["input_h"], cfg["input_w"])
        mask_512, prob_512 = _predict_tensor(model, tensor)

        # Resize outputs back to original size
        mask_orig = cv2.resize(mask_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        prob_orig = cv2.resize(prob_512, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        overlay = _overlay(img, mask_orig)

        # Convert probability to colormap heatmap
        heat = (prob_orig * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

        fg_ratio = float((mask_orig > 0).mean())

        return jsonify({
            "mask_b64":    _ndarray_to_b64png(mask_orig),
            "overlay_b64": _ndarray_to_b64png(overlay),
            "heatmap_b64": _ndarray_to_b64png(heat_color),
            "fg_ratio":    round(fg_ratio * 100, 2),
            "source":      source,
        })

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/api/adapt_and_predict", methods=["POST"])
def api_adapt_and_predict():
    """
    Light in-browser adaptation demo:
    - 1 forward pass on uploaded image to get prediction from source model
    - 1 gradient step on that pseudo-label
    - 1 forward pass with adapted weights to compare
    Expect same fields as /api/predict
    """
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        source = request.form.get("source", "multi_dataset_fedavg_3containers_r2_20260530")
        model_dir = MODELS_DIR / source
        weights = model_dir / "model.pth"
        if not weights.exists():
            return jsonify({"error": f"Model not found: {source}"}), 404

        cfg_path = list(model_dir.glob("config_*.yml"))[0]
        with open(cfg_path, encoding="utf-8-sig") as f:
            cfg = yaml.safe_load(f)
        cfg.setdefault("lr", 1e-4)
        cfg.setdefault("weight_decay", 1e-4)
        cfg.setdefault("loss", "BCEDiceLoss")

        file_bytes = np.frombuffer(request.files["image"].read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Cannot decode image"}), 400

        orig_h, orig_w = img.shape[:2]

        # ── Source model (frozen) ──────────────────────────────────────────
        src_model = _load_model(
            cfg["arch"], cfg["num_classes"], cfg["input_channels"],
            cfg["deep_supervision"], str(weights)
        )
        tensor = _preprocess(img, cfg["input_h"], cfg["input_w"])

        # Source prediction (no grad)
        src_mask, src_prob = _predict_tensor(src_model, tensor)

        # ── Adaptation model (copy weights, fine-tune 1 step) ─────────────
        import copy
        ada_model = copy.deepcopy(src_model)
        ada_model.train()

        criterion = losses.__dict__[cfg.get("loss", "BCEDiceLoss")]().to(device)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, ada_model.parameters()),
            lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"])
        )

        # Pseudo label from source model
        pseudo = torch.from_numpy((src_prob >= 0.5).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        x = tensor.to(device)

        adapt_steps = int(request.form.get("adapt_steps", 3))
        losses_hist = []
        for _ in range(adapt_steps):
            optimizer.zero_grad()
            out = ada_model(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            loss = criterion(out, pseudo)
            loss.backward()
            optimizer.step()
            losses_hist.append(round(float(loss.item()), 4))

        # Post-adaptation prediction
        ada_model.eval()
        ada_mask, ada_prob = _predict_tensor(ada_model, tensor)

        # Resize back
        src_mask_orig = cv2.resize(src_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        ada_mask_orig = cv2.resize(ada_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        src_overlay = _overlay(img, src_mask_orig, color=(255, 100, 0))
        ada_overlay = _overlay(img, ada_mask_orig, color=(0, 220, 0))

        src_fg = round(float((src_mask_orig > 0).mean()) * 100, 2)
        ada_fg = round(float((ada_mask_orig > 0).mean()) * 100, 2)

        return jsonify({
            "source_mask_b64":     _ndarray_to_b64png(src_mask_orig),
            "source_overlay_b64":  _ndarray_to_b64png(src_overlay),
            "adapted_mask_b64":    _ndarray_to_b64png(ada_mask_orig),
            "adapted_overlay_b64": _ndarray_to_b64png(ada_overlay),
            "source_fg_ratio":     src_fg,
            "adapted_fg_ratio":    ada_fg,
            "adapt_losses":        losses_hist,
            "adapt_steps":         adapt_steps,
            "source":              source,
        })

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


if __name__ == "__main__":
    print(f"Device: {device}")
    print(f"Models dir: {MODELS_DIR}")
    app.run(host="0.0.0.0", port=5000, debug=False)
