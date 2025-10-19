import argparse
import numpy as np
import os
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from sklearn.metrics import roc_curve
from torch.utils.data import Dataset, DataLoader

from DataUtils import (
    index_dataframe,  # list files into a dataframe
    IMG_EXTS,  # valid image extensions
    REQUIRED_COLS,  # fixed output schema
    standardise_predictions,  # schema/dtype helper
)  # uses the repo's DataUtils module
from models.clip_models import CLIPModelShuffleAttentionPenultimateLayer

# Simple normalisation stats (same idea as original)
MEAN = {"imagenet": [0.485, 0.456, 0.406], "clip": [0.48145466, 0.4578275, 0.40821073]}
STD = {"imagenet": [0.229, 0.224, 0.225], "clip": [0.26862954, 0.26130258, 0.27577711]}


def find_best_threshold(y_true, y_pred):
    """
    Robust threshold search:
    - No assumption on ordering or class balance.
    - If only one class is present, return 0.5 as a safe default.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)

    # single-class set: no meaningful ROC; return neutral threshold
    if len(np.unique(y_true)) < 2:
        return 0.5

    # Youden's J: argmax(tpr - fpr)
    fpr, tpr, thr = roc_curve(y_true, y_pred)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


class FakePartsV2Dataset(Dataset):
    """
    Dataset backed by DataUtils.index_dataframe. It mimics RealFakeDataset but
    takes rows directly from the dataframe produced by index_dataframe.
    """

    def __init__(self, df: pd.DataFrame, arch: str = "clip", is_norm: bool = True, image_size: int = 224):
        self.df = df.reset_index(drop=True)
        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        if is_norm:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = Image.open(r["abs_path"]).convert("RGB")
        img = self.transform(img)
        label = int(r["label"])  # 0=real, 1=fake (ground truth)
        return img, label, idx  # return row index to recover metadata later


@torch.no_grad()
def infer_to_rows(model, loader, df_idx: pd.DataFrame, model_name: str, threshold: float = 0.5):
    """
    Forward pass and build rows matching REQUIRED_COLS.
    score: probability for fake (higher = more fake).
    pred:  int(score >= threshold).
    """
    rows = []
    model.eval()
    device = next(model.parameters()).device

    for imgs, labels, idxs in loader:
        imgs = imgs.to(device, non_blocking=True)
        out = model(imgs)  # shape [B,2] or [B,1]/[B]
        if out.shape[-1] == 2:
            logit_fake = out[:, 1]  # use fake channel
        else:
            logit_fake = out.squeeze(-1)  # assume single logit is fake
        prob_fake = torch.sigmoid(logit_fake).float()  # [B]

        if threshold is not None:
            preds = (prob_fake >= threshold).long().cpu().tolist()
        else:
            preds = [-1] * len(prob_fake)  # placeholder; will be set later
        scores = prob_fake.cpu().tolist()
        labels_np = labels.long().cpu().tolist()
        idxs_np = idxs.long().cpu().tolist()

        for j, ridx in enumerate(idxs_np):
            meta = df_idx.iloc[ridx]
            rows.append({
                "sample_id": str(meta["rel_path"]),
                "task": str(meta["task"]),
                "method": str(meta["method"]),
                "subset": str(meta["subset"]),
                "label": int(labels_np[j]),
                "model": str(model_name),
                "mode": str(meta["mode"]),
                "score": float(scores[j]),
                "pred": int(preds[j]),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, type=str)
    ap.add_argument("--results", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str, help="Path to attention head checkpoint (.pth)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", default=32, type=int)
    ap.add_argument("--arch", default="clip", type=str)  # controls normalisation choice
    ap.add_argument("--granularity", default=14, type=int)  # ViT-L/14 patches
    ap.add_argument("--model_name", default="D3_ViT-L14", type=str)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Fixed decision threshold; if omitted, find best via ROC-Youden.")
    ap.add_argument("--image_size", default=224, type=int)
    ap.add_argument("--num_workers", default=4, type=int)
    args = ap.parse_args()

    os.makedirs(args.results, exist_ok=True)
    out_csv = str(Path(args.results) / "predictions.csv")

    # 1) Build index dataframe from data_root
    df_idx = index_dataframe(Path(args.data_root), file_exts=IMG_EXTS)  # images only
    if len(df_idx) == 0:
        raise SystemExit(f"No images found under {args.data_root} with extensions {IMG_EXTS}.")

    # 2) Create dataset/loader
    dataset = FakePartsV2Dataset(df_idx, arch=args.arch, is_norm=True, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # 3) Create and load the detector
    model = CLIPModelShuffleAttentionPenultimateLayer(
        "ViT-L/14", shuffle_times=1, original_times=1, patch_size=[args.granularity]
    )
    state_dict = torch.load(args.ckpt, map_location="cpu")
    model.attention_head.load_state_dict(state_dict)
    model.to(args.device).eval()

    # 4) Forward pass → rows in REQUIRED_COLS format
    rows = infer_to_rows(model, loader, df_idx, model_name=args.model_name, threshold=args.threshold)
    if args.threshold is None:
        print("Finding best threshold based on all predictions ...")
        y_true = np.array([r["label"] for r in rows])
        y_pred = np.array([r["score"] for r in rows])
        best_thr = find_best_threshold(y_true, y_pred)
        args.threshold = best_thr
        print(f"Best threshold found: {best_thr:.4f}")

    # Now recompute preds using the chosen threshold
    for r in rows:
        r["pred"] = int(r["score"] >= args.threshold)

    # 5) Standardise and save CSV
    df_out = standardise_predictions(rows)  # dtype + column check
    df_out = df_out.loc[:, list(REQUIRED_COLS)]  # enforce order
    df_out.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(f"Rows: {len(df_out)}")


if __name__ == "__main__":
    main()
