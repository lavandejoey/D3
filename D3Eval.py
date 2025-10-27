from __future__ import annotations
from typing import Dict, Any, List, Optional, Sequence, Tuple, Union
import argparse, os, time, logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # avoid PIL DecompressionBomb warnings for large imgs

# Repo utilities
from DataUtils import (
    index_dataframe, IMG_EXTS, REQUIRED_COLS, standardise_predictions,
    FakePartsV2DatasetBase, collate_skip_none, find_best_threshold
)
# Model
from models.clip_models import CLIPModelShuffleAttentionPenultimateLayer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("D3Eval")

MEAN = {"imagenet": [0.485, 0.456, 0.406], "clip": [0.48145466, 0.4578275, 0.40821073]}
STD = {"imagenet": [0.229, 0.224, 0.225], "clip": [0.26862954, 0.26130258, 0.27577711]}


def build_transform(arch: str, image_size: int, do_norm: bool = True) -> T.Compose:
    """Pick CLIP vs ImageNet stats based on arch; keep it simple."""
    stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
    ops = [T.Resize((image_size, image_size)), T.ToTensor()]
    if do_norm:
        ops.append(T.Normalize(mean=MEAN[stat_from], std=STD[stat_from]))
    return T.Compose(ops)


def _append_rows_to_csv(rows_batch: Sequence[Dict[str, Any]], out_csv: Union[str, Path]) -> None:
    """Append a batch to CSV in REQUIRED_COLS order; write header only once."""
    if not rows_batch:
        return
    df_chunk = standardise_predictions(rows_batch)  # schema+types hygiene  :contentReference[oaicite:5]{index=5}
    df_chunk = df_chunk.loc[:, list(REQUIRED_COLS)]
    out_csv = str(out_csv)
    need_header = (not os.path.exists(out_csv)) or os.path.getsize(out_csv) == 0
    df_chunk.to_csv(out_csv, mode="a", index=False, header=need_header)


@torch.no_grad()
def infer_to_rows(
        model: torch.nn.Module,
        loader: DataLoader,
        model_name: str,
        threshold: Optional[float],  # None = just collect scores
        out_csv: Optional[Union[str, Path]],  # if set & save_per_batch=True, stream-save
        save_per_batch: bool,
) -> List[Dict[str, Any]]:
    """Forward pass. Builds rows matching REQUIRED_COLS. Score = P(fake)."""
    rows: List[Dict[str, Any]] = []
    model.eval()
    device = next(model.parameters()).device

    skipped_batches = 0
    seen = 0
    t0 = time.time()

    for batch in tqdm(loader, desc="Inference"):
        if batch is None:  # collate_skip_none returns None when all samples in this batch are invalid
            skipped_batches += 1
            continue

        imgs, labels, metas = batch
        imgs = imgs.to(device, non_blocking=True)

        # Accept head output shape [B,2] (fake channel=1) or [B,1]/[B] (single fake logit)
        out = model(imgs)
        logit_fake = out[:, 1] if out.dim() == 2 and out.size(1) == 2 else out.squeeze(-1)
        prob_fake = torch.sigmoid(logit_fake).float()  # [B]

        preds = (
            (prob_fake >= threshold).long().cpu().tolist()
            if threshold is not None else
            [-1] * prob_fake.size(0)
        )
        scores = prob_fake.cpu().tolist()
        labels_np = labels.long().cpu().tolist()
        B = len(scores)
        seen += B

        batch_rows = []
        for i in range(B):
            batch_rows.append({
                "sample_id": str(metas["sample_id"][i]),
                "task": str(metas["task"][i]),
                "method": str(metas["method"][i]),
                "subset": str(metas["subset"][i]),
                "label": int(labels_np[i]),
                "model": str(model_name),
                "mode": str(metas["mode"][i]),
                "score": float(scores[i]),
                "pred": int(preds[i]),
            })

        if save_per_batch and out_csv is not None:
            _append_rows_to_csv(batch_rows, out_csv)

        rows.extend(batch_rows)

    dt = time.time() - t0
    if dt > 0:
        log.info("Inference done: %d samples in %.1fs (%.1f imgs/s), skipped batches=%d",
                 seen, dt, seen / dt, skipped_batches)
    else:
        log.info("Inference done: %d samples.", seen)
    return rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("D3Eval: CLIP attention head evaluator (frames)")
    ap.add_argument("--data_root", required=True, type=str, help="Dataset root folder")
    ap.add_argument("--data_csv", type=str, default=None, help="Optional prebuilt index CSV")
    ap.add_argument("--done_csv_list", nargs="*", default=[], type=str)
    ap.add_argument("--results", required=True, type=str, help="Folder to save predictions.csv")
    ap.add_argument("--ckpt", required=True, type=str, help="Attention head checkpoint (.pth)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", default=950, type=int)
    ap.add_argument("--arch", default="clip", type=str, help="Affects normalisation stats")
    ap.add_argument("--granularity", default=14, type=int, help="ViT-L/14 patch granularity")
    ap.add_argument("--model_name", default="D3_ViT-L14", type=str)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Fixed decision threshold; set empty to auto-search via Youden")
    ap.add_argument("--image_size", default=224, type=int)
    ap.add_argument("--num_workers", default=4, type=int)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.results)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "predictions.csv"

    # # 1) Index
    # log.info("Indexing under %s (csv=%s)", args.data_root, args.data_csv or "None")
    # df_idx = index_dataframe(Path(args.data_root), file_exts=IMG_EXTS, csv_path=args.data_csv)
    # if len(df_idx) == 0:
    #     raise SystemExit(f"No images under {args.data_root} with {IMG_EXTS}.")
    # log.info("Indexed %d files; mode counts: %s",
    #          len(df_idx),
    #          dict(df_idx["mode"].value_counts().items()))  # uses DataUtils index  :contentReference[oaicite:6]{index=6}

    # 2) Dataset/Loader
    transform = build_transform(args.arch, args.image_size, do_norm=True)
    dataset = FakePartsV2DatasetBase(
        data_root=Path(args.data_root),
        mode="frame",
        csv_path=args.data_csv,
        model_name=args.model_name,
        transform=transform,
        done_csv_list=args.done_csv_list,
        on_corrupt="warn",  # skip unreadable images politely via collate
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
        drop_last=False,
        collate_fn=collate_skip_none,  # safe with None samples  :contentReference[oaicite:7]{index=7}
        persistent_workers=(args.num_workers or 0) > 0,
    )
    log.info("DataLoader ready: batch_size=%d, workers=%d", args.batch_size, args.num_workers)

    # 3) Model
    log.info("Loading attention head: %s", args.ckpt)
    model = CLIPModelShuffleAttentionPenultimateLayer(
        "ViT-L/14", shuffle_times=1, original_times=1, patch_size=[args.granularity]
    )
    state_dict = torch.load(args.ckpt, map_location="cpu")
    model.attention_head.load_state_dict(state_dict)
    model.to(args.device).eval()
    log.info("Model to %s", args.device)

    # 4) Inference (+ threshold strategy)
    if os.path.exists(out_csv):
        log.warning("Removing existing %s to avoid mixing runs.", out_csv)
        out_csv.unlink()

    # If threshold is provided → single pass; else → pass 1 scores, pick best, pass 2 write.
    if args.threshold is not None:
        log.info("Using fixed threshold: %.4f", args.threshold)
        _ = infer_to_rows(model, loader, args.model_name, args.threshold, out_csv, save_per_batch=True)
        df = pd.read_csv(out_csv)
        log.info("Saved %s (%d rows, cols=%s)", out_csv, len(df), list(df.columns))
        return

    # Auto threshold
    log.info("No threshold provided — collecting scores for auto-search (Youden).")
    rows = infer_to_rows(model, loader, args.model_name, threshold=None, out_csv=None, save_per_batch=False)
    y_true = np.array([r["label"] for r in rows])
    y_pred = np.array([r["score"] for r in rows])
    best_thr = find_best_threshold(y_true, y_pred)  # DataUtils implementation  :contentReference[oaicite:8]{index=8}
    log.info("Best threshold = %.4f", best_thr)

    # Second pass writes per-batch with chosen threshold
    loader2 = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
        drop_last=False, collate_fn=collate_skip_none, persistent_workers=(args.num_workers or 0) > 0
    )
    _ = infer_to_rows(model, loader2, args.model_name, best_thr, out_csv, save_per_batch=True)
    df = pd.read_csv(out_csv)
    log.info("Saved %s (%d rows).", out_csv, len(df))


if __name__ == "__main__":
    main()
