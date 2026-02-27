import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset

MPL_CONFIG_DIR = Path(__file__).resolve().parent / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
DATA_ROOT = Path("/Users/haelpark/Desktop/c1_embeddings")

SEED = 42
BATCH_SIZE = 16
EPOCHS = 10000
LR = 3e-4
WEIGHT_DECAY = 1e-4

TRAIN_SIZE = 0.6
TEST_SIZE = 0.4
CV_ROUNDS = 8

MAX_LEN = 720
N_FEATURES = 9

PLOT_LIVE = True
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

PATIENCE = 10
MIN_DELTA = 1e-4

AUGMENT_TRAIN = True
AUGMENT_TTA = True
TTA_PASSES = 5

USE_GROUP_SPLIT = True


torch.manual_seed(SEED)
np.random.seed(SEED)


# ----------------------------
# Utilities
# ----------------------------
def infer_group_id(path: Path) -> str:
    """
    Heuristic group id from filename for grouped split.
    e.g., captureA_01.npy -> captureA, cla1-3.npy -> cla1
    """
    stem = path.stem
    match = re.match(r"(.+?)[_-]\d+$", stem)
    if match:
        return match.group(1)
    return stem


def discover_samples(root: Path) -> list[tuple[Path, int]]:
    root = Path(root)
    samples: list[tuple[Path, int]] = []

    for label_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        if label_dir.name not in ("0", "1"):
            continue
        label = int(label_dir.name)
        for f in sorted(label_dir.glob("*.npy")):
            samples.append((f, label))

    if not samples:
        raise FileNotFoundError(f"No .npy files found under {root}/{{0,1}}")

    counts = {0: 0, 1: 0}
    for _, y in samples:
        counts[y] += 1
    print(f"Dataset loaded: {len(samples)} files | label0={counts[0]} label1={counts[1]}")

    return samples


def fit_train_scaler(
    samples: list[tuple[Path, int]],
    train_idx: list[int],
    n_features: int,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train-only normalization stats over raw timesteps (no leakage).
    """
    feat_sum = np.zeros((n_features,), dtype=np.float64)
    feat_sq_sum = np.zeros((n_features,), dtype=np.float64)
    n_rows = 0

    for idx in train_idx:
        path, _ = samples[idx]
        x = np.load(path)
        if x.ndim != 2 or x.shape[1] != n_features:
            raise ValueError(f"Invalid shape at {path}: expected (T,{n_features}), got {x.shape}")

        x = x.astype(np.float64, copy=False)
        feat_sum += x.sum(axis=0)
        feat_sq_sum += np.square(x).sum(axis=0)
        n_rows += x.shape[0]

    if n_rows == 0:
        raise ValueError("No rows in train split to fit scaler.")

    mean = feat_sum / n_rows
    var = np.maximum((feat_sq_sum / n_rows) - np.square(mean), eps)
    std = np.sqrt(var)

    return mean.astype(np.float32), std.astype(np.float32)


def _valid_binary_split(labels: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> bool:
    train_unique = np.unique(labels[train_idx])
    test_unique = np.unique(labels[test_idx])
    return len(train_unique) == 2 and len(test_unique) == 2


def repeated_split_indices(
    labels: np.ndarray,
    groups: np.ndarray,
    rounds: int,
    seed: int,
) -> tuple[int, np.ndarray, np.ndarray, str]:
    """
    Repeated CV splits. Tries group-aware split first, falls back to stratified split.
    """
    n = len(labels)
    indices = np.arange(n)

    unique_labels, counts = np.unique(labels, return_counts=True)
    can_stratify = len(unique_labels) > 1 and np.all(counts >= 2)

    for fold in range(rounds):
        split_mode = "stratified"
        train_idx = None
        test_idx = None

        if USE_GROUP_SPLIT and len(np.unique(groups)) >= 2:
            # Try multiple random group splits until we find one containing both classes.
            gss = GroupShuffleSplit(
                n_splits=24,
                train_size=TRAIN_SIZE,
                test_size=TEST_SIZE,
                random_state=seed + fold,
            )
            for train_pos, test_pos in gss.split(indices, y=labels, groups=groups):
                if _valid_binary_split(labels, train_pos, test_pos):
                    train_idx, test_idx = train_pos, test_pos
                    split_mode = "group"
                    break

        if train_idx is None or test_idx is None:
            stratify_labels = labels if can_stratify else None
            train_idx, test_idx = train_test_split(
                indices,
                train_size=TRAIN_SIZE,
                test_size=TEST_SIZE,
                shuffle=True,
                stratify=stratify_labels,
                random_state=seed + fold,
            )
            if not _valid_binary_split(labels, train_idx, test_idx):
                raise RuntimeError(
                    f"Fold {fold + 1}: invalid split with missing class in train or test."
                )

        yield fold, train_idx, test_idx, split_mode


# ----------------------------
# Dataset
# ----------------------------
class CaptureSequenceDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        indices: list[int],
        max_len: int,
        n_features: int,
        scaler_mean: np.ndarray,
        scaler_std: np.ndarray,
        augment: bool,
        augment_profile: str,
        rng_seed: int,
    ):
        self.samples = samples
        self.indices = list(indices)
        self.max_len = max_len
        self.n_features = n_features
        self.scaler_mean = scaler_mean.astype(np.float32, copy=False)
        self.scaler_std = scaler_std.astype(np.float32, copy=False)
        self.augment = augment
        self.augment_profile = augment_profile  # "train" or "tta"
        self.rng = np.random.default_rng(rng_seed)

    def __len__(self):
        return len(self.indices)

    def _truncate_or_pad(self, x: np.ndarray) -> tuple[np.ndarray, int]:
        t = x.shape[0]
        length = min(t, self.max_len)
        out = np.zeros((self.max_len, self.n_features), dtype=np.float32)
        out[:length] = x[:length].astype(np.float32, copy=False)
        return out, length

    def _random_time_crop(self, x: np.ndarray, min_ratio: float, max_ratio: float) -> np.ndarray:
        t = x.shape[0]
        if t < 16:
            return x

        ratio = float(self.rng.uniform(min_ratio, max_ratio))
        crop_len = max(8, int(t * ratio))
        if crop_len >= t:
            return x

        start = int(self.rng.integers(0, t - crop_len + 1))
        return x[start:start + crop_len]

    def _random_shift(self, x: np.ndarray, max_fraction: float) -> np.ndarray:
        t = x.shape[0]
        if t < 4:
            return x

        max_shift = max(1, int(t * max_fraction))
        shift = int(self.rng.integers(-max_shift, max_shift + 1))
        if shift == 0:
            return x

        out = np.zeros_like(x)
        if shift > 0:
            out[shift:] = x[: t - shift]
        else:
            out[: t + shift] = x[-shift:]
        return out

    def _augment(self, x: np.ndarray) -> np.ndarray:
        if self.augment_profile == "train":
            if self.rng.random() < 0.6:
                x = self._random_time_crop(x, 0.75, 1.0)
            if self.rng.random() < 0.6:
                x = self._random_shift(x, 0.10)
            if self.rng.random() < 0.5:
                # Feature dropout: zero one feature channel.
                feat = int(self.rng.integers(0, self.n_features))
                x[:, feat] = 0.0
            if self.rng.random() < 0.9:
                noise = self.rng.normal(0.0, 0.03, size=x.shape).astype(np.float32)
                x = x + noise
        else:  # tta
            if self.rng.random() < 0.4:
                x = self._random_time_crop(x, 0.85, 1.0)
            if self.rng.random() < 0.4:
                x = self._random_shift(x, 0.06)
            if self.rng.random() < 0.7:
                noise = self.rng.normal(0.0, 0.01, size=x.shape).astype(np.float32)
                x = x + noise

        return x

    def __getitem__(self, i: int):
        global_idx = self.indices[i]
        path, label = self.samples[global_idx]

        x = np.load(path)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"Invalid shape at {path}: expected (T,{self.n_features}), got {x.shape}")

        x = x.astype(np.float32, copy=False)

        # Train-only fitted normalization applied to every fold dataset.
        x = (x - self.scaler_mean) / self.scaler_std

        if self.augment:
            x = self._augment(x)

        x_pad, length = self._truncate_or_pad(x)

        x_tensor = torch.from_numpy(x_pad)
        length_tensor = torch.tensor(length, dtype=torch.long)
        y_tensor = torch.tensor(label, dtype=torch.float32)
        return x_tensor, length_tensor, y_tensor


# ----------------------------
# Model: Short Temporal CNN
# ----------------------------
class ShortConvNet(nn.Module):
    def __init__(self, n_features: int, dropout: float = 0.30):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        h = self.backbone(x)  # (B, C, T)

        bsz, _, t = h.shape
        lengths = lengths.clamp(min=1, max=t)

        mask = (torch.arange(t, device=h.device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(1)  # (B,1,T)
        denom = mask.sum(dim=2).clamp(min=1).to(h.dtype)

        mean_pool = (h * mask).sum(dim=2) / denom

        neg_inf = torch.full_like(h, -1e4)
        max_pool = torch.where(mask, h, neg_inf).max(dim=2).values

        feats = torch.cat([mean_pool, max_pool], dim=1)
        return self.head(feats).squeeze(1)


# ----------------------------
# Live plot (per fold)
# ----------------------------
class LivePlot:
    def __init__(self, title: str):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title(title)
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.train_line, = self.ax.plot([], [], label="train_loss")
        self.val_line, = self.ax.plot([], [], label="val_loss")
        self.ax.legend()
        self.epochs = []
        self.train_losses = []
        self.val_losses = []

    def update(self, epoch: int, train_loss: float, val_loss: float):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_line.set_data(self.epochs, self.train_losses)
        self.val_line.set_data(self.epochs, self.val_losses)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        plt.ioff()
        plt.show()


# ----------------------------
# Train/Eval helpers
# ----------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, lengths, y in loader:
        x = x.to(DEVICE)
        lengths = lengths.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def eval_loader(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    probs_all, y_all = [], []

    for x, lengths, y in loader:
        x = x.to(DEVICE)
        lengths = lengths.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x, lengths)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct += (preds == y).sum().item()
        total += x.size(0)

        probs_all.append(probs.detach().cpu().numpy())
        y_all.append(y.detach().cpu().numpy())

    probs = np.concatenate(probs_all) if probs_all else None
    ys = np.concatenate(y_all) if y_all else None
    return total_loss / max(total, 1), correct / max(total, 1), probs, ys


@torch.no_grad()
def predict_with_tta(
    model,
    samples: list[tuple[Path, int]],
    test_idx: list[int],
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    criterion,
    fold_seed: int,
):
    passes = max(1, TTA_PASSES)

    probs_passes = []
    ys_ref = None

    for p in range(passes):
        use_aug = AUGMENT_TTA and p > 0
        ds = CaptureSequenceDataset(
            samples=samples,
            indices=test_idx,
            max_len=MAX_LEN,
            n_features=N_FEATURES,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
            augment=use_aug,
            augment_profile="tta",
            rng_seed=fold_seed + 1000 + p,
        )
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        _, _, probs, ys = eval_loader(model, loader, criterion)

        if probs is None or ys is None:
            raise RuntimeError("TTA inference returned empty predictions.")

        if ys_ref is None:
            ys_ref = ys
        elif not np.array_equal(ys_ref.astype(int), ys.astype(int)):
            raise RuntimeError("Inconsistent labels across TTA passes.")

        probs_passes.append(probs)

    probs_mean = np.mean(np.stack(probs_passes, axis=0), axis=0)
    return probs_mean, ys_ref


def main():
    samples = discover_samples(DATA_ROOT)
    labels = np.array([y for _, y in samples], dtype=int)
    groups = np.array([infer_group_id(path) for path, _ in samples])

    print(f"Device: {DEVICE}")
    print(f"CV rounds: {CV_ROUNDS} | split={int(TRAIN_SIZE * 100)}/{int(TEST_SIZE * 100)}")
    print(f"Model: short CNN | epochs={EPOCHS} | batch={BATCH_SIZE}")
    print(f"Adam (L2) lr={LR} wd={WEIGHT_DECAY} | early_stop patience={PATIENCE}")
    print(f"Aug(train)={AUGMENT_TRAIN} | TTA passes={max(1, TTA_PASSES)}")

    fold_results = []
    pred_rows = []

    for fold, train_idx_np, test_idx_np, split_mode in repeated_split_indices(
        labels=labels,
        groups=groups,
        rounds=CV_ROUNDS,
        seed=SEED,
    ):
        train_idx = list(train_idx_np)
        test_idx = list(test_idx_np)

        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        print("\n" + "=" * 74)
        print(
            f"FOLD {fold + 1}/{CV_ROUNDS} | mode={split_mode} | train={len(train_idx)} test={len(test_idx)} "
            f"| train(y0={int((train_labels == 0).sum())}, y1={int((train_labels == 1).sum())}) "
            f"| test(y0={int((test_labels == 0).sum())}, y1={int((test_labels == 1).sum())})"
        )
        print("=" * 74)

        scaler_mean, scaler_std = fit_train_scaler(samples, train_idx, n_features=N_FEATURES)

        train_ds = CaptureSequenceDataset(
            samples=samples,
            indices=train_idx,
            max_len=MAX_LEN,
            n_features=N_FEATURES,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
            augment=AUGMENT_TRAIN,
            augment_profile="train",
            rng_seed=SEED + fold,
        )
        test_ds = CaptureSequenceDataset(
            samples=samples,
            indices=test_idx,
            max_len=MAX_LEN,
            n_features=N_FEATURES,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
            augment=False,
            augment_profile="tta",
            rng_seed=SEED + 500 + fold,
        )

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = ShortConvNet(n_features=N_FEATURES, dropout=0.30).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        plotter = LivePlot(title=f"CNN Fold {fold + 1}/{CV_ROUNDS}") if PLOT_LIVE else None

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, EPOCHS + 1):
            print(f"\n  [Fold {fold + 1}/{CV_ROUNDS}] Epoch {epoch}/{EPOCHS}")

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc, _, _ = eval_loader(model, test_loader, criterion)

            print(f"    train: loss={train_loss:.4f} acc={train_acc:.4f}")
            print(f"    val:   loss={val_loss:.4f} acc={val_acc:.4f}")

            if plotter is not None:
                plotter.update(epoch, train_loss, val_loss)

            if val_loss < (best_val_loss - MIN_DELTA):
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    print(f"    early stop triggered (no improvement in {PATIENCE} epochs)")
                    break

        if plotter is not None:
            plotter.close()

        if best_state is not None:
            model.load_state_dict(best_state)

        probs, ys = predict_with_tta(
            model=model,
            samples=samples,
            test_idx=test_idx,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
            criterion=criterion,
            fold_seed=SEED + fold,
        )

        y_true_arr = ys.astype(int)
        y_pred_arr = (probs >= 0.5).astype(int)
        correct_arr = (y_pred_arr == y_true_arr).astype(int)
        fold_acc = float(correct_arr.mean())

        fold_results.append((fold + 1, split_mode, fold_acc, len(test_idx)))

        for idx, y_true, prob, y_pred, correct in zip(
            test_idx,
            y_true_arr,
            probs,
            y_pred_arr,
            correct_arr,
        ):
            path, _ = samples[int(idx)]
            pred_rows.append(
                (
                    fold + 1,
                    split_mode,
                    int(idx),
                    path.name,
                    int(y_true),
                    float(prob),
                    int(y_pred),
                    int(correct),
                )
            )

        print(f"\n  >>> Fold {fold + 1}: accuracy={fold_acc:.4f} | split_mode={split_mode}")

    overall_acc = sum(r[-1] for r in pred_rows) / max(len(pred_rows), 1)
    fold_accs = [acc for _, _, acc, _ in fold_results]
    fold_mean = float(np.mean(fold_accs)) if fold_accs else 0.0
    fold_std = float(np.std(fold_accs)) if fold_accs else 0.0

    print("\n" + "#" * 74)
    print(f"CV DONE | rounds={CV_ROUNDS} | sample_acc={overall_acc:.4f} | fold_mean={fold_mean:.4f} +/- {fold_std:.4f}")
    print("#" * 74)

    out_csv = Path("cv_cnn_predictions.csv")
    with out_csv.open("w") as f:
        f.write("fold,split_mode,test_idx,file,y_true,prob,y_pred,correct\n")
        for fold, split_mode, idx, file_name, y_true, prob, y_pred, correct in pred_rows:
            f.write(f"{fold},{split_mode},{idx},{file_name},{y_true},{prob:.6f},{y_pred},{correct}\n")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
