import argparse
import json
import os
import inspect
from typing import Dict, Tuple, Optional, Any

import numpy as np
import torch


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pick_key(d: Dict[str, Any], candidates):
    for k in candidates:
        if k in d:
            return k
    return None


def load_npz(npz_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    z = np.load(npz_path, allow_pickle=True)
    keys = list(z.keys())

    xk = _pick_key(z, ["X", "x", "data", "inputs", "features"])
    yk = _pick_key(z, ["y", "Y", "labels", "label", "targets", "target"])

    if xk is None:
        raise KeyError(f"Could not find input array in {npz_path}. Keys found: {keys}")

    X = np.asarray(z[xk])
    y = np.asarray(z[yk]) if yk is not None else None
    return X, y


def ensure_4d_input(X: np.ndarray) -> np.ndarray:
    # Expected: (B, S=5, C=1, T=3000)
    if X.ndim == 4:
        return X
    if X.ndim == 3:
        # (S,C,T) -> (1,S,C,T) OR (B,S,T)->(B,S,1,T)
        if X.shape[1] == 1 and X.shape[2] == 3000:
            return X[None, ...]
        return X[:, :, None, :]
    if X.ndim == 2:
        # (S,T) -> (1,S,1,T)
        return X[None, :, None, :]
    raise ValueError(f"Unsupported input shape {X.shape}")


def maybe_fix_scalar_y(y: Optional[np.ndarray], n: int) -> Optional[np.ndarray]:
    if y is None:
        return None
    y = np.asarray(y)
    if y.ndim == 0:
        y = y.reshape(1)
    if y.shape[0] != n:
        raise ValueError(f"Label length mismatch: len(y)={y.shape[0]} but len(X)={n}")
    return y


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def normalize_label_map(label_map: Dict[str, Any]) -> Dict[int, str]:
    # Supports either {"0":"W"} or {"W":0}
    id_to_name: Dict[int, str] = {}
    for k, v in label_map.items():
        if isinstance(v, (int, np.integer)) and isinstance(k, str):
            id_to_name[int(v)] = str(k)
        else:
            try:
                id_to_name[int(k)] = str(v)
            except Exception:
                pass
    if not id_to_name:
        raise ValueError(f"Could not interpret label_map.json: {label_map}")
    return id_to_name


def instantiate_model_from_config(config: Dict[str, Any], n_classes: int):
    """
    Your model requires n_classes. We ALWAYS pass it.
    No architecture changes; just correct construction.
    """
    from model import CNNLSTMSleepNet

    sig = inspect.signature(CNNLSTMSleepNet.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")

    raw_kwargs = {}
    if isinstance(config.get("model_kwargs", None), dict):
        raw_kwargs.update(config["model_kwargs"])
    raw_kwargs.update({k: v for k, v in config.items() if k != "model_kwargs"})

    # Common aliases people use in configs
    if "num_classes" in raw_kwargs and "n_classes" not in raw_kwargs:
        raw_kwargs["n_classes"] = raw_kwargs["num_classes"]
    if "classes" in raw_kwargs and "n_classes" not in raw_kwargs and isinstance(raw_kwargs["classes"], int):
        raw_kwargs["n_classes"] = raw_kwargs["classes"]

    # Force correct value (source of truth = label_map)
    raw_kwargs["n_classes"] = int(n_classes)

    kwargs = {k: raw_kwargs[k] for k in raw_kwargs.keys() if k in allowed}

    # Now this cannot fail due to missing n_classes
    model = CNNLSTMSleepNet(**kwargs)
    return model


def load_model(deploy_dir: str, device: torch.device):
    weights_path = os.path.join(deploy_dir, "model_weights.pth")
    config_path = os.path.join(deploy_dir, "config.json")
    labelmap_path = os.path.join(deploy_dir, "label_map.json")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing weights: {weights_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not os.path.exists(labelmap_path):
        raise FileNotFoundError(f"Missing label map: {labelmap_path}")

    config = load_json(config_path)
    label_map = load_json(labelmap_path)
    id_to_name = normalize_label_map(label_map)
    n_classes = len(id_to_name)

    model = instantiate_model_from_config(config, n_classes=n_classes).to(device)

    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    model.load_state_dict(state, strict=True)
    model.eval()
    return model, id_to_name


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    denom = cm.sum(axis=1)
    acc = np.zeros(cm.shape[0], dtype=np.float64)
    for i in range(cm.shape[0]):
        acc[i] = (cm[i, i] / denom[i]) if denom[i] > 0 else 0.0
    return acc


def save_cm_csv(cm: np.ndarray, id_to_name: Dict[int, str], out_path: str):
    names = [id_to_name[i] for i in sorted(id_to_name.keys())]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("," + ",".join(names) + "\n")
        for i, name in enumerate(names):
            f.write(name + "," + ",".join(str(int(x)) for x in cm[i]) + "\n")


@torch.no_grad()
def run(npz_path: str, deploy_dir: str, device_str: str, batch_size: int, show_samples: int, save_cm: Optional[str]):
    device = get_device(device_str)
    model, id_to_name = load_model(deploy_dir, device)

    X, y = load_npz(npz_path)
    X = ensure_4d_input(X)
    n = X.shape[0]
    y = maybe_fix_scalar_y(y, n)

    xb = torch.from_numpy(X).float().to(device)

    preds = []
    for i in range(0, n, batch_size):
        logits = model(xb[i:i + batch_size])
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        preds.append(pred)
    y_pred = np.concatenate(preds, axis=0)

    print("\n=== Demo Output ===")
    print(f"NPZ: {npz_path}")
    print(f"Samples: {n}")

    if y is None:
        print("No labels found in NPZ → predictions only\n")
        for i in range(min(show_samples, n)):
            p = int(y_pred[i])
            print(f"[{i:04d}] pred={p} ({id_to_name[p]})")
        return

    correct = int((y_pred == y).sum())
    acc = correct / n if n else 0.0
    print(f"Accuracy (this NPZ): {acc * 100:.2f}% ({correct}/{n})\n")

    for i in range(min(show_samples, n)):
        t = int(y[i]); p = int(y_pred[i])
        mark = "✅" if t == p else "❌"
        print(f"[{i:04d}] true={t} ({id_to_name[t]}) | pred={p} ({id_to_name[p]}) {mark}")

    num_classes = len(id_to_name)
    names = [id_to_name[i] for i in sorted(id_to_name.keys())]
    cm = confusion_matrix(y, y_pred, num_classes)
    pca = per_class_accuracy(cm)

    print("\n=== Confusion Matrix (rows=true, cols=pred) ===")
    print(" " * 12 + " ".join([f"{n:>6}" for n in names]))
    for i, name in enumerate(names):
        row = " ".join([f"{cm[i, j]:6d}" for j in range(num_classes)])
        print(f"{name:>10}  {row}")

    print("\n=== Per-class Accuracy ===")
    for i, name in enumerate(names):
        total_true = int(cm.sum(axis=1)[i])
        print(f"{name:>4}: {pca[i] * 100:6.2f}% ({int(cm[i, i])}/{total_true})")

    if save_cm is not None:
        save_cm_csv(cm, id_to_name, save_cm)
        print(f"\nSaved confusion matrix CSV → {save_cm}")


def main():
    ap = argparse.ArgumentParser(description="CLI inference + confusion matrix (Step 9)")
    ap.add_argument("--npz", required=True, help="Path to NPZ (sample.npz / batch_sample.npz)")
    ap.add_argument("--deploy", default="sleepnet_deploy", help="Deploy folder with weights/config/label_map")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--show-samples", type=int, default=10)
    ap.add_argument("--save-cm", default=None, help="Optional CSV output path")
    args = ap.parse_args()

    run(
        npz_path=args.npz,
        deploy_dir=args.deploy,
        device_str=args.device,
        batch_size=args.batch_size,
        show_samples=args.show_samples,
        save_cm=args.save_cm,
    )


if __name__ == "__main__":
    main()
