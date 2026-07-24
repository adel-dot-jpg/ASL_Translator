"""
Held-out test-set accuracy benchmark for the ASL vision model.

Matches the label scheme used by the corrected train.py: sign_mnist's raw
labels (0-24, with a gap at 9 for J) are compacted down to a contiguous
0-23 range, and "unknown" occupies a clean, non-colliding class 24. This
is also what main.py's ALPHABET string already assumes at inference time.

Uses the same held-out split as train.py: sign_mnist_test.csv (confirmed
as the 7172-row set) plus the unknown_hands/valid and unknown_gestures/valid
folders.
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from models.vision_model import initialize_model as init_vision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Data"

IMG_HEIGHT = 28
IMG_WIDTH = 28
IMG_CHS = 1
UNKNOWN_CLASS = 24

CLASS_NAMES = {
    0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h", 8: "i",
    9: "k", 10: "l", 11: "m", 12: "n", 13: "o", 14: "p", 15: "q", 16: "r",
    17: "s", 18: "t", 19: "u", 20: "v", 21: "w", 22: "x", 23: "y",
    24: "unknown",
}


def compact_label(raw_label: int) -> int:
    """Same compaction as train.py -- must stay in sync."""
    if raw_label < 9:
        return raw_label
    elif raw_label > 9:
        return raw_label - 1
    else:
        raise ValueError(f"Unexpected label 9 (J) -- should not appear in sign_mnist data")


# --- copied from train.py, keep in sync ---

class MyDataset(Dataset):
    def __init__(self, base_df):
        x_df = base_df.copy()
        y_df = x_df.pop('label').apply(compact_label)
        x_df = x_df.values / 255
        x_df = x_df.reshape(-1, IMG_CHS, IMG_WIDTH, IMG_HEIGHT)
        self.xs = torch.tensor(x_df).float().to(device)
        self.ys = torch.tensor(y_df.values).to(device)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

    def __len__(self):
        return len(self.xs)


class UnknownHandDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        for ext in ("*.jpg", "*.png", "*.jpeg"):
            self.image_paths.extend(self.root_dir.rglob(ext))

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(img)
        x_df = img_array / 255
        x_df = x_df.reshape(IMG_CHS, IMG_WIDTH, IMG_HEIGHT)
        x = torch.tensor(x_df).float().to(device)
        y = torch.tensor(UNKNOWN_CLASS).to(device)
        return x, y

    def __len__(self):
        return len(self.image_paths)


def build_valid_loader(batch_size=32):
    valid_df = pd.read_csv(DATA_DIR / "ASL_Data" / "sign_mnist_test.csv")
    valid_asl = MyDataset(valid_df)

    valid_hands = UnknownHandDataset(root_dir=DATA_DIR / "Non_ASL_Data" / "unknown_hands" / "Hands" / "valid")
    valid_gestures = UnknownHandDataset(root_dir=DATA_DIR / "Non_ASL_Data" / "unknown_gestures" / "valid")
    valid_unknown = ConcatDataset([valid_hands, valid_gestures])

    valid_data = ConcatDataset([valid_asl, valid_unknown])
    return DataLoader(valid_data, batch_size=batch_size)


def main():
    model = init_vision()
    model.eval()

    valid_loader = build_valid_loader()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in valid_loader:
            output = model(x)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    print(f"Held-out validation accuracy: {acc * 100:.2f}%  (n={len(all_labels)})")

    present_labels = sorted(set(all_labels) | set(all_preds))
    target_names = [CLASS_NAMES.get(l, str(l)) for l in present_labels]

    print("\nPer-class report:")
    print(classification_report(
        all_labels, all_preds,
        labels=present_labels, target_names=target_names, zero_division=0
    ))

    print("\nConfusion matrix (rows=true, cols=predicted), label order:", present_labels)
    print(confusion_matrix(all_labels, all_preds, labels=present_labels))


if __name__ == "__main__":
    main()