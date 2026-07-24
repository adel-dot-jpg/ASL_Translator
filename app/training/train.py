import torch.nn as nn
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms.v2 as transforms
from PIL import Image

import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Anchor all paths to this script's own folder, regardless of cwd
# Anchor to the *parent* of this script's folder (app/), not this folder
# itself (app/training/), so this always reads the same Data/ that
# benchmark_accuracy.py and main.py use -- regardless of which directory
# you invoke `python train.py` from.
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "Data"

IMG_HEIGHT = 28
IMG_WIDTH = 28
IMG_CHS = 1
NUM_ASL_CLASSES = 24   # sign_mnist excludes J and Z
UNKNOWN_CLASS = 24     # final class (unknown) -- now guaranteed not to collide
N_CLASSES = 25         # 24 compacted letters + 1 unknown

# Raw sign_mnist labels run 0-24 with a GAP at 9 (J never appears, since it
# requires motion). This maps every raw label to a compacted 0-23 range with
# no gap, matching what main.py's ALPHABET string already assumes at
# inference time. Labels 0-8 (a-i) are unchanged; labels 10-24 (k-y) shift
# down by 1 to become 9-23.
def compact_label(raw_label: int) -> int:
    if raw_label < 9:
        return raw_label
    elif raw_label > 9:
        return raw_label - 1
    else:
        raise ValueError(f"Unexpected label 9 (J) -- should not appear in sign_mnist data")


train_df = pd.read_csv(DATA_DIR / "ASL_Data" / "sign_mnist_train.csv")
# sign_mnist_test.csv confirmed as the 7172-row held-out set used as "valid"
valid_df = pd.read_csv(DATA_DIR / "ASL_Data" / "sign_mnist_test.csv")


class MyDataset(Dataset):
	def __init__(self, base_df):
		x_df = base_df.copy()
		y_df = x_df.pop('label').apply(compact_label)
		x_df = x_df.values / 255  # Normalize values from 0 to 1
		x_df = x_df.reshape(-1, IMG_CHS, IMG_WIDTH, IMG_HEIGHT)
		self.xs = torch.tensor(x_df).float().to(device)
		self.ys = torch.tensor(y_df.values).to(device)

	def __getitem__(self, idx):
		x = self.xs[idx]
		y = self.ys[idx]
		return x, y

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
		img = Image.open(self.image_paths[idx]).convert("L")  # grayscale
		img = img.resize((IMG_WIDTH, IMG_HEIGHT))
		img_array = np.array(img)
		x_df = img_array / 255
		x_df = x_df.reshape(IMG_CHS, IMG_WIDTH, IMG_HEIGHT)
		x = torch.tensor(x_df).float().to(device)
		y = torch.tensor(UNKNOWN_CLASS).to(device)  # 24, now collision-free
		return x, y

	def __len__(self):
		return len(self.image_paths)


n = 32
ASL_dataset = MyDataset(train_df)

hands_dataset = UnknownHandDataset(
	root_dir=DATA_DIR / "Non_ASL_Data" / "unknown_hands" / "Hands" / "train"
)

gestures_dataset = UnknownHandDataset(
	root_dir=DATA_DIR / "Non_ASL_Data" / "unknown_gestures" / "train"
)

unknown_dataset = ConcatDataset([hands_dataset, gestures_dataset])
train_data = ConcatDataset([ASL_dataset, unknown_dataset])

train_loader = DataLoader(train_data, batch_size=n, shuffle=True)
train_N = len(train_loader.dataset)


valid_ASL_dataset = MyDataset(valid_df)

valid_hands_dataset = UnknownHandDataset(
	root_dir=DATA_DIR / "Non_ASL_Data" / "unknown_hands" / "Hands" / "valid"
)

valid_gestures_dataset = UnknownHandDataset(
	root_dir=DATA_DIR / "Non_ASL_Data" / "unknown_gestures" / "valid"
)

valid_unknown_dataset = ConcatDataset([valid_hands_dataset, valid_gestures_dataset])
valid_data = ConcatDataset([valid_ASL_dataset, valid_unknown_dataset])

valid_loader = DataLoader(valid_data, batch_size=n)
valid_N = len(valid_loader.dataset)


class MyConvBlock(nn.Module):
	def __init__(self, in_ch, out_ch, dropout_p):
		kernel_size = 3
		super().__init__()

		self.model = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(),
			nn.Dropout(dropout_p),
			nn.MaxPool2d(2, stride=2)
		)

	def forward(self, x):
		return self.model(x)


flattened_img_size = 75 * 3 * 3

base_model = nn.Sequential(
	MyConvBlock(IMG_CHS, 25, 0),
	MyConvBlock(25, 50, 0.2),
	MyConvBlock(50, 75, 0),
	nn.Flatten(),
	nn.Linear(flattened_img_size, 512),
	nn.Dropout(.3),
	nn.ReLU(),
	nn.Linear(512, N_CLASSES)
)

loss_function = nn.CrossEntropyLoss()
optimizer = Adam(base_model.parameters())

model = base_model.to(device)

random_transforms = transforms.Compose([
	transforms.RandomRotation(5),
	transforms.RandomResizedCrop((IMG_WIDTH, IMG_HEIGHT), scale=(.9, 1), ratio=(1, 1)),
	transforms.RandomHorizontalFlip(),
	transforms.ColorJitter(brightness=.2, contrast=.5)
])


def train():
	loss = 0
	accuracy = 0

	model.train()
	for x, y in train_loader:
		output = model(random_transforms(x))
		optimizer.zero_grad()
		batch_loss = loss_function(output, y)
		batch_loss.backward()
		optimizer.step()

		loss += batch_loss.item()
		accuracy += utils.get_batch_accuracy(output, y, train_N)
	print('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))


def validate():
	loss = 0
	accuracy = 0

	model.eval()
	with torch.no_grad():
		for x, y in valid_loader:
			output = model(x)

			loss += loss_function(output, y).item()
			accuracy += utils.get_batch_accuracy(output, y, valid_N)
	print('Valid - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))


if __name__ == "__main__":
	epochs = 20

	for epoch in range(epochs):
		print('Epoch: {}'.format(epoch))
		train()
		validate()

	# Saved to app/, matching where vision_model.py loads from
	model_name = "newest_try" + ".pth"
	torch.save(model.state_dict(), SCRIPT_DIR.parent / model_name)
	print("Saved to", SCRIPT_DIR.parent / model_name)