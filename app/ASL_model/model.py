import torch
import torch.nn as nn

# this file rebuilds the model identical to how it was created from the model_weights.pth file

IMG_CHS = 1
IMG_WIDTH = 28
IMG_HEIGHT = 28
N_CLASSES = 24
flattened_img_size = 75 * 3 * 3

class MyConvBlock(nn.Module):
	def __init__(self, in_ch, out_ch, dropout_p):
		super().__init__()
		self.model = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(),
			nn.Dropout(dropout_p),
			nn.MaxPool2d(2, stride=2)
		)

	def forward(self, x):
		return self.model(x)


def create_model():
	return nn.Sequential(
		MyConvBlock(IMG_CHS, 25, 0),     # 25 x 14 x 14
		MyConvBlock(25, 50, 0.2),        # 50 x 7 x 7
		MyConvBlock(50, 75, 0),          # 75 x 3 x 3
		nn.Flatten(),
		nn.Linear(flattened_img_size, 512),
		nn.Dropout(0.3),
		nn.ReLU(),
		nn.Linear(512, N_CLASSES)
	)


def initialize_model(): # create a model instance based on the weights file
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = create_model().to(device)

	state_dict = torch.load("model_weights.pth", map_location=device)
	model.load_state_dict(state_dict)

	model.eval()  # IMPORTANT for inference
	return model