import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import torch
import torch.nn as nn
from ASL_model.model import initialize_model
import torchvision.io as tv_io
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
from utils import write_to_camera

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#functionify the main loop bitch

# Download the hand landmarker model if not already present
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
	print("Downloading hand landmarker model")
	url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
	try:
		urllib.request.urlretrieve(url, model_path)
	except:
		print("Model could not be downloaded")
	else:
		print("Model downloaded successfully")

translator_model = initialize_model()

IMG_WIDTH = 28 # based on expected input from inference model
IMG_HEIGHT = 28

preprocess_trans = transforms.Compose([
    transforms.ToDtype(torch.float32, scale=True), # Converts [0, 255] to [0, 1]
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
])

# Create HandLandmarker options
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
	base_options=base_options,
	num_hands=2, # default 2
	min_hand_detection_confidence=0.5,	# default 0.5
	min_hand_presence_confidence=0.5, 	# default 0.5
	min_tracking_confidence=0.5 		# default 0.5
)

padding = 70 # default 20 to get just the whole hand with no extra padding

# Alphabet does not contain j or z because they require movement
ALPHABET = "abcdefghiklmnopqrstuvwxy1" # 1 means none/unknown 

CONFIDENCE_THRESHOLD = 0.85 # model output probability threshold to interpret a letter

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 7200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

WINDOW_NAME = 'ASL translator'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# Create the hand landmarker
alive = True
with vision.HandLandmarker.create_from_options(options) as landmarker:
	
	while alive:
		success, frame = cap.read()
		if not success:
			print("Failed to read from camera")
			break
		# check for available camera or request camera access?
		
		# Flip frame horizontally for mirror effect ( moving hand left moves hand left on screen )
		frame = cv2.flip(frame, 1)
		
		# Convert BGR to RGB and collect hand info
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
		detection_result = landmarker.detect(mp_image)
		h, w, c = frame.shape
		
		# Draw bounding boxes around detected hands
		if detection_result.hand_landmarks:
			for hand_landmarks in detection_result.hand_landmarks:
				# Get all landmark coordinates
				x_coords = [lm.x * w for lm in hand_landmarks]
				y_coords = [lm.y * h for lm in hand_landmarks]
				
				# Calculate bounding box coordinates
				x_min = int(min(x_coords))
				x_max = int(max(x_coords))
				y_min = int(min(y_coords))
				y_max = int(max(y_coords))
				
				# Add padding to the bounding box to get whole hand and then some
				x_min = max(0, x_min - padding)
				x_max = min(w, x_max + padding)
				y_min = max(0, y_min - padding)
				y_max = min(h, y_max + padding)

				# use rectangle to crop frame for preprocessing
				cropped_hand = frame[y_min:y_max, x_min:x_max] # [start_y:end_y, start_x:end_x]

				# draw hand onto frame for reference
				h, w = (200, 200)
				padding_x, padding_y = (30, 30)
				resized_hand = cv2.resize(cropped_hand, (h, w))
				frame[0+padding_y:h+padding_y, frame.shape[1]-w-padding_x:frame.shape[1]-padding_x] = resized_hand

				# Draw rectangle around hand
				cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

				# preprocess the hand for input to model (expects input of form torch.Size([1, 1, 28, 28]))
				cropped_hand_gray = cv2.cvtColor(cropped_hand, cv2.COLOR_BGR2GRAY) # greyscale
				cropped_hand_tensor = torch.from_numpy(cropped_hand_gray)      # [H, W]         (base from numpy)
				cropped_hand_tensor = cropped_hand_tensor.unsqueeze(0)         # [1, H, W]      (channel dimension added, this is what the preprocess_trans will expect as input)
				processed_image = preprocess_trans(cropped_hand_tensor)        # [1, 28, 28]    (resized)
				batched_image = processed_image.unsqueeze(0)                   # [1, 1, 28, 28] (added batch dimension)

				# make sure both model and intake are on the same device
				batched_image_gpu = batched_image.to(device)

				# send input to model to recieve output
				output = translator_model(batched_image_gpu)
				probs = torch.softmax(output, dim=1)          # softmax to turn logit outputs to percentage for confidence thresholding
				max_prob, max_index = probs.max(dim=1)
				conf = max_prob.item()
				prediction = max_index.item()
				letter = ALPHABET[prediction]

				# save coords for later
				# if conf >= CONFIDENCE_THRESHOLD and letter != '1':
				# 	write_to_camera(letter, frame, (0, 0), fill=True)
				# else:
				# 	write_to_camera("unrecognized", frame, (0, 0), fill=True)

				if conf >= CONFIDENCE_THRESHOLD and letter != '1':
					write_to_camera(letter, frame, (frame.shape[1]-w+(2*padding_x), h+int(2*padding_y)), fill=True)
				else:
					write_to_camera("unrecognized", frame, (frame.shape[1]-w, h+int(2*padding_y)), fill=True)
				
				# Draw hand landmarks (dont need them though)
				# for landmark in hand_landmarks:
				#     x = int(landmark.x * w)
				#     y = int(landmark.y * h)
				#     cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
		
		# Display the frame
		cv2.imshow(WINDOW_NAME, frame)
		
		# Exit
		key = cv2.waitKey(1)
		if key == ord("Q") or key == ord("q") or key == 27:
			alive = False
		elif cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
			alive = False

# Release resources
cap.release()
cv2.destroyAllWindows()