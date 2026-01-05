import cv2
import numpy as np

def write_to_camera(text: str, frame: np.ndarray, topleft_coords: tuple[int, int] = (0, 0), text_font: int = 1, text_color: tuple[int, int, int] = (255, 255, 255), fill: bool = False, fill_color: tuple[int, int, int] = (0, 0, 0), text_scale: int = 1):
	"""write text to camera with specified options"""

	padding_x = 20
	padding_y = 15

	(text_w, text_h), baseline = cv2.getTextSize(
		text,
		text_font,
		text_scale,
		2
	)
	x = (int((frame.shape[1]) // 2) - int(text_w // 2) - padding_x) if topleft_coords[0] == 0 else topleft_coords[0] # default to center
	y = int((frame.shape[0]) * 0.8) if topleft_coords[1] == 0 else topleft_coords[1] # default to 70% down

	if fill:
		cv2.rectangle(
			frame,
			(x, y),
			(x + padding_x + text_w + padding_x, y + padding_y + text_h + padding_y),
			fill_color,
			-1  # filled
		)

	cv2.putText(
		frame,
		text,
		(x + padding_x, y + text_h + padding_y),
		text_font,
		text_scale,
		text_color,
		2,
		cv2.LINE_AA
)