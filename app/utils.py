import cv2
import numpy as np

def write_to_camera(text: str, frame: np.ndarray, topleft_coords: tuple[int, int], text_font: int = 1, text_color: tuple[int, int, int] = (255, 255, 255), fill: bool = False, fill_color: tuple[int, int, int] = (0, 0, 0)):
	"""write text to camera with specified options"""
	(text_w, text_h), baseline = cv2.getTextSize(
		text,
		text_font,
		3.0,
		6
	)

	x, y = topleft_coords

	if fill:
		cv2.rectangle(
			frame,
			(x, y - text_h),
			(x + text_w, y + baseline),
			fill_color,
			-1  # filled
		)

	cv2.putText(
		frame,
		text,
		(x, y),
		text_font,
		3.0,
		text_color,
		6,
		cv2.LINE_AA
)