import cv2
import numpy as np
import time
from collections import Counter

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

def get_smoothed_prediction(pred_window, min_votes):
    valid_preds = [p for p in pred_window if p is not None]
    if not valid_preds:
        return None

    counts = Counter(valid_preds)
    label, votes = counts.most_common(1)[0]

    if votes >= min_votes:
        return label

    return None


class SessionStats:
	"""
	Tracks live benchmarking data during a webcam session:
	  - per-frame inference latency (avg / p95)
	  - raw per-frame prediction flicker (how often the unsmoothed
	    label changes frame-to-frame)
	  - smoothed/committed letter count (how many letters temporal
	    smoothing actually confirmed)

	Comparing raw_transitions to committed_letters gives a flicker
	reduction percentage without needing to run two separate sessions,
	since the raw prediction is computed every frame regardless of
	whether smoothing is applied downstream.
	"""

	def __init__(self):
		self.session_start = time.perf_counter()
		self.frame_count = 0
		self.raw_transitions = 0
		self.committed_letters = 0
		self.prev_raw_label = None
		self.inference_times = []  # seconds

	def log_frame(self, raw_label, inference_time_sec: float):
		"""Call once per processed frame with the raw (pre-smoothing) label
		('unrecognized' or the letter) and the model inference duration."""
		self.frame_count += 1
		self.inference_times.append(inference_time_sec)

		if self.prev_raw_label is not None and raw_label != self.prev_raw_label:
			self.raw_transitions += 1
		self.prev_raw_label = raw_label

	def log_committed_letter(self):
		"""Call each time temporal smoothing confirms a letter into the caption."""
		self.committed_letters += 1

	def summary(self) -> str:
		if not self.inference_times:
			return "No frames were processed this session."

		elapsed = time.perf_counter() - self.session_start
		avg_ms = (sum(self.inference_times) / len(self.inference_times)) * 1000
		sorted_times = sorted(self.inference_times)
		p95_index = min(int(len(sorted_times) * 0.95), len(sorted_times) - 1)
		p95_ms = sorted_times[p95_index] * 1000
		fps = self.frame_count / elapsed if elapsed > 0 else 0

		flicker_reduction = None
		if self.raw_transitions > 0:
			flicker_reduction = (1 - (self.committed_letters / self.raw_transitions)) * 100

		lines = [
			"\n--- Session Benchmark Summary ---",
			f"Session duration: {elapsed:.1f}s",
			f"Frames processed: {self.frame_count} ({fps:.1f} fps)",
			f"Avg inference latency: {avg_ms:.2f} ms",
			f"P95 inference latency: {p95_ms:.2f} ms",
			f"Raw per-frame label transitions: {self.raw_transitions}",
			f"Smoothed/committed letters: {self.committed_letters}",
		]
		if flicker_reduction is not None:
			lines.append(f"Flicker reduction from smoothing: {flicker_reduction:.1f}%")
		lines.append("----------------------------------\n")
		return "\n".join(lines)