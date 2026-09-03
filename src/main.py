"""Real-time CS:GO player detection and automated target interaction.

Educational computer-vision experiment built around a custom YOLO model.
"""

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import mss
import numpy as np
import pyautogui
from ultralytics import YOLO


@dataclass(frozen=True)
class Config:
    model_path: Path = Path(__file__).resolve().parents[1] / "weights" / "best.pt"
    capture_left: int = 1289
    capture_top: int = 0
    capture_width: int = 629
    capture_height: int = 499
    confidence_threshold: float = 0.57354
    target_height_ratio: float = 1 / 3
    action_cooldown: float = 0.08
    max_targets_per_frame: int = 1
    show_preview: bool = True

    @property
    def monitor(self) -> dict[str, int]:
        return {
            "left": self.capture_left,
            "top": self.capture_top,
            "width": self.capture_width,
            "height": self.capture_height,
        }


def choose_team() -> int:
    while True:
        value = input("Enter 1 if you play Terrorist, or 0 if you play Counter-Terrorist: ").strip()
        if value in {"0", "1"}:
            return int(value)
        print("Please enter 0 or 1.")


def load_model(config: Config) -> YOLO:
    if not config.model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {config.model_path}")
    return YOLO(str(config.model_path))


def capture_frame(sct: mss.mss, monitor: dict[str, int]) -> np.ndarray:
    screenshot = np.asarray(sct.grab(monitor))
    return cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)


def get_target_point(box: np.ndarray, height_ratio: float) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    target_x = x1 + (x2 - x1) / 2
    target_y = y1 + (y2 - y1) * height_ratio
    return round(target_x), round(target_y)


def select_targets(result, own_team: int, config: Config) -> list[tuple[float, np.ndarray, int]]:
    candidates = []
    if result.boxes is None:
        return candidates

    for box in result.boxes:
        confidence = float(box.conf[0])
        if confidence < config.confidence_threshold:
            continue

        class_id = int(box.cls[0])
        if class_id == own_team:
            continue

        xyxy = box.xyxy[0].cpu().numpy()
        candidates.append((confidence, xyxy, class_id))

    # Highest-confidence detections are preferred. Limit actions per frame.
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[: config.max_targets_per_frame]


def draw_detections(frame: np.ndarray, result, model: YOLO, threshold: float) -> None:
    if result.boxes is None:
        return

    for box in result.boxes:
        confidence = float(box.conf[0])
        if confidence < threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        label = model.names[class_id]
        text = f"{label}: {confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            text,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )


def interact_with_target(
    target: tuple[float, np.ndarray, int], config: Config, last_action: float
) -> float:
    now = time.perf_counter()
    if now - last_action < config.action_cooldown:
        return last_action

    _, box, _ = target
    target_x, target_y = get_target_point(box, config.target_height_ratio)

    # Convert coordinates from the captured region to desktop coordinates.
    screen_x = config.capture_left + target_x
    screen_y = config.capture_top + target_y

    pyautogui.click(x=screen_x, y=screen_y)
    return now


def run() -> None:
    config = Config()
    own_team = choose_team()
    model = load_model(config)
    last_action = 0.0

    print(f"Loaded model: {config.model_path}")
    print("Detection started. Press Q in the preview window to stop.")

    with mss.mss() as sct:
        while True:
            frame = capture_frame(sct, config.monitor)
            results = model(frame, verbose=False)

            for result in results:
                draw_detections(frame, result, model, config.confidence_threshold)
                targets = select_targets(result, own_team, config)
                if targets:
                    last_action = interact_with_target(targets[0], config, last_action)

            if config.show_preview:
                cv2.imshow("CS:GO Player Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
