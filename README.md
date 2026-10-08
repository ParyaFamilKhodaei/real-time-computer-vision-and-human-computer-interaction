# 👁️ Real-Time Player Detection and Automated Targeting in Counter Strike 1.6

A small computer-vision experiment built around a custom YOLO model trained to distinguish Terrorist and Counter-Terrorist players in Counter-Strike: Global Offensive.

The interesting part of the project is not just the detector. The project connects model inference to a complete real-time pipeline: screen capturing, object detection, class filtering, target localization, coordinate transformation, and automated mouse interaction.
https://github.com/user-attachments/assets/d4007c49-dee0-4a0a-ace8-373cf0f08b91
> **Educational note:** This repository is intended for computer-vision experimentation and should only be used where automated input is permitted.

## 🕹️ Pipeline

```text
Screen capture
      ↓
YOLO inference
      ↓
Confidence filtering
      ↓
Team filtering
      ↓
Target selection
      ↓
Target-point localization
      ↓
Crop + desktop coordinate mapping
      ↓
Mouse interaction
```

## 🌱 What my first implementation did

The first version of the project used `PIL.ImageGrab` to capture a fixed region of the screen, passed every frame to a custom YOLO model, filtered detections by confidence, removed detections belonging to the player's own team, calculated a point in the upper part of each remaining bounding box, and used the resulting desktop coordinates for mouse input.

This version keeps that core idea but makes the pipeline easier to understand, configure, and extend.

## 🔧 Refactoring changes

### 1. Portable model path

The original code used a machine-specific path such as:

```python
C:\\Users\\Parya\\OneDrive\\Desktop\\csgo\\best.pt
```

The refactored version expects the weights at:

```text
weights/best.pt
```

and resolves the path relative to the project itself. The repository can therefore be moved to another machine without editing the Python source.

### 2. Faster screen capture with MSS

The previous implementation imported `mss` but used `PIL.ImageGrab`. The refactored version uses `mss` consistently and converts the captured BGRA frame to BGR before passing it to OpenCV/YOLO.

### 3. Clear separation of responsibilities

The main loop is now deliberately small. Individual functions handle:

- team selection
- model loading
- screen capture
- target-point calculation
- target filtering/selection
- visualization
- mouse interaction

This makes individual parts easier to test and replace.

### 4. Target selection

The first implementation acted on every enemy detection in a frame. The new version ranks valid candidates by confidence and limits interaction to the highest-confidence target in each frame.

### 5. Action cooldown

A small configurable cooldown prevents the interaction layer from issuing actions continuously on every processed frame. Detection and interaction are therefore no longer effectively tied to the same frequency.

### 6. Coordinate handling

My previous code mixed the capture origin with a hard-coded `1280` x-offset even though the capture started at `1289`. This version uses the capture origin itself for the transformation:

```python
screen_x = capture_left + target_x
screen_y = capture_top + target_y
```

This removes the hidden offset and makes the coordinate system explicit.

### 7. Configuration in one place

Capture geometry, confidence threshold, target-point position, cooldown, and target count are stored in `Config` instead of being scattered through the loop.

## 📊 Model and dataset

The supplied dataset configuration contains two classes:

```yaml
names:
  0: ct
  1: t
```

with separate training and validation image directories.

The trained weights are stored locally as `weights/best.pt` and are not reproduced by this repository from scratch; they represent the result of the model-training stage of the project.


## 🎮 Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Then:

```bash
pip install -r requirements.txt
```

Place the trained model at:

```text
weights/best.pt
```

Run:

```bash
python src/main.py
```

The program asks which team is being played:

```text
0 → Counter-Terrorist
1 → Terrorist
```

Press `Q` in the preview window to stop the program.

## ⚙️ Configuration

The main parameters are defined in `Config`:

| Parameter | Purpose |
|---|---|
| `confidence_threshold` | Minimum detection confidence |
| `capture_left` / `capture_top` | Screen-capture origin |
| `capture_width` / `capture_height` | Captured region size |
| `target_height_ratio` | Vertical position of the target point inside a box |
| `action_cooldown` | Minimum time between mouse actions |
| `max_targets_per_frame` | Maximum targets acted upon per frame |

The current capture region is based on the setup used during development and may need to be changed for a different display configuration.

## 🤖 Technical details

### Detection

Each captured frame is passed to the trained YOLO model. For every detection, the model provides a bounding box, confidence score, and class ID.

Only detections above the configured confidence threshold are considered for downstream processing.

### Team filtering

The two classes correspond to the two teams. The player's selected team is excluded from the candidate set, leaving detections belonging to the opposing class.

### Target localization

Rather than using a corner of the bounding box, the interaction point is calculated from the box center horizontally and a configurable fraction of its height vertically:

```python
target_x = x1 + (x2 - x1) / 2
target_y = y1 + (y2 - y1) * target_height_ratio
```

### Coordinate transformation

YOLO works in coordinates relative to the captured frame. Mouse input uses desktop coordinates. The system therefore adds the capture region's desktop origin to the detected point before sending the interaction.

## 📈 Limitations and next steps

The current project is intentionally compact. possible further development areas:

- quantitative evaluation of precision, recall, and false positives
- FPS and end-to-end latency measurement
- temporal tracking across frames
- configurable target-ranking strategies
- separating inference frequency from interaction frequency
- moving runtime parameters to a YAML/TOML configuration file
- unit tests for coordinate conversion and target selection
- logging detection statistics for reproducible experiments

These changes would turn the prototype into a more general real-time object-detection framework rather than a game-specific script.

## 💡 Why this project matters

This project gave me practical experience with a part of computer vision that is easy to overlook: the gap between a model making a prediction and a real application doing something useful with that prediction.

The system has to deal with continuously changing visual input, imperfect predictions, coordinate systems, latency, filtering, and downstream actions. Working through those issues helped me understand how a trained model can become one component of a larger real-time software system.
