# God's Eye

AI-powered multi-camera person tracking demo for recorded mobile-camera footage.

## MVP Goal

The first working version focuses on a final-year-project demo:

1. Detect people in images and videos.
2. Track simple movement direction inside one camera.
3. Use camera-flow rules and time windows for nearby non-overlapping cameras.
4. Ask for user confirmation when cross-camera matching is uncertain.
5. Build a timeline from camera events.

## Recommended Demo Setup

Use 4 mobile cameras with short recorded clips:

```text
Cam1: Entry area
Cam2: Corridor or corner
Cam3: Exit side
Cam4: Alternate path
```

Keep videos short at first, ideally 30 seconds to 2 minutes, at 480p or 640p.

## Install

Create a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If Windows says `python` is not recognized, install Python 3.10 or newer and
enable "Add Python to PATH" during installation. Inside this Codex workspace, the
bundled Python runtime can also run the tests.

The AI detector uses Ultralytics YOLO. By default the commands expect a YOLO
model such as `yolo11n.pt`. If it is not already present, Ultralytics may try to
download it.

## Run Image Detection

## Dashboard-First Demo

Put recordings in `datasets\video\cam1.mp4` through `datasets\video\cam8.mp4`, then run:

```powershell
python main.py
```

The dashboard opens with the footage list from `datasets\video`. Click one file, close the dashboard automatically, then the program analyzes footage in sequence with a live visible OpenCV window. Each run is cleaned up before the next footage starts so the player terminates cleanly.

### User Guidance

- Keep your source footage in `datasets/video`.
- Start the app with `python main.py`.
- Choose one footage file from the dashboard.
- Wait for the dashboard to close before analysis begins.
- Watch the live video window while the file is being analyzed.
- When the gallery appears, choose the main target person.
- If the same real person appears as another gallery tile, select that too in the duplicate-check window.
- Let the player finish and close before the next footage starts.

Face comparison is used when the optional `face-recognition` package can detect a clear face; body/clothing appearance remains the fallback for side views and distant people.

```powershell
python main.py detect-image --image path\to\person.jpg --camera-id cam1 --model yolo11n.pt --out result\image\image_detections.jsonl --annotated result\image\image_annotated.jpg
```

## Run Video Detection

```powershell
python main.py detect-video --video path\to\cam1.mp4 --camera-id cam1 --model yolo11n.pt --every 3 --out result\video\cam1_detections.jsonl --annotated result\video\cam1_annotated.mp4
```

`--every 3` processes every third frame, which is better for laptops without a
strong GPU.

The video command reports unique tracked people, not only frame detections:

```text
Detected 1 unique person(s).
Frame detections: 146 saved to result/video/video_detections.jsonl
Track summary: result/video/video_tracks.jsonl
```

`result/video/video_tracks.jsonl` stores each person's approximate entry time, exit time,
duration, direction, and first/last zone.

For an exhibition/demo, show the annotated video while detection runs:

```powershell
python main.py detect-video --video path\to\cam1.mp4 --camera-id cam1 --model yolo11n.pt --every 1 --show --hold-window --annotated result\video\cam1_annotated.mp4
```

Press `q` to close the video window while processing. With `--hold-window`, the
last frame stays open after processing until you press any key. The bounding-box
label includes tracking ID, person label, movement direction, and confidence,
for example:

```text
ID 1 person right 0.88
```

To play a saved output video later:

```powershell
python main.py play-video --video result\video\cam1_annotated.mp4
```

## Compare Two Cameras

Process each camera video with separate output names:

```powershell
python main.py detect-video --video datasets\video\cam1.mp4 --camera-id cam1 --every 3 --annotated result\video\cam1_output.mp4
python main.py detect-video --video datasets\video\cam2.mp4 --camera-id cam2 --every 3 --annotated result\video\cam2_output.mp4
```

By default, output files use the camera ID:

```text
result/video/cam1_detections.jsonl
result/video/cam1_tracks.jsonl
result/video/cam1_summary.txt

result/video/cam2_detections.jsonl
result/video/cam2_tracks.jsonl
result/video/cam2_summary.txt
```

Then compare the track files:

```powershell
python main.py compare-cameras --config configs\cameras.example.json --source-tracks result\video\cam1_tracks.jsonl --target-tracks result\video\cam2_tracks.jsonl --out result\video\cam1_cam2_match_report.txt
```

The report uses camera flow rules, exit/entry zones, and time gap to decide:

```text
AUTO MATCH
ASK USER TO CONFIRM
REJECT
```

Weak tracks are filtered by default. If a video has poor lighting and misses a
real person, lower the threshold:

```powershell
python main.py detect-video --video path\to\cam1.mp4 --camera-id cam1 --min-track-confidence 0.45
```

For easy reading, the command also writes:

```text
result/video/video_tracks_summary.txt
```

Example:

```text
Camera: cam1
Unique Persons: 1

Person ID: 1
Entered: 0.0 sec
Exited: 12.4 sec
Duration: 12.4 sec
Direction: right
Entry Zone: left
Exit Zone: right
Detections: 146
Average Confidence: 0.87
```

## Run Tests

```powershell
python -m unittest discover -s tests
```

## Current Structure

```text
godseye/
  detection/      YOLO person detection wrapper
  matching/       camera-flow and time-window matching
  movement/       direction and zone analysis
  storage/        JSONL event output
  timeline/       timeline builder
  video/          recorded-video processing
datasets/
  image/         still-image inputs
  video/         recorded-video inputs
result/
  image/         image analysis outputs
  video/         video analysis outputs
configs/          example camera graph
tests/            logic tests that do not require YOLO
```
