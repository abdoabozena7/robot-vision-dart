# YOLO Video Detection And Counting

This project opens a small desktop GUI first, lets you pick any video, shows live minimal annotations while YOLO is processing, and can send the computed counts to a local Ollama model for a simple English analysis.

## What it saves

- An annotated video with bounding boxes and a live count panel
- A CSV file with per-frame counts
- A JSON summary with max counts and approximate unique tracked objects per class

## Setup

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the GUI

```powershell
.\.venv\Scripts\python.exe .\detect_and_count.py
```

Use `Pick Video` to choose the file. The live preview opens in a separate window.
When processing finishes, click `Analyze` to get a simple English summary from your local Ollama model.
Outputs are written to `output/`. Press `q` to stop the live preview window.

## Run directly from the command line

```powershell
.\.venv\Scripts\python.exe .\detect_and_count.py --source .\vlog.mp4
```

## Useful options

```powershell
.\.venv\Scripts\python.exe .\detect_and_count.py `
  --source .\vlog.mp4 `
  --model yolo11s.pt `
  --classes person car dog `
  --conf 0.4
```

To run without opening the preview window:

```powershell
.\.venv\Scripts\python.exe .\detect_and_count.py --source .\vlog.mp4 --no-show
```

## Notes

- `unique_tracked_objects` is approximate. It depends on tracker quality.
- If you want better accuracy, use a larger model like `yolo11s.pt` or `yolo11m.pt`.
- If processing is slow, try `--skip-frames 1` or lower `--imgsz`.
- `Analyze` now prefers `gpt-oss:120b-cloud` by default when it exists in `ollama list`. You can override the model with `--ollama-model your-model-name`.
