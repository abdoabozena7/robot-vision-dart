from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import re
import subprocess
import threading
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, scrolledtext
from typing import Callable, NamedTuple

import tkinter as tk

import cv2
from ultralytics import YOLO


WINDOW_NAME = "Detection Preview"
DEFAULT_OLLAMA_MODEL = "gpt-oss:120b-cloud"


class PreviewLayout(NamedTuple):
    width: int
    height: int
    x: int
    y: int


@dataclass
class ProcessingResult:
    source: Path
    summary: dict
    summary_path: Path
    csv_path: Path
    annotated_video_path: Path


@dataclass(frozen=True)
class AnnotationStyle:
    box_thickness: int
    shadow_thickness: int
    corner_length: int
    label_font_scale: float
    label_thickness: int
    label_padding_x: int
    label_padding_y: int
    panel_font_scale: float
    panel_thickness: int
    panel_line_height: int
    panel_width: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO detection + tracking on a video and count objects by class."
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Path to the input video file. If omitted, the desktop GUI opens first.",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="YOLO model weights. Example: yolo11n.pt, yolo11s.pt, or a local .pt file.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for the annotated video and count reports.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS/tracking.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Inference image size.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Torch device, for example "cpu", "0", or "0,1".',
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Ultralytics tracker config.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help='Optional class filter by name or id. Example: --classes person car 0',
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="Process every N+1 frame. Example: 2 means process 1 frame and skip 2.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional limit for quick tests. 0 means process the full video.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable the live preview window.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open the desktop GUI even if --source is provided.",
    )
    parser.add_argument(
        "--ollama-model",
        default=os.getenv("OLLAMA_MODEL"),
        help="Optional Ollama model override for the Analyze button.",
    )
    return parser.parse_args()


def normalize_names(names: dict[int, str] | list[str]) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(class_id): str(name) for class_id, name in names.items()}
    return {index: str(name) for index, name in enumerate(names)}


def resolve_classes(raw_classes: list[str] | None, class_names: dict[int, str]) -> list[int] | None:
    if not raw_classes:
        return None

    name_to_id = {name.lower(): class_id for class_id, name in class_names.items()}
    resolved: list[int] = []

    for item in raw_classes:
        if item.isdigit():
            resolved.append(int(item))
            continue

        class_id = name_to_id.get(item.lower())
        if class_id is None:
            known = ", ".join(sorted(name_to_id)[:12])
            raise ValueError(f'Unknown class "{item}". Known examples: {known}')
        resolved.append(class_id)

    return sorted(set(resolved))


def ensure_video(source: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")


def build_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


def sorted_counts(counts: Counter[str]) -> list[tuple[str, int]]:
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def color_for_class(class_name: str) -> tuple[int, int, int]:
    palette = [
        (0, 214, 255),
        (80, 255, 170),
        (255, 184, 77),
        (255, 107, 107),
        (188, 140, 255),
        (255, 230, 109),
    ]
    return palette[sum(ord(char) for char in class_name) % len(palette)]


def compact_counts_line(title: str, counts: Counter[str], empty_text: str) -> str:
    items = sorted_counts(counts)
    if not items:
        return f"{title} {empty_text}"

    preview = "  ".join(f"{name} {count}" for name, count in items[:4])
    if len(items) > 4:
        preview = f"{preview}  +{len(items) - 4}"
    return f"{title} {preview}"


def draw_label(
    frame,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    style: AnnotationStyle,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = style.label_font_scale
    thickness = style.label_thickness
    padding_x = style.label_padding_x
    padding_y = style.label_padding_y
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    label_x1 = max(x, 6)
    label_y2 = max(y, text_height + padding_y * 2 + 6)
    label_x2 = label_x1 + text_width + padding_x * 2
    label_y1 = label_y2 - (text_height + padding_y * 2 + baseline)

    overlay = frame.copy()
    cv2.rectangle(overlay, (label_x1, label_y1), (label_x2, label_y2), (14, 14, 14), -1)
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, max(style.box_thickness - 1, 1), cv2.LINE_AA)
    cv2.putText(
        frame,
        text,
        (label_x1 + padding_x, label_y2 - padding_y - baseline + 1),
        font,
        font_scale,
        (245, 245, 245),
        thickness,
        cv2.LINE_AA,
    )


def draw_box_corners(frame, x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int], length: int, thickness: int) -> None:
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness, cv2.LINE_AA)


def draw_minimal_detections(
    frame,
    result,
    class_names: dict[int, str],
    style: AnnotationStyle,
) -> None:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return

    xyxy_list = boxes.xyxy.int().cpu().tolist()
    class_ids = boxes.cls.int().cpu().tolist()
    track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(class_ids)

    for xyxy, class_id, track_id in zip(xyxy_list, class_ids, track_ids):
        x1, y1, x2, y2 = xyxy
        class_name = class_names.get(int(class_id), str(class_id))
        color = color_for_class(class_name)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (10, 10, 10), style.shadow_thickness, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, style.box_thickness, cv2.LINE_AA)
        draw_box_corners(frame, x1, y1, x2, y2, color, style.corner_length, style.box_thickness + 1)

        label = class_name
        if track_id is not None:
            label = f"{class_name} #{track_id}"
        draw_label(frame, label, x1, y1 - 8, color, style)


def draw_panel(
    frame,
    frame_counts: Counter[str],
    unique_counts: Counter[str],
    frame_number: int,
    timestamp_sec: float,
    style: AnnotationStyle,
) -> None:
    lines = [
        f"frame {frame_number}   {timestamp_sec:.2f}s",
        compact_counts_line("live", frame_counts, "none"),
        compact_counts_line("total", unique_counts, "tracking..."),
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = style.panel_font_scale
    thickness = style.panel_thickness
    line_height = style.panel_line_height
    box_width = style.panel_width
    box_height = 16 + line_height * len(lines)

    overlay = frame.copy()
    cv2.rectangle(overlay, (14, 14), (14 + box_width, 14 + box_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.42, frame, 0.58, 0, frame)

    y = 35
    for line in lines:
        cv2.putText(frame, line, (26, y), font, font_scale, (248, 248, 248), thickness, cv2.LINE_AA)
        y += line_height


def extract_counts(result, class_names: dict[int, str], seen_tracks: set[tuple[int, int]], unique_counts: Counter[str]) -> Counter[str]:
    frame_counts: Counter[str] = Counter()
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return frame_counts

    class_ids = boxes.cls.int().cpu().tolist()
    track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(class_ids)

    for class_id, track_id in zip(class_ids, track_ids):
        class_name = class_names.get(int(class_id), str(class_id))
        frame_counts[class_name] += 1

        if track_id is None:
            continue

        track_key = (int(class_id), int(track_id))
        if track_key in seen_tracks:
            continue

        seen_tracks.add(track_key)
        unique_counts[class_name] += 1

    return frame_counts


def write_frame_counts_csv(output_path: Path, rows: list[dict], class_names: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["frame_number", "timestamp_sec", *class_names]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            payload = {
                "frame_number": row["frame_number"],
                "timestamp_sec": f'{row["timestamp_sec"]:.4f}',
            }
            payload.update({name: row["counts"].get(name, 0) for name in class_names})
            writer.writerow(payload)


def build_simple_summary_text(summary: dict) -> str:
    max_visible = summary.get("max_visible_per_frame", {})
    tracked_total = summary.get("unique_tracked_objects", {})
    tracking_enabled = summary.get("tracking_enabled", False)

    lines = [
        f'Video: {Path(summary.get("source_video", "unknown")).name}',
        f'Frames checked: {summary.get("processed_frames", 0)}',
    ]

    if max_visible:
        visible_text = ", ".join(
            f"{name} up to {count} at one time" for name, count in list(max_visible.items())[:5]
        )
        if len(max_visible) > 5:
            visible_text = f"{visible_text}, and more"
        lines.append(f"Main things seen on screen: {visible_text}.")
    else:
        lines.append("Main things seen on screen: nothing clear was detected.")

    if tracking_enabled and tracked_total:
        tracked_text = ", ".join(
            f"{name} about {count}" for name, count in list(tracked_total.items())[:5]
        )
        if len(tracked_total) > 5:
            tracked_text = f"{tracked_text}, and more"
        lines.append(f"Estimated total objects in the video: {tracked_text}.")
    elif tracking_enabled:
        lines.append("Estimated total objects in the video: no stable tracked objects were counted.")
    else:
        lines.append("Estimated total objects in the video: tracking was off, so only live frame counts are available.")

    lines.append("Note: totals are estimates, not exact counts.")
    return "\n".join(lines)


def get_screen_size() -> tuple[int, int]:
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return width, height
    except Exception:
        return 1280, 720


def build_preview_layout(frame_width: int, frame_height: int) -> PreviewLayout:
    screen_width, screen_height = get_screen_size()
    max_width = int(screen_width * 0.7)
    max_height = int(screen_height * 0.7)

    scale = min(max_width / frame_width, max_height / frame_height, 1.0)
    preview_width = max(320, int(frame_width * scale))
    preview_height = max(240, int(frame_height * scale))

    preview_width = min(preview_width, screen_width)
    preview_height = min(preview_height, screen_height)

    x = max((screen_width - preview_width) // 2, 0)
    y = max((screen_height - preview_height) // 2, 0)
    return PreviewLayout(preview_width, preview_height, x, y)


def compute_annotation_style(
    frame_width: int,
    frame_height: int,
    preview_layout: PreviewLayout | None,
) -> AnnotationStyle:
    preview_width = preview_layout.width if preview_layout is not None else frame_width
    preview_height = preview_layout.height if preview_layout is not None else frame_height
    downscale = min(preview_width / frame_width, preview_height / frame_height, 1.0)
    display_min = max(int(min(frame_width * downscale, frame_height * downscale)), 240)

    target_label_font = min(max(display_min / 1200.0, 0.42), 0.72)
    label_font_scale = min(max(target_label_font / downscale, 0.42), 1.55)

    target_box_thickness = min(max(round(display_min / 320.0), 1), 3)
    box_thickness = min(max(int(round(target_box_thickness / downscale)), 1), 8)

    target_corner_length = min(max(int(round(display_min / 22.0)), 16), 26)
    corner_length = min(max(int(round(target_corner_length / downscale)), 18), 60)

    target_panel_font = min(max(display_min / 1350.0, 0.42), 0.62)
    panel_font_scale = min(max(target_panel_font / downscale, 0.42), 1.1)

    panel_line_height = int(max(20, round((panel_font_scale + 0.18) * 28)))
    panel_width = int(max(340, min(round(preview_width * 0.42), 560)))

    return AnnotationStyle(
        box_thickness=box_thickness,
        shadow_thickness=min(box_thickness + 2, 10),
        corner_length=corner_length,
        label_font_scale=label_font_scale,
        label_thickness=max(1, min(int(round(box_thickness * 0.45)), 3)),
        label_padding_x=max(8, int(round(label_font_scale * 14))),
        label_padding_y=max(6, int(round(label_font_scale * 9))),
        panel_font_scale=panel_font_scale,
        panel_thickness=max(1, min(int(round(panel_font_scale * 1.8)), 2)),
        panel_line_height=panel_line_height,
        panel_width=panel_width,
    )


def configure_preview_window(layout: PreviewLayout) -> None:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, layout.width, layout.height)
    cv2.moveWindow(WINDOW_NAME, layout.x, layout.y)


def run_inference(model: YOLO, frame, args: argparse.Namespace, selected_classes: list[int] | None, tracking_enabled: bool):
    if tracking_enabled:
        return model.track(
            frame,
            persist=True,
            tracker=args.tracker,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            classes=selected_classes,
            device=args.device,
            verbose=False,
        )

    return model.predict(
        frame,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        classes=selected_classes,
        device=args.device,
        verbose=False,
    )


def process_video(
    args: argparse.Namespace,
    source: Path,
    *,
    show_preview: bool,
    status_callback: Callable[[str], None] | None = None,
) -> ProcessingResult:
    def push_status(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    ensure_video(source)
    push_status(f"Loading YOLO model: {args.model}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    class_names = normalize_names(model.names)
    selected_classes = resolve_classes(args.classes, class_names)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    source_frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    preview_layout = build_preview_layout(width, height) if show_preview else None
    annotation_style = compute_annotation_style(width, height, preview_layout)

    processed_fps = fps / (args.skip_frames + 1)
    annotated_video_path = output_dir / f"{source.stem}_annotated.mp4"
    writer = build_writer(annotated_video_path, processed_fps, width, height)

    seen_tracks: set[tuple[int, int]] = set()
    unique_counts: Counter[str] = Counter()
    max_visible_counts: Counter[str] = Counter()
    summed_visible_counts: Counter[str] = Counter()
    frame_rows: list[dict] = []
    seen_class_names: set[str] = set()
    tracking_enabled = True
    tracking_note = (
        "unique_tracked_objects is an approximate video-level count based on tracker ids. "
        "If tracking is lost and the same object gets a new id, the total can be over-counted."
    )
    preview_configured = False

    source_frame_number = 0
    processed_frames = 0

    push_status(f"Processing video: {source.name}")
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            source_frame_number += 1
            if args.skip_frames and (source_frame_number - 1) % (args.skip_frames + 1) != 0:
                continue

            if args.max_frames and processed_frames >= args.max_frames:
                break

            try:
                results = run_inference(model, frame, args, selected_classes, tracking_enabled)
            except (ImportError, ModuleNotFoundError) as exc:
                tracking_enabled = False
                tracking_note = (
                    "Tracking dependencies were unavailable, so the run fell back to plain detection. "
                    "unique_tracked_objects may be empty for this output."
                )
                print(f"Tracking disabled after dependency error: {exc}")
                results = run_inference(model, frame, args, selected_classes, tracking_enabled)

            result = results[0]

            frame_counts = extract_counts(result, class_names, seen_tracks, unique_counts)
            for name, count in frame_counts.items():
                summed_visible_counts[name] += count
                max_visible_counts[name] = max(max_visible_counts[name], count)
                seen_class_names.add(name)

            timestamp_sec = (source_frame_number - 1) / fps if fps else 0.0
            annotated_frame = frame.copy()
            draw_minimal_detections(annotated_frame, result, class_names, annotation_style)
            draw_panel(
                annotated_frame,
                frame_counts=frame_counts,
                unique_counts=unique_counts,
                frame_number=source_frame_number,
                timestamp_sec=timestamp_sec,
                style=annotation_style,
            )

            writer.write(annotated_frame)
            frame_rows.append(
                {
                    "frame_number": source_frame_number,
                    "timestamp_sec": timestamp_sec,
                    "counts": dict(frame_counts),
                }
            )
            processed_frames += 1

            if show_preview:
                if not preview_configured:
                    assert preview_layout is not None
                    configure_preview_window(preview_layout)
                    preview_configured = True
                cv2.imshow(WINDOW_NAME, annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    ordered_class_names = sorted(seen_class_names)
    csv_output_path = output_dir / f"{source.stem}_frame_counts.csv"
    write_frame_counts_csv(csv_output_path, frame_rows, ordered_class_names)

    summary = {
        "source_video": str(source),
        "model": args.model,
        "source_frames": source_frame_total,
        "processed_frames": processed_frames,
        "source_fps": fps,
        "output_fps": processed_fps,
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "classes_filter": selected_classes,
        "max_visible_per_frame": dict(sorted_counts(max_visible_counts)),
        "sum_visible_over_processed_frames": dict(sorted_counts(summed_visible_counts)),
        "unique_tracked_objects": dict(sorted_counts(unique_counts)),
        "tracking_enabled": tracking_enabled,
        "note": tracking_note,
        "annotated_video": str(annotated_video_path),
        "frame_counts_csv": str(csv_output_path),
    }
    summary["plain_english_summary"] = build_simple_summary_text(summary)

    summary_output_path = output_dir / f"{source.stem}_summary.json"
    summary_output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    push_status(f"Finished: {source.name}")

    return ProcessingResult(
        source=source,
        summary=summary,
        summary_path=summary_output_path,
        csv_path=csv_output_path,
        annotated_video_path=annotated_video_path,
    )


def list_ollama_models() -> list[str]:
    try:
        completed = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []

    models: list[str] = []
    for line in completed.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            models.append(parts[0])
    return models


def choose_ollama_model(requested_model: str | None) -> str:
    models = list_ollama_models()
    if requested_model:
        return requested_model
    if not models:
        raise RuntimeError("No local Ollama models were found. Run `ollama list` and pull a model first.")

    if DEFAULT_OLLAMA_MODEL in models:
        return DEFAULT_OLLAMA_MODEL
    return models[0]


def sample_frame_rows(csv_path: Path, limit: int = 8) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []

    samples: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            visible_total = sum(
                int(value)
                for key, value in row.items()
                if key not in {"frame_number", "timestamp_sec"} and value and value.isdigit()
            )
            if visible_total <= 0:
                continue

            condensed = {
                "frame_number": row["frame_number"],
                "timestamp_sec": row["timestamp_sec"],
            }
            for key, value in row.items():
                if key in condensed or not value or value == "0":
                    continue
                condensed[key] = value
            samples.append(condensed)

            if len(samples) >= limit:
                break
    return samples


def build_analysis_prompt(result: ProcessingResult) -> str:
    timeline_samples = sample_frame_rows(result.csv_path)
    payload = {
        "summary": result.summary,
        "sampled_frames": timeline_samples,
    }
    return (
        "You are analyzing object-detection annotations produced from a video.\n"
        "Use only the structured data below. Do not invent visual details that are not supported.\n"
        "Write the answer in very simple English for everyday readers.\n"
        "Keep it short, clear, and easy to understand.\n"
        "Avoid technical words and avoid long explanations.\n"
        "Use this format:\n"
        "1. One short paragraph called 'Simple Summary'.\n"
        "2. A short bullet list called 'What was seen'. Use plain object names and simple counts.\n"
        "3. One short line called 'What to keep in mind'.\n"
        "If the data is limited, say that clearly in simple words.\n"
        "Do not mention JSON, tracking ids, frame samples, or model names unless needed.\n\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}"
    )


def clean_ollama_output(text: str) -> str:
    ansi_pattern = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    cleaned = ansi_pattern.sub("", text)
    cleaned = re.sub(r"[\u2800-\u28ff]+", "", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def run_ollama_analysis(
    result: ProcessingResult,
    requested_model: str | None,
    *,
    status_callback: Callable[[str], None] | None = None,
) -> tuple[str, str, Path]:
    def push_status(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    model_name = choose_ollama_model(requested_model)
    prompt = build_analysis_prompt(result)
    push_status(f"Analyzing with Ollama model: {model_name}")
    env = os.environ.copy()
    env.setdefault("OLLAMA_NOHISTORY", "1")
    env.setdefault("TERM", "dumb")

    completed = subprocess.run(
        ["ollama", "run", model_name, prompt, "--hidethinking"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if completed.returncode != 0:
        error_text = clean_ollama_output(completed.stderr) or clean_ollama_output(completed.stdout) or "Unknown Ollama error."
        raise RuntimeError(error_text)

    analysis_text = clean_ollama_output(completed.stdout)
    analysis_path = result.summary_path.with_name(f"{result.source.stem}_analysis.txt")
    analysis_path.write_text(analysis_text, encoding="utf-8")
    push_status(f"Saved analysis: {analysis_path.name}")
    return analysis_text, model_name, analysis_path


def center_tk_window(window: tk.Tk | tk.Toplevel, width: int, height: int) -> None:
    window.update_idletasks()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = max((screen_width - width) // 2, 0)
    y = max((screen_height - height) // 2, 0)
    window.geometry(f"{width}x{height}+{x}+{y}")


class VideoAnalyzerGUI:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = tk.Tk()
        self.root.title("Video Vision")
        self.root.configure(bg="#121212")
        self.root.minsize(760, 460)
        center_tk_window(self.root, 860, 540)

        self.selected_video: Path | None = Path(args.source) if args.source else None
        self.current_result: ProcessingResult | None = None
        self.processing = False
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.ollama_model = self._resolve_initial_ollama_model()

        self.status_var = tk.StringVar(value="Pick a video to start live detection.")
        self.video_var = tk.StringVar(
            value=self.selected_video.name if self.selected_video else "No video selected"
        )
        self.model_var = tk.StringVar(value=f"Ollama: {self.ollama_model}")

        self._build_ui()
        self.root.after(150, self._poll_events)

        if self.selected_video is not None:
            self._start_processing(self.selected_video)

    def _resolve_initial_ollama_model(self) -> str:
        try:
            return choose_ollama_model(self.args.ollama_model)
        except RuntimeError:
            return self.args.ollama_model or "not available"

    def _build_ui(self) -> None:
        container = tk.Frame(self.root, bg="#121212")
        container.pack(fill="both", expand=True, padx=28, pady=26)

        title = tk.Label(
            container,
            text="YOLO Live Video Counter",
            bg="#121212",
            fg="#F5F5F5",
            font=("Segoe UI Semibold", 22),
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            container,
            text="Pick a video, watch the live minimal annotations, then get a simple English summary from your local Ollama model.",
            bg="#121212",
            fg="#A8A8A8",
            font=("Segoe UI", 10),
            wraplength=760,
            justify="left",
        )
        subtitle.pack(anchor="w", pady=(8, 18))

        info_card = tk.Frame(container, bg="#1B1B1B", bd=0, highlightthickness=0)
        info_card.pack(fill="x")

        video_label = tk.Label(
            info_card,
            textvariable=self.video_var,
            bg="#1B1B1B",
            fg="#EDEDED",
            font=("Segoe UI", 12),
            anchor="w",
            padx=18,
            pady=14,
        )
        video_label.pack(fill="x")

        model_label = tk.Label(
            info_card,
            textvariable=self.model_var,
            bg="#1B1B1B",
            fg="#9ED8FF",
            font=("Segoe UI", 10),
            anchor="w",
            padx=18,
            pady=0,
        )
        model_label.pack(fill="x", pady=(0, 14))

        buttons_row = tk.Frame(container, bg="#121212")
        buttons_row.pack(fill="x", pady=(18, 16))

        self.pick_button = tk.Button(
            buttons_row,
            text="Pick Video",
            command=self.pick_video,
            bg="#F5F5F5",
            fg="#121212",
            activebackground="#FFFFFF",
            activeforeground="#121212",
            relief="flat",
            bd=0,
            padx=26,
            pady=12,
            font=("Segoe UI Semibold", 11),
            cursor="hand2",
        )
        self.pick_button.pack(side="left")

        self.analyze_button = tk.Button(
            buttons_row,
            text="Analyze",
            command=self.analyze_latest,
            bg="#2A2A2A",
            fg="#F5F5F5",
            activebackground="#333333",
            activeforeground="#FFFFFF",
            relief="flat",
            bd=0,
            padx=26,
            pady=12,
            font=("Segoe UI Semibold", 11),
            cursor="hand2",
            state="disabled",
        )
        self.analyze_button.pack(side="left", padx=(12, 0))

        status_label = tk.Label(
            container,
            textvariable=self.status_var,
            bg="#121212",
            fg="#C7C7C7",
            font=("Segoe UI", 10),
            anchor="w",
        )
        status_label.pack(fill="x", pady=(0, 10))

        self.output_box = scrolledtext.ScrolledText(
            container,
            height=14,
            bg="#181818",
            fg="#ECECEC",
            insertbackground="#ECECEC",
            relief="flat",
            bd=0,
            wrap="word",
            font=("Consolas", 10),
            padx=16,
            pady=16,
        )
        self.output_box.pack(fill="both", expand=True)
        self.output_box.insert(
            "1.0",
            "The live preview will open in a separate window after you pick a video.\n"
            "When processing finishes, Analyze will write a simple English summary with Ollama.\n",
        )
        self.output_box.configure(state="disabled")

    def _append_output(self, text: str) -> None:
        self.output_box.configure(state="normal")
        self.output_box.insert("end", f"{text}\n\n")
        self.output_box.see("end")
        self.output_box.configure(state="disabled")

    def _set_busy(self, busy: bool) -> None:
        self.processing = busy
        self.pick_button.configure(state="disabled" if busy else "normal")
        if busy or self.current_result is None:
            self.analyze_button.configure(state="disabled")
        else:
            self.analyze_button.configure(state="normal")

    def pick_video(self) -> None:
        if self.processing:
            return

        file_path = filedialog.askopenfilename(
            title="Pick video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v *.webm"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        source = Path(file_path)
        self._start_processing(source)

    def _start_processing(self, source: Path) -> None:
        self.selected_video = source
        self.current_result = None
        self.video_var.set(source.name)
        self.status_var.set(f"Starting live detection for {source.name}")
        self._append_output(f"Selected video: {source}")
        self._set_busy(True)

        worker = threading.Thread(target=self._run_processing_worker, args=(source,), daemon=True)
        worker.start()

    def _run_processing_worker(self, source: Path) -> None:
        try:
            result = process_video(
                self.args,
                source,
                show_preview=True,
                status_callback=lambda message: self.events.put(("status", message)),
            )
            self.events.put(("processed", result))
        except Exception:
            self.events.put(("error", traceback.format_exc()))

    def analyze_latest(self) -> None:
        if self.processing or self.current_result is None:
            return

        self._set_busy(True)
        self.status_var.set("Running Ollama analysis...")
        worker = threading.Thread(target=self._run_analyze_worker, daemon=True)
        worker.start()

    def _run_analyze_worker(self) -> None:
        try:
            assert self.current_result is not None
            analysis_text, model_name, analysis_path = run_ollama_analysis(
                self.current_result,
                self.args.ollama_model,
                status_callback=lambda message: self.events.put(("status", message)),
            )
            self.events.put(("analysis", (analysis_text, model_name, analysis_path)))
        except Exception:
            self.events.put(("error", traceback.format_exc()))

    def _poll_events(self) -> None:
        try:
            while True:
                event_type, payload = self.events.get_nowait()
                if event_type == "status":
                    self.status_var.set(str(payload))
                elif event_type == "processed":
                    self.current_result = payload  # type: ignore[assignment]
                    self.status_var.set("Processing finished. You can analyze the computed annotations now.")
                    self._append_output(self.current_result.summary["plain_english_summary"])
                    self._set_busy(False)
                elif event_type == "analysis":
                    analysis_text, model_name, analysis_path = payload  # type: ignore[misc]
                    self.model_var.set(f"Ollama: {model_name}")
                    self.status_var.set(f"Analysis saved to {analysis_path.name}")
                    self._append_output(analysis_text)
                    self._set_busy(False)
                elif event_type == "error":
                    self.status_var.set("An error occurred. Check the details below.")
                    self._append_output(str(payload))
                    self._set_busy(False)
        except queue.Empty:
            pass
        finally:
            self.root.after(150, self._poll_events)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    args = parse_args()
    if args.gui or not args.source:
        VideoAnalyzerGUI(args).run()
        return

    result = process_video(args, Path(args.source), show_preview=not args.no_show)
    print(json.dumps(result.summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
