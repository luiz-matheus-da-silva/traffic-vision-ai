# TrafficVision AI

Browser-based vehicle detection for traffic videos. It runs **YOLOv5n** with **TensorFlow.js** entirely on the client: the video never leaves your machine and there is no inference server.

Select a video, press play, and the app draws bounding boxes over cars, trucks, buses and motorcycles while showing live counts, recent detections and inference latency.

## Features

- Local video analysis (any format your browser can play)
- YOLOv5n inference on the GPU through the TensorFlow.js WebGL backend
- Inference in a Web Worker, so the UI and the video stay responsive
- Tiled inference: overlapping, magnified crops help detect small and distant vehicles, with automatic fallback to full-frame analysis when the device is slow
- Live settings: confidence threshold, target FPS and vehicle classes
- Live counters for the current frame, a list of recent detections and a latency indicator

## Requirements

- Node.js 18 or newer
- A recent Chromium-based browser, Firefox or Safari with WebGL and `OffscreenCanvas` support

## Getting started

```bash
npm install
npm run dev
```

Open <http://localhost:5173>, select a video file and press play.

Other scripts:

| Command           | Description                      |
| ----------------- | -------------------------------- |
| `npm run build`   | Production build into `dist/`    |
| `npm run preview` | Serve the production build       |
| `npm test`        | Run the unit tests (Vitest)      |

## How it works

```
<video> ──► capture frame (ImageBitmap) ──► Web Worker
                                              │
                    ┌─────────────────────────┘
                    ▼
        split into regions (2 overlapping crops)
                    ▼
        letterbox to 640×640 → YOLOv5n (TF.js, WebGL)
                    ▼
        decode boxes → map back to the original frame
                    ▼
        merge duplicates (class-aware NMS)
                    ▼
Main thread ◄── detections ── filter by confidence/class
     │
     ├─ draw boxes on the canvas over the video
     └─ update counters and the recent detections list
```

1. **Capture** – `main.js` samples the playing video at the target FPS and sends one frame at a time to the worker (it never queues frames, so latency does not pile up).
2. **Pre-processing** – each region is resized with *letterbox* (keeps the aspect ratio, pads with gray) to the 640×640 input of the model.
3. **Inference** – YOLOv5n returns up to 100 boxes after a built-in NMS step. Coordinates are normalized to the model input.
4. **Post-processing** – boxes are converted back to pixels of the original frame (undoing padding, scale and crop offset), then results from all regions are merged.
5. **Rendering** – the main thread filters by confidence and enabled classes, draws the boxes and updates the UI.

## Project structure

```
index.html                     UI (Bootstrap 5 + Bootstrap Icons)
public/models/yolov5n_web_model/   TF.js graph model, weights and labels
src/js/
  main.js                      UI wiring, frame capture loop, rendering
  config/ModelConfig.js        Model input size and confidence floor
  services/VideoService.js     Loads the video file and captures frames
  workers/detection.worker.js  Runs the whole detection pipeline off the main thread
  core/inference/              Model loading, pre-processing, inference
  core/detection/              Post-processing (decode + merge) and canvas drawing
```

## Notes on the model

- Classes come from the COCO dataset. The UI only shows `car`, `truck`, `bus` and `motorcycle`.
- The exported model has an **agnostic NMS** step baked in. Its IoU threshold was raised from 0.45 to 0.7 (patched in `group1-shard2of2.bin`) so a rider and their motorcycle are not merged into a single detection.
- The Worker keeps a low confidence floor (0.25); the slider in the UI filters above it.

## Limitations

- Counters show what is in the **current frame**. There is no object tracking yet, so cumulative counts are not available.
- Detection quality depends on video resolution and camera angle; very small objects can still be missed.

## Tech stack

TensorFlow.js · YOLOv5n · Web Workers · Vite · Bootstrap 5
