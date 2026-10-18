# FORMUP

FORMUP is a TV-first fitness coaching prototype for the Amazon Developer Hackathon Fire TV track. It helps a user choose a short squat session, set up a camera, follow live pose guidance, and finish with a simple locally displayed progress summary.

## What is included

- A remote-friendly welcome, workout selection, camera setup, live session, completion, and progress flow.
- Live webcam capture with MediaPipe Pose Landmarker when camera permission and network model assets are available.
- Knee-angle smoothing and a guarded squat state machine that counts a rep only after descent, depth, ascent, and standing thresholds are met.
- A clearly labelled deterministic demo mode for presentations without a camera or external model download.
- High-contrast, large-type visual design intended for a television viewing distance.

## Technology

- React 18 and Vite
- MediaPipe Tasks Vision (`@mediapipe/tasks-vision`)
- Browser MediaDevices API and Canvas 2D overlay
- No AWS services or credentials are currently configured. The application does not claim cloud storage, Bedrock analysis, or an API backend.

## Run locally

```bash
npm install
npm run dev
```

Open the local Vite URL in a modern browser. Camera mode requires HTTPS or localhost and camera permission. The model is downloaded from the public MediaPipe CDN at runtime. If that is unavailable, choose **Preview demo mode** from camera setup.

## Build and test

```bash
npm run build
```

There is no automated test suite in this prototype yet. The production build is the current repeatable validation command. Use the demo script in [DEMO.md](./DEMO.md) for a deterministic walkthrough.

## Known limitations

- This is a web prototype, not a packaged Fire TV application. Fire TV integration, remote key mapping, and device certification remain submission work.
- Live pose tracking depends on browser camera permission and downloading the MediaPipe WASM/model assets.
- Progress is presented locally for the current session; there is no account, database, or cloud sync.
- The demo mode is simulated sample data. It is never presented as computer vision or camera tracking.

## Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the implementation boundaries and runtime behavior.

