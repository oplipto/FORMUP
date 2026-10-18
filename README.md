# FORMUP

**Move with confidence.**

FORMUP is a TV-first fitness coaching prototype built for the **Amazon Developer Hackathon — Fire TV track**. It demonstrates a guided squat workout experience with camera-based pose tracking, rep counting, and a simple session summary.

**Live Demo:** https://formup-tau.vercel.app/  
**Source Code:** https://github.com/oplipto/FORMUP

## Overview

FORMUP explores how a television can become an interactive fitness companion, combining a large-screen workout interface with browser-based pose tracking.

The prototype guides users through a short squat session, provides visual feedback when camera tracking is available, and offers a deterministic preview mode for demonstrations.

## Features

- **TV-oriented interface:** High-contrast visuals, large typography, and a workout flow designed for viewing at a distance.
- **Guided workout flow:** Welcome screen, workout selection, camera setup, live session, completion, and progress summary.
- **Camera-based pose tracking:** Uses MediaPipe Pose Landmarker when camera permission and the required model assets are available.
- **Squat rep detection:** Smooths knee-angle measurements and uses a state machine with descent, depth, ascent, and standing thresholds to count repetitions.
- **Preview demo mode:** Provides clearly labelled simulated workout data when camera tracking is unavailable.
- **Session summary:** Displays locally available session progress without requiring an account.

## Technology Stack

| Technology | Purpose |
|---|---|
| React 18 | User interface |
| Vite | Development server and production build |
| MediaPipe Tasks Vision | Pose landmark detection |
| Browser MediaDevices API | Webcam access |
| Canvas 2D | Pose overlay rendering |

## Try the Live Demo

Open the deployed prototype:

**https://formup-tau.vercel.app/**

For a deterministic walkthrough:

1. Start a workout.
2. Select the lower-body workout.
3. Continue to camera setup.
4. Choose **Preview demo mode**.
5. Review the session counter and timer.
6. Finish the session and open the progress summary.

Preview mode uses simulated data. It does not represent actual camera tracking.

If you want to test live tracking, allow camera access and ensure the browser can download the required MediaPipe assets.

## Run Locally

Requirements: Node.js and npm.

Clone the repository and install dependencies:

```bash
git clone https://github.com/oplipto/FORMUP.git
cd FORMUP
npm install
npm run dev
```

Open the local URL printed by Vite.

Camera access requires a secure context, such as `localhost` or HTTPS, and browser camera permission. MediaPipe model and WASM assets are downloaded from the public CDN at runtime.

If those assets cannot be loaded, use Preview demo mode.

## Build and Validation

Run the production build:

```bash
npm run build
```

This is the current repeatable build-validation command. The prototype does not yet include an automated test suite.

See [DEMO.md](./DEMO.md) for the demonstration walkthrough and [ARCHITECTURE.md](./ARCHITECTURE.md) for implementation details.

## Architecture and Current Scope

FORMUP is currently a **web prototype**, not a packaged Fire TV or Vega OS application.

The current implementation does not include:

- Fire TV/Vega OS packaging, device integration, or remote-control key mapping.
- AWS backend services, Amazon Bedrock analysis, or cloud data storage.
- User accounts, a database, or cross-session cloud synchronization.
- Device certification or verified compatibility with physical Fire TV hardware.

Live pose tracking depends on camera access and the availability of the required MediaPipe assets. Preview mode is simulated and is kept distinct from real camera tracking.

These boundaries describe the current prototype rather than features that have already been implemented.

## Project Status

FORMUP is an early-stage proof of concept exploring TV-first fitness guidance and browser-based squat tracking. Further work would be needed to package and validate the experience for the target Fire TV/Vega OS platform.

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE).
