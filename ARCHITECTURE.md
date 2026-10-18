# FORMUP architecture

## Runtime

`src/main.jsx` mounts the React application and `src/App.jsx` owns the product flow. The UI is a small state machine with these screens:

`welcome -> select -> setup -> workout -> complete -> progress`

The existing pose implementation remains isolated inside the workout screen. When live mode is selected, it:

1. Requests a video stream with `getUserMedia`.
2. Loads MediaPipe WASM and the pose landmarker from their public runtime assets.
3. Processes video frames and draws reliable body landmarks on a canvas.
4. Smooths short landmark dropouts.
5. Calculates left and right knee angles.
6. Updates a four-state squat tracker (`STANDING`, `DESCENDING`, `BOTTOM`, `ASCENDING`).

Camera errors are visible and actionable. The user can continue with demo mode without pretending that tracking occurred.

## Data boundaries

There is no server, authentication, AWS account configuration, API route, persistent database, or secret in this repository. Rep counts and progress shown by the prototype live in React state and reset when the page/session is reset.

## Styling

`src/styles.css` provides a charcoal/lime visual system with large type, high contrast, visible focus outlines, generous spacing, and responsive fallbacks for smaller browser windows. The layout is designed to be readable from a distance and uses native buttons so it remains keyboard and remote-control friendly.

