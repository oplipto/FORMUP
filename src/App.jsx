import React, { useCallback, useEffect, useRef, useState } from 'react';
import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';

const WASM_PATH =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm';
const MODEL_PATH =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

const STATUS = {
  CAMERA_STARTING: 'Camera starting',
  CAMERA_READY: 'Camera ready',
  MODEL_LOADING: 'Loading pose model',
  ACTIVE: 'Pose detection active',
  PARTIAL: 'Full-body pose not visible',
  ERROR: 'Error',
};

const VISIBILITY_THRESHOLD = 0.5;
const SMOOTHING_FACTOR = 0.65;
const MAX_MISSED_FRAMES = 3;
const SQUAT_ANGLE_THRESHOLDS = {
  descendingEntry: 160,
  bottomEntry: 145,
  bottomExit: 155,
  standingExit: 155,
};
const TRANSITION_STABILITY_FRAMES = 2;
const MAX_LOST_TRACKING_FRAMES = 10;
const SQUAT_STATES = {
  STANDING: 'STANDING',
  DESCENDING: 'DESCENDING',
  BOTTOM: 'BOTTOM',
  ASCENDING: 'ASCENDING',
};
const BODY_LANDMARK_INDICES = new Set([
  11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
]);
const FULL_BODY_LANDMARK_INDICES = [11, 12, 23, 24, 25, 26, 27, 28];

function isReliable(point) {
  return (point.visibility ?? point.presence ?? 0) >= VISIBILITY_THRESHOLD;
}

function calculateKneeAngle(hip, knee, ankle) {
  const hipVector = { x: hip.x - knee.x, y: hip.y - knee.y };
  const ankleVector = { x: ankle.x - knee.x, y: ankle.y - knee.y };
  const hipLength = Math.hypot(hipVector.x, hipVector.y);
  const ankleLength = Math.hypot(ankleVector.x, ankleVector.y);

  if (hipLength === 0 || ankleLength === 0) {
    return null;
  }

  const cosine =
    (hipVector.x * ankleVector.x + hipVector.y * ankleVector.y) /
    (hipLength * ankleLength);
  return (Math.acos(Math.max(-1, Math.min(1, cosine))) * 180) / Math.PI;
}

function getKneeAngle(landmarks, hipIndex, kneeIndex, ankleIndex) {
  const points = [hipIndex, kneeIndex, ankleIndex].map(
    (index) => landmarks[index],
  );
  if (points.some((point) => !point || !isReliable(point))) {
    return null;
  }

  return calculateKneeAngle(points[0], points[1], points[2]);
}

function getKneeAngles(landmarks) {
  const leftAngle = getKneeAngle(landmarks, 23, 25, 27);
  const rightAngle = getKneeAngle(landmarks, 24, 26, 28);
  const leftConfidence = [23, 25, 27].map((index) =>
    landmarks[index] ? landmarks[index].visibility ?? landmarks[index].presence ?? 0 : 0,
  );
  const rightConfidence = [24, 26, 28].map((index) =>
    landmarks[index] ? landmarks[index].visibility ?? landmarks[index].presence ?? 0 : 0,
  );

  return {
    leftAngle,
    rightAngle,
    leftConfidence: Math.min(...leftConfidence),
    rightConfidence: Math.min(...rightConfidence),
    average:
      leftAngle === null && rightAngle === null
        ? null
        : [leftAngle, rightAngle]
            .filter((angle) => angle !== null)
            .reduce((sum, angle, _index, angles) => sum + angle / angles.length, 0),
  };
}

function createSquatTracker() {
  return {
    state: SQUAT_STATES.STANDING,
    candidateState: null,
    candidateFrames: 0,
    cycleStarted: false,
    reachedBottom: false,
    lostTrackingFrames: 0,
    repCount: 0,
    reason: 'Waiting for a stable standing pose',
  };
}

function updateSquatTracker(tracker, kneeAngle) {
  if (kneeAngle === null) {
    tracker.lostTrackingFrames += 1;
    tracker.candidateState = null;
    tracker.candidateFrames = 0;
    if (tracker.lostTrackingFrames >= MAX_LOST_TRACKING_FRAMES) {
      tracker.state = SQUAT_STATES.STANDING;
      tracker.cycleStarted = false;
      tracker.reachedBottom = false;
      tracker.reason = 'Tracking lost; squat cycle reset safely';
    } else {
      tracker.reason = 'Waiting for reliable hip, knee, and ankle landmarks';
    }
    return tracker;
  }

  tracker.lostTrackingFrames = 0;
  let targetState = null;
  let reason = 'Holding current squat state';

  switch (tracker.state) {
    case SQUAT_STATES.STANDING:
      if (kneeAngle <= SQUAT_ANGLE_THRESHOLDS.descendingEntry) {
        targetState = SQUAT_STATES.DESCENDING;
        reason = 'Descent threshold reached';
      } else {
        reason = 'Standing; waiting for a new descent';
      }
      break;
    case SQUAT_STATES.DESCENDING:
      if (kneeAngle <= SQUAT_ANGLE_THRESHOLDS.bottomEntry) {
        targetState = SQUAT_STATES.BOTTOM;
        reason = 'Minimum squat depth reached';
      } else if (kneeAngle >= SQUAT_ANGLE_THRESHOLDS.standingExit) {
        tracker.state = SQUAT_STATES.STANDING;
        tracker.cycleStarted = false;
        tracker.reachedBottom = false;
        reason = 'Descent aborted before minimum depth';
      } else {
        reason = 'Descending; continue toward minimum depth';
      }
      break;
    case SQUAT_STATES.BOTTOM:
      if (kneeAngle >= SQUAT_ANGLE_THRESHOLDS.bottomExit) {
        targetState = SQUAT_STATES.ASCENDING;
        reason = 'Bottom exit threshold reached';
      } else {
        reason = 'Bottom confirmed; waiting for ascent';
      }
      break;
    case SQUAT_STATES.ASCENDING:
      if (kneeAngle <= SQUAT_ANGLE_THRESHOLDS.bottomEntry) {
        tracker.state = SQUAT_STATES.BOTTOM;
        reason = 'Returned toward bottom; ascent not complete';
      } else if (kneeAngle >= SQUAT_ANGLE_THRESHOLDS.standingExit) {
        targetState = SQUAT_STATES.STANDING;
        reason = 'Standing threshold reached after valid bottom';
      } else {
        reason = 'Ascending; continue toward standing';
      }
      break;
    default:
      tracker.state = SQUAT_STATES.STANDING;
      tracker.cycleStarted = false;
      tracker.reachedBottom = false;
      reason = 'Invalid state; reset safely';
  }

  if (targetState === null) {
    tracker.candidateState = null;
    tracker.candidateFrames = 0;
    tracker.reason = reason;
    return tracker;
  }

  if (tracker.candidateState === targetState) {
    tracker.candidateFrames += 1;
  } else {
    tracker.candidateState = targetState;
    tracker.candidateFrames = 1;
  }

  if (tracker.candidateFrames < TRANSITION_STABILITY_FRAMES) {
    tracker.reason = `${reason}; stabilizing ${tracker.candidateFrames}/${TRANSITION_STABILITY_FRAMES}`;
    return tracker;
  }

  const previousState = tracker.state;
  tracker.state = targetState;
  tracker.candidateState = null;
  tracker.candidateFrames = 0;

  if (targetState === SQUAT_STATES.BOTTOM) {
    tracker.reachedBottom = true;
  }
  if (
    previousState === SQUAT_STATES.STANDING &&
    targetState === SQUAT_STATES.DESCENDING
  ) {
    tracker.cycleStarted = true;
  }

  if (
    previousState === SQUAT_STATES.ASCENDING &&
    targetState === SQUAT_STATES.STANDING &&
    tracker.cycleStarted &&
    tracker.reachedBottom
  ) {
    tracker.repCount += 1;
    tracker.cycleStarted = false;
    tracker.reachedBottom = false;
    tracker.reason = `Rep counted: ${tracker.repCount}; cycle locked until new descent`;
  } else {
    tracker.reason = `${previousState} -> ${targetState} accepted`;
  }

  return tracker;
}

function stabilizeLandmarks(landmarks, state) {
  return landmarks.map((point, index) => {
    if (!BODY_LANDMARK_INDICES.has(index) || !point) {
      return point;
    }

    const previous = state.get(index);
    if (isReliable(point)) {
      const smoothedPoint = previous
        ? {
            ...point,
            x:
              previous.point.x +
              (point.x - previous.point.x) * SMOOTHING_FACTOR,
            y:
              previous.point.y +
              (point.y - previous.point.y) * SMOOTHING_FACTOR,
            z:
              (previous.point.z ?? 0) +
              ((point.z ?? 0) - (previous.point.z ?? 0)) * SMOOTHING_FACTOR,
          }
        : point;
      state.set(index, { point: smoothedPoint, missedFrames: 0 });
      return smoothedPoint;
    }

    if (previous && previous.missedFrames < MAX_MISSED_FRAMES) {
      const retainedPoint = {
        ...previous.point,
        visibility: Math.max(
          VISIBILITY_THRESHOLD,
          previous.point.visibility ?? 0,
        ),
      };
      state.set(index, {
        point: retainedPoint,
        missedFrames: previous.missedFrames + 1,
      });
      return retainedPoint;
    }

    state.delete(index);
    return point;
  });
}

function drawPose(canvas, video, landmarks) {
  const context = canvas.getContext('2d');
  if (!context) {
    return { visibleCount: 0, fullBody: false };
  }

  const width = video.videoWidth;
  const height = video.videoHeight;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  context.clearRect(0, 0, width, height);

  const reliableLandmarks = landmarks.map(isReliable);
  const fullBody = FULL_BODY_LANDMARK_INDICES.every(
    (index) => reliableLandmarks[index],
  );

  context.lineCap = 'round';
  context.lineJoin = 'round';
  context.strokeStyle = '#071b2d';
  context.lineWidth = 10;
  for (const connection of PoseLandmarker.POSE_CONNECTIONS) {
    if (
      BODY_LANDMARK_INDICES.has(connection.start) &&
      BODY_LANDMARK_INDICES.has(connection.end) &&
      reliableLandmarks[connection.start] &&
      reliableLandmarks[connection.end]
    ) {
      const start = landmarks[connection.start];
      const end = landmarks[connection.end];
      context.beginPath();
      context.moveTo(start.x * width, start.y * height);
      context.lineTo(end.x * width, end.y * height);
      context.stroke();
    }
  }

  context.strokeStyle = '#41e6a0';
  context.lineWidth = 5;
  for (const connection of PoseLandmarker.POSE_CONNECTIONS) {
    if (
      BODY_LANDMARK_INDICES.has(connection.start) &&
      BODY_LANDMARK_INDICES.has(connection.end) &&
      reliableLandmarks[connection.start] &&
      reliableLandmarks[connection.end]
    ) {
      const start = landmarks[connection.start];
      const end = landmarks[connection.end];
      context.beginPath();
      context.moveTo(start.x * width, start.y * height);
      context.lineTo(end.x * width, end.y * height);
      context.stroke();
    }
  }

  context.fillStyle = '#ffffff';
  context.strokeStyle = '#41e6a0';
  context.lineWidth = 3;
  let visibleCount = 0;
  for (const index of BODY_LANDMARK_INDICES) {
    const point = landmarks[index];
    if (!point || !reliableLandmarks[index]) {
      continue;
    }

    visibleCount += 1;
    context.beginPath();
    context.arc(point.x * width, point.y * height, 5, 0, Math.PI * 2);
    context.fill();
    context.stroke();
  }

  return { visibleCount, fullBody };
}

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const poseLandmarkerRef = useRef(null);
  const animationFrameRef = useRef(null);
  const videoFrameCallbackRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);
  const landmarkStateRef = useRef(new Map());
  const squatTrackerRef = useRef(createSquatTracker());
  const processedFrameCountRef = useRef(0);
  const lastDiagnosticTimeRef = useRef(0);
  const [status, setStatus] = useState(STATUS.CAMERA_STARTING);
  const [errorMessage, setErrorMessage] = useState('');
  const [landmarkCount, setLandmarkCount] = useState(0);
  const [repCount, setRepCount] = useState(0);
  const [squatState, setSquatState] = useState(SQUAT_STATES.STANDING);
  const [kneeAngle, setKneeAngle] = useState(null);
  const [kneeAngles, setKneeAngles] = useState({
    left: null,
    right: null,
    leftConfidence: 0,
    rightConfidence: 0,
  });
  const [squatDebug, setSquatDebug] = useState(
    'Waiting for a stable standing pose',
  );
  const [screen, setScreen] = useState('welcome');
  const [isDemo, setIsDemo] = useState(false);
  const [demoSeconds, setDemoSeconds] = useState(0);

  const stopCamera = useCallback(() => {
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (
      videoFrameCallbackRef.current !== null &&
      videoRef.current?.cancelVideoFrameCallback
    ) {
      videoRef.current.cancelVideoFrameCallback(videoFrameCallbackRef.current);
      videoFrameCallbackRef.current = null;
    }

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    poseLandmarkerRef.current?.close();
    poseLandmarkerRef.current = null;
    landmarkStateRef.current.clear();
    squatTrackerRef.current = createSquatTracker();
  }, []);

  const showError = useCallback((message) => {
    setStatus(STATUS.ERROR);
    setErrorMessage(message);
  }, []);

  useEffect(() => {
    let isMounted = true;

    if (screen !== 'workout' || isDemo) {
      return () => {
        isMounted = false;
        stopCamera();
      };
    }

    const processVideoFrame = () => {
      const video = videoRef.current;
      const poseLandmarker = poseLandmarkerRef.current;

      if (!isMounted || !video || !poseLandmarker) {
        return;
      }

      if (
        video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
        video.currentTime !== lastVideoTimeRef.current
      ) {
        try {
          const result = poseLandmarker.detectForVideo(video, performance.now());
          const rawLandmarks = result.landmarks[0] ?? [];
          const landmarks = stabilizeLandmarks(
            rawLandmarks,
            landmarkStateRef.current,
          );
          const pose = drawPose(canvasRef.current, video, landmarks);
          const angles = getKneeAngles(landmarks);
          const kneeAngle = angles.average;
          const previousRepCount = squatTrackerRef.current.repCount;
          updateSquatTracker(squatTrackerRef.current, kneeAngle);
          setRepCount(squatTrackerRef.current.repCount);
          setSquatState(squatTrackerRef.current.state);
          setKneeAngle(kneeAngle);
          setKneeAngles({
            left: angles.leftAngle,
            right: angles.rightAngle,
            leftConfidence: angles.leftConfidence,
            rightConfidence: angles.rightConfidence,
          });
          setSquatDebug(squatTrackerRef.current.reason);
          setLandmarkCount(pose.visibleCount);
          setStatus(pose.fullBody ? STATUS.ACTIVE : STATUS.PARTIAL);
          lastVideoTimeRef.current = video.currentTime;

          processedFrameCountRef.current += 1;
          const now = performance.now();
          if (now - lastDiagnosticTimeRef.current >= 5000) {
            const confidenceValues = landmarks
              .map((point, index) => ({
                index,
                confidence: point.visibility ?? point.presence ?? 0,
              }))
              .filter(({ index }) => BODY_LANDMARK_INDICES.has(index))
              .map(({ confidence }) => confidence);
            const averageConfidence = confidenceValues.length
              ? confidenceValues.reduce((sum, value) => sum + value, 0) /
                confidenceValues.length
              : 0;
            console.debug('FORMUP pose diagnostics', {
              processedFrames: processedFrameCountRef.current,
              currentTime: video.currentTime,
              videoDimensions: `${video.videoWidth}x${video.videoHeight}`,
              visibleLandmarks: pose.visibleCount,
              averageBodyConfidence: Number(averageConfidence.toFixed(2)),
              squatState: squatTrackerRef.current.state,
              leftKneeAngle:
                angles.leftAngle === null
                  ? null
                  : Number(angles.leftAngle.toFixed(1)),
              rightKneeAngle:
                angles.rightAngle === null
                  ? null
                  : Number(angles.rightAngle.toFixed(1)),
              leftKneeConfidence: Number(angles.leftConfidence.toFixed(2)),
              rightKneeConfidence: Number(angles.rightConfidence.toFixed(2)),
              squatReason: squatTrackerRef.current.reason,
            });
            lastDiagnosticTimeRef.current = now;
          }
          if (squatTrackerRef.current.repCount !== previousRepCount) {
            console.debug('FORMUP squat rep counted', {
              repCount: squatTrackerRef.current.repCount,
              kneeAngle:
                kneeAngle === null ? null : Number(kneeAngle.toFixed(1)),
              reason: squatTrackerRef.current.reason,
            });
          }
        } catch (error) {
          console.error('FORMUP pose detection failed for video frame', error);
        }
      }

      if (typeof video.requestVideoFrameCallback === 'function') {
        videoFrameCallbackRef.current = video.requestVideoFrameCallback(
          processVideoFrame,
        );
      } else {
        animationFrameRef.current = requestAnimationFrame(processVideoFrame);
      }
    };

    const start = async () => {
      let stage = 'camera initialization';

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { facingMode: 'user' },
        });

        if (!isMounted) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        streamRef.current = stream;
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        console.debug('FORMUP video ready', {
          readyState: videoRef.current.readyState,
          paused: videoRef.current.paused,
          videoDimensions: `${videoRef.current.videoWidth}x${videoRef.current.videoHeight}`,
        });
        setStatus(STATUS.CAMERA_READY);
        setStatus(STATUS.MODEL_LOADING);

        stage = 'MediaPipe WASM loading';
        const vision = await FilesetResolver.forVisionTasks(WASM_PATH);

        stage = 'pose model loading';
        let poseLandmarker;
        try {
          poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath: MODEL_PATH,
              delegate: 'GPU',
            },
            runningMode: 'VIDEO',
            numPoses: 1,
          });
        } catch (gpuError) {
          console.warn('FORMUP GPU pose delegate unavailable; using CPU', gpuError);
          poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath: MODEL_PATH,
              delegate: 'CPU',
            },
            runningMode: 'VIDEO',
            numPoses: 1,
          });
        }

        if (!isMounted) {
          poseLandmarker.close();
          return;
        }

        poseLandmarkerRef.current = poseLandmarker;
        processVideoFrame();
      } catch (error) {
        if (!isMounted) {
          return;
        }

        console.error(`FORMUP ${stage} failed`, error);

        const message =
          error?.name === 'NotAllowedError'
            ? 'Camera permission was denied. Allow camera access and reload the page.'
            : 'Unable to start the camera or load the pose model. Check your connection and try again.';
        showError(message);
      }
    };

    if (!navigator.mediaDevices?.getUserMedia) {
      showError('This browser does not support webcam access.');
    } else {
      start();
    }

    return () => {
      isMounted = false;
      stopCamera();
    };
  }, [isDemo, screen, showError, stopCamera]);

  useEffect(() => {
    if (!isDemo || screen !== 'workout') {
      return undefined;
    }

    setRepCount(0);
    setDemoSeconds(0);
    const timer = window.setInterval(() => {
      setDemoSeconds((seconds) => {
        const nextSeconds = seconds + 1;
        setRepCount(Math.min(12, Math.floor(nextSeconds / 2)));
        return nextSeconds;
      });
    }, 1000);
    return () => window.clearInterval(timer);
  }, [isDemo, screen]);

  const beginWorkout = (demo = false) => {
    setIsDemo(demo);
    setRepCount(0);
    setErrorMessage('');
    setScreen('workout');
  };

  const finishWorkout = () => {
    stopCamera();
    setScreen('complete');
  };

  const resetWorkout = () => {
    stopCamera();
    setIsDemo(false);
    setRepCount(0);
    setScreen('welcome');
  };

  const Header = () => (
    <header className="topbar">
      <button className="brand" onClick={resetWorkout} aria-label="Return to home">
        <span className="brand-mark">F</span>
        <span>FORMUP</span>
      </button>
      <div className="topbar-meta">
        <span className="session-label">TV FITNESS COACH</span>
        {screen === 'workout' && (
          <span className={`status status-${status === STATUS.ERROR ? 'error' : 'ok'}`}>
            <span className="status-dot" /> {isDemo ? 'Demo mode' : status}
          </span>
        )}
      </div>
    </header>
  );

  if (screen === 'welcome') {
    return (
      <main className="app"><Header />
        <section className="hero">
          <div className="hero-copy">
            <p className="eyebrow">YOUR FORM. YOUR MOMENT.</p>
            <h1>Move with<br /><span>confidence.</span></h1>
            <p className="lede">A focused, camera-guided workout built for the big screen. Start small, stay consistent, and let every rep count.</p>
            <button className="button button-primary" onClick={() => setScreen('select')}>Start a workout <span>→</span></button>
          </div>
          <div className="hero-art" aria-hidden="true"><div className="ring ring-large" /><div className="ring ring-small" /><span className="hero-number">01</span><span className="hero-caption">FORM<br />FIRST</span></div>
        </section>
        <section className="feature-row">
          <div><strong>01</strong><span>Choose a focused session</span></div>
          <div><strong>02</strong><span>Set your camera position</span></div>
          <div><strong>03</strong><span>Move at your pace</span></div>
        </section>
      </main>
    );
  }

  if (screen === 'select') {
    return (
      <main className="app"><Header />
        <section className="screen-heading"><p className="eyebrow">STEP 01 / CHOOSE SESSION</p><h1>What are we<br /><span>working on?</span></h1><p className="lede">Pick a simple movement to get started. More sessions will be added as your practice grows.</p></section>
        <div className="workout-grid">
          <button className="workout-card selected" onClick={() => setScreen('setup')}><span className="card-index">01</span><span className="exercise-icon">↓</span><h2>Lower body<br />foundation</h2><p>Squat form · 12 reps · 5 min</p><span className="card-arrow">→</span></button>
          <div className="workout-card locked"><span className="card-index">02</span><span className="lock">LOCKED</span><h2>Upper body<br />activation</h2><p>Coming soon</p></div>
        </div>
        <button className="text-button" onClick={resetWorkout}>← Back to home</button>
      </main>
    );
  }

  if (screen === 'setup') {
    return (
      <main className="app"><Header />
        <section className="setup-layout"><div className="setup-copy"><p className="eyebrow">STEP 02 / CAMERA SETUP</p><h1>Make space<br /><span>to move.</span></h1><p className="lede">Place your screen at eye level and stand far enough back for your full body to fit inside the frame.</p><div className="setup-list"><div><strong>01</strong><span>Keep the camera steady</span></div><div><strong>02</strong><span>Use a well-lit space</span></div><div><strong>03</strong><span>Keep head to ankles visible</span></div></div><button className="button button-primary" onClick={() => beginWorkout(false)}>Connect camera <span>→</span></button><button className="button button-secondary" onClick={() => beginWorkout(true)}>Preview demo mode</button></div><div className="setup-visual"><div className="camera-frame"><span className="frame-corner corner-tl" /><span className="frame-corner corner-tr" /><span className="frame-corner corner-bl" /><span className="frame-corner corner-br" /><div className="person-outline">●<br /><span>╲│╱<br />╱ ╲</span></div></div><span>FULL BODY IN FRAME</span></div></section>
        <button className="text-button" onClick={() => setScreen('select')}>← Choose another session</button>
      </main>
    );
  }

  if (screen === 'complete') {
    return (
      <main className="app"><Header />
        <section className="complete"><div className="complete-badge">✓</div><p className="eyebrow">SESSION COMPLETE</p><h1>That’s a<br /><span>strong finish.</span></h1><p className="lede">You showed up and moved with intention. That is how progress is built.</p><div className="result-card"><div><span>COMPLETED</span><strong>{repCount}</strong><small>reps</small></div><div><span>SESSION</span><strong>05:00</strong><small>lower body</small></div><div><span>MODE</span><strong>{isDemo ? 'DEMO' : 'LIVE'}</strong><small>{isDemo ? 'sample data' : 'camera tracked'}</small></div></div><button className="button button-primary" onClick={() => setScreen('progress')}>See your progress <span>→</span></button><button className="text-button" onClick={resetWorkout}>Return home</button></section>
      </main>
    );
  }

  if (screen === 'progress') {
    return (
      <main className="app"><Header />
        <section className="screen-heading"><p className="eyebrow">YOUR MOMENTUM</p><h1>Keep showing<br /><span>up.</span></h1><p className="lede">Your FORMUP practice is a collection of small wins. Today’s session is logged locally on this device.</p></section>
        <div className="progress-card"><div className="progress-stat"><strong>{repCount}</strong><span>reps completed today</span></div><div className="progress-bars"><span /><span /><span /><span className="current" /><span /></div><p>SESSION 01 <b>·</b> BUILDING A BASE</p></div>
        <button className="button button-primary" onClick={resetWorkout}>Finish <span>→</span></button>
      </main>
    );
  }

  return (
    <main className="app workout-app"><Header />
      <div className="workout-heading"><div><p className="eyebrow">{isDemo ? 'DEMO MODE / SAMPLE DATA' : 'LIVE SESSION / SQUATS'}</p><h1>Lower body <span>foundation</span></h1></div><button className="text-button" onClick={finishWorkout}>Finish session</button></div>
      {isDemo ? <div className="demo-stage"><div className="demo-person">●<span>╲│╱<br />╱ ╲</span></div><p>Sample movement preview</p><small>This is a deterministic demo. No camera or computer vision is active.</small></div> : <section className="camera-card" aria-label="Live camera and pose landmarks"><video ref={videoRef} className="camera" autoPlay muted playsInline /><canvas ref={canvasRef} className="landmarks" />{status === STATUS.CAMERA_STARTING && <p className="overlay-message">Requesting camera access...</p>}{status === STATUS.MODEL_LOADING && <p className="overlay-message">Loading pose model...</p>}{status === STATUS.PARTIAL && <p className="overlay-message partial-message">Step back so your whole body fits inside the camera frame.</p>}{status === STATUS.ERROR && <div className="overlay-message error-message"><strong>Camera unavailable</strong><span>{errorMessage}</span><button className="button button-secondary" onClick={() => beginWorkout(true)}>Continue in demo mode</button></div>}</section>}
      <section className="live-metrics"><div><span>REPS</span><strong>{repCount}<small>/ 12</small></strong></div><div><span>FORM STATUS</span><strong className="metric-state">{isDemo ? 'DEMO' : squatState}</strong></div><div><span>{isDemo ? 'TIME' : 'LANDMARKS'}</span><strong>{isDemo ? `0${Math.min(5, Math.floor(demoSeconds / 60))}:${String(demoSeconds % 60).padStart(2, '0')}` : landmarkCount}</strong></div></section>
      {!isDemo && <p className="hint">{status === STATUS.ERROR ? 'Camera access is optional. Choose demo mode to continue.' : squatDebug}</p>}
    </main>
  );
}

export default App;
