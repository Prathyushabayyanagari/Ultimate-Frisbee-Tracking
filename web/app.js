const cameraVideo = document.getElementById("cameraVideo");
const processedFrame = document.getElementById("processedFrame");
const videoPlaceholder = document.getElementById("videoPlaceholder");
const cameraSelect = document.getElementById("cameraSelect");
const sizeSelect = document.getElementById("sizeSelect");
const personConfidence = document.getElementById("personConfidence");
const deviceSelect = document.getElementById("deviceSelect");
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const statusTitle = document.getElementById("statusTitle");
const statusDescription = document.getElementById("statusDescription");
const statusIndicator = document.getElementById("statusIndicator");
const sidebarIndicator = document.getElementById("sidebarIndicator");
const sidebarStatus = document.getElementById("sidebarStatus");
const feedBadge = document.getElementById("feedBadge");
const headerLiveDot = document.getElementById("headerLiveDot");
const headerSessionText = document.getElementById("headerSessionText");
const frameTag = document.getElementById("frameTag");
const frameOverlay = document.querySelector(".frame-overlay");
const fpsValue = document.getElementById("fpsValue");
const playersValue = document.getElementById("playersValue");
const frisbeeValue = document.getElementById("frisbeeValue");
const whiteOvalValue = document.getElementById("whiteOvalValue");
const analysisRows = document.getElementById("analysisRows");
const analysisFrame = document.getElementById("analysisFrame");
const toast = document.getElementById("toast");

let cameraStream = null;
let sessionActive = false;
let sessionNumber = 0;
let activePersonThreshold = 0.4;
let previousFrameUrl = null;
let toastTimeout = null;

function setStatus(status, detail, isError = false) {
  statusTitle.textContent = status;
  statusDescription.textContent = detail;
  statusIndicator.classList.toggle("active", !isError && sessionActive);
  statusIndicator.classList.toggle("error", isError);
  sidebarIndicator.classList.toggle("active", !isError && sessionActive);
  sidebarIndicator.classList.toggle("error", isError);
  sidebarStatus.textContent = isError
    ? "SESSION ERROR"
    : sessionActive
      ? "TRACKING ACTIVE"
      : "SYSTEM IDLE";
  feedBadge.classList.toggle("active", sessionActive);
  feedBadge.innerHTML = `<span></span> ${sessionActive ? "LIVE" : "OFFLINE"}`;
  headerLiveDot.classList.toggle("active", sessionActive);
  headerSessionText.textContent = sessionActive ? "SESSION ACTIVE" : "READY TO CONNECT";
  frameTag.classList.toggle("visible", sessionActive);
  frameOverlay.classList.toggle("visible", sessionActive);
  startButton.disabled = sessionActive;
  stopButton.disabled = !sessionActive;
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("visible");
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => toast.classList.remove("visible"), 5500);
}

function renderAnalysis(analysis) {
  const detections = Array.isArray(analysis?.detections) ? analysis.detections : [];
  const width = Number(analysis?.frame_size?.width) || 0;
  const height = Number(analysis?.frame_size?.height) || 0;
  analysisFrame.textContent = width && height
    ? `FRAME ${width} × ${height}`
    : "FRAME ANALYSIS";
  analysisRows.replaceChildren();

  if (detections.length === 0) {
    const row = document.createElement("tr");
    row.className = "empty-row";
    const cell = document.createElement("td");
    cell.colSpan = 5;
    cell.textContent = "No player tracks or frisbee / sports-ball candidates in this frame.";
    row.append(cell);
    analysisRows.append(row);
    return;
  }

  detections.forEach((detection) => {
    const row = document.createElement("tr");
    const classCell = document.createElement("td");
    const classBadge = document.createElement("span");
    classBadge.className = `class-badge ${detection.class_name === "person" ? "person" : detection.class_name === "frisbee" ? "frisbee" : "candidate"}`;
    classBadge.textContent = detection.class_name || "unknown";
    classCell.append(classBadge);

    const trackCell = document.createElement("td");
    trackCell.textContent = detection.track_id == null
      ? "—"
      : `#${detection.track_id}`;

    const confidenceCell = document.createElement("td");
    confidenceCell.textContent = detection.confidence == null
      ? (detection.shape_score == null ? "—" : `shape ${Math.round(Number(detection.shape_score) * 100)}%`)
      : `${(Number(detection.confidence) * 100).toFixed(1)}%`;

    const boxCell = document.createElement("td");
    const box = Array.isArray(detection.bbox) ? detection.bbox : [];
    boxCell.textContent = box.map((coordinate) => Math.round(Number(coordinate))).join(", ");

    const statusCell = document.createElement("td");
    statusCell.textContent = detection.status || "Candidate";

    row.append(classCell, trackCell, confidenceCell, boxCell, statusCell);
    analysisRows.append(row);
  });
}

async function readError(response) {
  try {
    const payload = await response.json();
    return payload.error || `Request failed (${response.status}).`;
  } catch {
    return `Request failed (${response.status}).`;
  }
}

async function refreshCameras() {
  if (!navigator.mediaDevices?.enumerateDevices) {
    return;
  }
  const cameras = (await navigator.mediaDevices.enumerateDevices()).filter(
    (device) => device.kind === "videoinput",
  );
  const selected = cameraSelect.value;
  cameraSelect.replaceChildren(new Option("Default camera", ""));
  cameras.forEach((camera, index) => {
    cameraSelect.add(
      new Option(camera.label || `Camera ${index + 1}`, camera.deviceId),
    );
  });
  if (cameras.some((camera) => camera.deviceId === selected)) {
    cameraSelect.value = selected;
  }
}

async function startSession() {
  startButton.disabled = true;
  statusTitle.textContent = "Requesting camera access";
  statusDescription.textContent = "Allow camera access in your browser.";

  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error(
        "This browser cannot access the webcam. Open the site at http://127.0.0.1:8765.",
      );
    }

    const selectedDevice = cameraSelect.value;
    const videoConstraint = selectedDevice
      ? {
          deviceId: { exact: selectedDevice },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        }
      : {
          width: { ideal: 1280 },
          height: { ideal: 720 },
        };
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: videoConstraint,
      audio: false,
    });
    await refreshCameras();
    cameraVideo.srcObject = cameraStream;
    await cameraVideo.play();
    cameraVideo.classList.add("visible");
    videoPlaceholder.classList.add("hidden");

    const response = await fetch("/api/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        input_size: Number(sizeSelect.value),
        person_confidence: Number(personConfidence.value),
        device: deviceSelect.value,
      }),
    });
    if (!response.ok) {
      throw new Error(await readError(response));
    }

    sessionActive = true;
    sessionNumber += 1;
    activePersonThreshold = Number(personConfidence.value);
    setStatus(
      "Loading detector",
      `Player confidence ≥ ${activePersonThreshold.toFixed(2)}; confirm across 2 frames.`,
    );
    monitorModel(sessionNumber);
  } catch (error) {
    await stopCamera();
    sessionActive = false;
    setStatus("Could not start session", error.message, true);
    showToast(error.message);
  }
}

async function monitorModel(activeSession) {
  while (sessionActive && activeSession === sessionNumber) {
    try {
      const response = await fetch("/api/status", { cache: "no-store" });
      if (!response.ok) {
        throw new Error(await readError(response));
      }
      const state = await response.json();
      if (state.status === "running") {
        setStatus(
          "Tracking live",
          `Player confidence ≥ ${activePersonThreshold.toFixed(2)} · 2-frame confirmation.`,
        );
        processNextFrame(activeSession);
        return;
      }
      if (state.status === "error") {
        throw new Error(state.error || "Model initialization failed.");
      }
      statusTitle.textContent = "Loading detector";
      statusDescription.textContent =
        `Player confidence ≥ ${activePersonThreshold.toFixed(2)} · loading model.`;
      await new Promise((resolve) => setTimeout(resolve, 400));
    } catch (error) {
      sessionActive = false;
      await stopCamera();
      setStatus("Could not start session", error.message, true);
      showToast(error.message);
      return;
    }
  }
}

async function processNextFrame(activeSession) {
  if (!sessionActive || activeSession !== sessionNumber) {
    return;
  }
  if (cameraVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    setTimeout(() => processNextFrame(activeSession), 100);
    return;
  }

  const canvas = document.createElement("canvas");
  const scale = Math.min(1, 1280 / cameraVideo.videoWidth);
  canvas.width = Math.max(1, Math.round(cameraVideo.videoWidth * scale));
  canvas.height = Math.max(1, Math.round(cameraVideo.videoHeight * scale));
  canvas.getContext("2d").drawImage(cameraVideo, 0, 0, canvas.width, canvas.height);

  try {
    const blob = await new Promise((resolve, reject) => {
      canvas.toBlob(
        (value) => (value ? resolve(value) : reject(new Error("Could not capture camera frame."))),
        "image/jpeg",
        0.75,
      );
    });
    const response = await fetch("/api/frame", {
      method: "POST",
      headers: { "Content-Type": "image/jpeg" },
      body: blob,
    });
    if (!response.ok) {
      throw new Error(await readError(response));
    }

    const image = await response.blob();
    const nextUrl = URL.createObjectURL(image);
    processedFrame.src = nextUrl;
    processedFrame.classList.add("visible");
    cameraVideo.classList.remove("visible");
    if (previousFrameUrl) {
      URL.revokeObjectURL(previousFrameUrl);
    }
    previousFrameUrl = nextUrl;

    fpsValue.textContent = response.headers.get("X-Tracker-Fps") || "--";
    playersValue.textContent = response.headers.get("X-Tracker-Players") || "0";
    frisbeeValue.textContent = response.headers.get("X-Tracker-Frisbees") || "0";
    whiteOvalValue.textContent =
      response.headers.get("X-Tracker-White-Ovals") || "0";
    const analysisHeader = response.headers.get("X-Tracker-Analysis");
    if (analysisHeader) {
      renderAnalysis(JSON.parse(analysisHeader));
    }
    frameTag.classList.add("visible");
    frameOverlay.classList.add("visible");
    setStatus(
      "Tracking live",
      `Player confidence ≥ ${activePersonThreshold.toFixed(2)} · 2-frame confirmation.`,
    );
  } catch (error) {
    if (sessionActive && activeSession === sessionNumber) {
      sessionActive = false;
      sessionNumber += 1;
      await stopCamera();
      try {
        await fetch("/api/stop", { method: "POST" });
      } catch {
        // The server may have stopped; the original frame error is more useful.
      }
      setStatus("Frame processing stopped", error.message, true);
      showToast(error.message);
      return;
    }
  }

  if (sessionActive && activeSession === sessionNumber) {
    setTimeout(() => processNextFrame(activeSession), 80);
  }
}

async function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
  cameraVideo.srcObject = null;
  cameraVideo.classList.remove("visible");
  processedFrame.classList.remove("visible");
  processedFrame.removeAttribute("src");
  if (previousFrameUrl) {
    URL.revokeObjectURL(previousFrameUrl);
    previousFrameUrl = null;
  }
  videoPlaceholder.classList.remove("hidden");
  fpsValue.textContent = "--";
  playersValue.textContent = "0";
  frisbeeValue.textContent = "0";
  whiteOvalValue.textContent = "0";
  analysisFrame.textContent = "WAITING FOR CAMERA";
  renderAnalysis({ detections: [] });
}

async function stopSession() {
  sessionActive = false;
  sessionNumber += 1;
  stopButton.disabled = true;
  statusTitle.textContent = "Stopping session";
  statusDescription.textContent = "Releasing camera and tracker.";
  await stopCamera();
  try {
    await fetch("/api/stop", { method: "POST" });
  } catch (error) {
    showToast(`Could not contact the local tracker: ${error.message}`);
  }
  setStatus("Session stopped", "Choose a camera and start a new session.");
}

startButton.addEventListener("click", startSession);
stopButton.addEventListener("click", stopSession);
cameraSelect.addEventListener("change", refreshCameras);
refreshCameras().catch(() => {});
setStatus("Ready when you are", "Choose a camera and start a session.");
