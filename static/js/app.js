/**
 * SignVision AI - Frontend Application Logic
 * Real-time sign language detection, image upload, sentence builder & TTS
 */

document.addEventListener("DOMContentLoaded", () => {
  // Navigation Tabs
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;
      tabBtns.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      const targetEl = document.getElementById(target);
      if (targetEl) targetEl.classList.add("active");
    });
  });

  // State
  let stream = null;
  let isDetecting = false;
  let isSendingFrame = false;
  let frameIntervalId = null;
  let currentMode = "browser"; // "browser" or "server"
  let mirrorMode = true;
  let speechEnabled = true;

  // Real-time translation accumulator state
  let sentence = "";
  let lastDetectedSign = null;
  let signStableCount = 0;
  const STABLE_THRESHOLD = 5; // number of consecutive detections to accept letter

  // DOM Elements - Real-Time
  const videoEl = document.getElementById("webcam-video");
  const canvasEl = document.getElementById("canvas-output");
  const serverImgEl = document.getElementById("server-video-feed");
  const placeholderEl = document.getElementById("camera-placeholder");

  const startCamBtn = document.getElementById("btn-start-camera");
  const stopCamBtn = document.getElementById("btn-stop-camera");
  const flipCamBtn = document.getElementById("btn-flip-camera");
  const snapshotBtn = document.getElementById("btn-snapshot");
  const modeSelect = document.getElementById("mode-select");
  const cameraDeviceSelect = document.getElementById("camera-device-select");

  // Practice Mode Elements & State
  const btnTogglePractice = document.getElementById("btn-toggle-practice");
  const practicePanel = document.getElementById("practice-panel");
  const practiceTargetEl = document.getElementById("practice-target");
  const practiceScoreEl = document.getElementById("practice-score");
  const practiceFeedbackEl = document.getElementById("practice-feedback");
  const btnSkipPractice = document.getElementById("btn-skip-practice");

  // Focus Guide Box
  const btnToggleFocus = document.getElementById("btn-toggle-focus");
  const focusGuide = document.getElementById("focus-guide");
  const focusStatus = document.getElementById("focus-status");
  let focusMode = true;

  if (btnToggleFocus) {
    btnToggleFocus.addEventListener("click", () => {
      focusMode = !focusMode;
      focusStatus.textContent = focusMode ? "ON" : "OFF";
      focusStatus.style.color = focusMode ? "var(--accent-cyan)" : "var(--text-sub)";
      if (isDetecting && focusMode) {
        focusGuide.classList.remove("d-none");
      } else {
        focusGuide.classList.add("d-none");
      }
    });
  }

  let practiceActive = false;
  let practiceScore = 0;
  let currentPracticeTarget = "";
  const PRACTICE_TARGETS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "O", "V", "W", "Y", "0", "1", "2", "3", "4", "5"];

  const confSlider = document.getElementById("conf-slider");
  const confVal = document.getElementById("conf-val");
  const iouSlider = document.getElementById("iou-slider");
  const iouVal = document.getElementById("iou-val");

  const hudFps = document.getElementById("hud-fps-val");
  const hudLatency = document.getElementById("hud-latency-val");
  const hudCount = document.getElementById("hud-count-val");

  const liveSignText = document.getElementById("live-sign-text");
  const liveSignConfBar = document.getElementById("live-sign-conf-fill");
  const liveSignConfText = document.getElementById("live-sign-conf-text");
  const sentenceTextEl = document.getElementById("sentence-text");
  const recentFeedEl = document.getElementById("recent-feed-list");

  // Sentence Builder Buttons
  const btnSpeak = document.getElementById("btn-speak");
  const btnSpace = document.getElementById("btn-space");
  const btnBackspace = document.getElementById("btn-backspace");
  const btnClear = document.getElementById("btn-clear");
  const btnCopy = document.getElementById("btn-copy");

  // Hidden offscreen canvas for frame capture
  const captureCanvas = document.createElement("canvas");
  const captureCtx = captureCanvas.getContext("2d");
  const displayCtx = canvasEl.getContext("2d");

  // Sliders display update
  confSlider.addEventListener("input", (e) => {
    confVal.textContent = `${Math.round(e.target.value * 100)}%`;
  });

  iouSlider.addEventListener("input", (e) => {
    iouVal.textContent = `${Math.round(e.target.value * 100)}%`;
  });

  // Camera Mode selection
  modeSelect.addEventListener("change", (e) => {
    currentMode = e.target.value;
    if (isDetecting) {
      stopDetection();
      startDetection();
    }
  });

  // Mirror camera toggle
  flipCamBtn.addEventListener("click", () => {
    mirrorMode = !mirrorMode;
    if (mirrorMode) {
      canvasEl.classList.add("mirror");
      videoEl.classList.add("mirror");
    } else {
      canvasEl.classList.remove("mirror");
      videoEl.classList.remove("mirror");
    }
  });

  // Start Real-Time Detection
  startCamBtn.addEventListener("click", startDetection);
  stopCamBtn.addEventListener("click", stopDetection);

  async function startDetection() {
    startCamBtn.classList.add("d-none");
    stopCamBtn.classList.remove("d-none");
    placeholderEl.classList.add("d-none");

    if (currentMode === "browser") {
      canvasEl.classList.remove("d-none");
      serverImgEl.classList.add("d-none");
      if (focusMode && focusGuide) focusGuide.classList.remove("d-none");
      await startBrowserWebcam();
    } else {
      canvasEl.classList.add("d-none");
      serverImgEl.classList.remove("d-none");
      if (focusGuide) focusGuide.classList.add("d-none");
      startServerStream();
    }
  }

  async function stopDetection() {
    isDetecting = false;
    startCamBtn.classList.remove("d-none");
    stopCamBtn.classList.add("d-none");
    if (focusGuide) focusGuide.classList.add("d-none");

    if (frameIntervalId) {
      clearInterval(frameIntervalId);
      frameIntervalId = null;
    }

    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      stream = null;
    }

    if (currentMode === "server") {
      try {
        await fetch("/api/camera/stop", { method: "POST" });
      } catch (e) {
        console.error("Error stopping server camera:", e);
      }
      serverImgEl.src = "";
    }

    canvasEl.classList.add("d-none");
    serverImgEl.classList.add("d-none");
    placeholderEl.classList.remove("d-none");

    hudFps.textContent = "0";
    hudLatency.textContent = "0ms";
    hudCount.textContent = "0";
    liveSignText.textContent = "—";
    liveSignConfBar.style.width = "0%";
    liveSignConfText.textContent = "Awaiting camera...";
  }

  async function populateCameraDevices() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(d => d.kind === "videoinput");
      if (videoDevices.length > 0) {
        const currentVal = cameraDeviceSelect.value;
        cameraDeviceSelect.innerHTML = "";
        videoDevices.forEach((dev, idx) => {
          const opt = document.createElement("option");
          opt.value = dev.deviceId;
          opt.textContent = dev.label || `Camera ${idx + 1}`;
          if (dev.deviceId === currentVal) opt.selected = true;
          cameraDeviceSelect.appendChild(opt);
        });
      }
    } catch (e) {
      console.error("Error enumerating devices:", e);
    }
  }

  if (cameraDeviceSelect) {
    cameraDeviceSelect.addEventListener("change", () => {
      if (isDetecting && currentMode === "browser") {
        stopDetection();
        startDetection();
      }
    });
  }

  async function startBrowserWebcam() {
    try {
      const selectedDeviceId = cameraDeviceSelect ? cameraDeviceSelect.value : null;
      const constraints = {
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          deviceId: selectedDeviceId ? { exact: selectedDeviceId } : undefined,
          facingMode: selectedDeviceId ? undefined : "user"
        },
        audio: false
      };

      stream = await navigator.mediaDevices.getUserMedia(constraints);
      await populateCameraDevices();

      videoEl.srcObject = stream;
      await videoEl.play();

      canvasEl.width = videoEl.videoWidth || 640;
      canvasEl.height = videoEl.videoHeight || 480;
      captureCanvas.width = 416; // native model training resolution for max speed
      captureCanvas.height = 416;

      isDetecting = true;
      let lastFrameTime = performance.now();

      frameIntervalId = setInterval(async () => {
        if (!isDetecting || isSendingFrame || videoEl.readyState < 2) return;

        isSendingFrame = true;
        const now = performance.now();
        const fps = Math.round(1000 / (now - lastFrameTime));
        lastFrameTime = now;
        hudFps.textContent = fps > 0 && fps < 100 ? fps : "—";

        // Determine crop bounds (Focus Mode crops central region so hand fills the frame like in training dataset)
        const vw = videoEl.videoWidth || 640;
        const vh = videoEl.videoHeight || 480;
        let sx = 0, sy = 0, sw = vw, sh = vh;

        if (focusMode) {
          sx = Math.round(vw * 0.20);
          sy = Math.round(vh * 0.10);
          sw = Math.round(vw * 0.60);
          sh = Math.round(vh * 0.80);
        }

        // Draw cropped region to 416x416 capture canvas
        captureCtx.drawImage(videoEl, sx, sy, sw, sh, 0, 0, captureCanvas.width, captureCanvas.height);
        const frameB64 = captureCanvas.toDataURL("image/jpeg", 0.7);

        try {
          const res = await fetch("/api/detect_frame", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              frame: frameB64,
              confidence: parseFloat(confSlider.value),
              iou: parseFloat(iouSlider.value)
            })
          });

          const data = await res.json();
          if (data.status === "success") {
            hudLatency.textContent = `${Math.round(data.latency_ms)}ms`;
            hudCount.textContent = data.count;

            // Render annotated detection onto main canvas
            if (data.annotated_image) {
              const img = new Image();
              img.onload = () => {
                displayCtx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);
                if (focusMode) {
                  // Render detection inside the focus region
                  displayCtx.drawImage(img, sx, sy, sw, sh);
                  displayCtx.strokeStyle = "rgba(6, 182, 212, 0.75)";
                  displayCtx.lineWidth = 2;
                  displayCtx.setLineDash([6, 6]);
                  displayCtx.strokeRect(sx, sy, sw, sh);
                  displayCtx.setLineDash([]);
                } else {
                  displayCtx.drawImage(img, 0, 0, canvasEl.width, canvasEl.height);
                }
              };
              img.src = data.annotated_image;
            }

            // Update Sign HUD and Sentence Builder
            processRealTimeDetections(data.detections);
          }
        } catch (err) {
          console.error("Frame detection error:", err);
        } finally {
          isSendingFrame = false;
        }
      }, 70); // ~14 FPS client capture loop for smooth real-time response

    } catch (err) {
      console.error("Webcam access error:", err);
      alert("Could not access your webcam. Please allow camera permissions in your browser, or switch to Server Stream mode.");
      stopDetection();
    }
  }

  function startServerStream() {
    isDetecting = true;
    fetch("/api/camera/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        confidence: parseFloat(confSlider.value),
        iou: parseFloat(iouSlider.value)
      })
    }).then(res => res.json()).then(data => {
      if (data.status === "success") {
        serverImgEl.src = "/video_feed?" + new Date().getTime();
        // Poll status for metrics & detections
        frameIntervalId = setInterval(async () => {
          if (!isDetecting) return;
          try {
            const statusRes = await fetch("/api/camera/status");
            const st = await statusRes.json();
            hudFps.textContent = st.fps || "—";
            hudLatency.textContent = `${Math.round(st.latency_ms)}ms`;
            hudCount.textContent = st.count;
            processRealTimeDetections(st.detections);
          } catch (e) {
            console.error("Status poll error:", e);
          }
        }, 200);
      } else {
        alert("Failed to start server camera. Make sure a webcam is connected to the machine.");
        stopDetection();
      }
    });
  }

  // Real-time Detection & Sentence Builder Logic
  function processRealTimeDetections(detections) {
    if (!detections || detections.length === 0) {
      liveSignText.textContent = "—";
      liveSignConfBar.style.width = "0%";
      liveSignConfText.textContent = "No sign detected";
      signStableCount = 0;
      return;
    }

    // Sort by confidence, pick top prediction
    const top = detections.slice().sort((a, b) => b.confidence - a.confidence)[0];
    const signName = top.name;
    const confPct = Math.round(top.confidence * 100);

    // Format readable sign (e.g. "One" -> "1", "Zero" -> "0")
    const signChar = formatSignCharacter(signName);

    liveSignText.textContent = signChar;
    liveSignConfBar.style.width = `${confPct}%`;
    liveSignConfText.textContent = `${signName} (${confPct}% confidence)`;

    // Check practice mode target
    if (practiceActive && currentPracticeTarget) {
      if ((signChar === currentPracticeTarget || signName.toLowerCase() === currentPracticeTarget.toLowerCase()) && confPct >= 40) {
        practiceScore += 10;
        practiceScoreEl.textContent = practiceScore;
        practiceFeedbackEl.innerHTML = `🎉 <strong style="color:var(--accent-emerald)">Awesome!</strong> Correct sign <strong>${currentPracticeTarget}</strong> (+10 pts)!`;
        currentPracticeTarget = ""; // prevent multi-counting
        setTimeout(pickNewPracticeTarget, 1000);
      }
    }

    // Debounced sentence accumulator
    if (signChar === lastDetectedSign) {
      signStableCount++;
      if (signStableCount === STABLE_THRESHOLD) {
        appendLetter(signChar);
        addToRecentFeed(signName, confPct, top.color);
      }
    } else {
      lastDetectedSign = signChar;
      signStableCount = 1;
    }
  }

  function formatSignCharacter(name) {
    const numMap = {
      "Zero": "0", "One": "1", "Two": "2", "Three": "3", "Four": "4",
      "Five": "5", "Six": "6", "Seven": "7", "Eight": "8", "Nine": "9"
    };
    return numMap[name] || name;
  }

  function appendLetter(char) {
    sentence += char;
    sentenceTextEl.textContent = sentence;
  }

  function addToRecentFeed(name, conf, color) {
    const item = document.createElement("div");
    item.className = "feed-item";
    const timeStr = new Date().toLocaleTimeString().split(" ")[0];

    item.innerHTML = `
      <div class="feed-sign">
        <span class="sign-indicator" style="background: ${color || 'var(--accent-cyan)'}"></span>
        <span>${name}</span>
      </div>
      <div class="feed-conf">${conf}% • <span style="color:var(--text-sub)">${timeStr}</span></div>
    `;

    recentFeedEl.insertBefore(item, recentFeedEl.firstChild);
    while (recentFeedEl.children.length > 8) {
      recentFeedEl.removeChild(recentFeedEl.lastChild);
    }
  }

  // Sentence Builder Controls
  btnSpace.addEventListener("click", () => {
    sentence += " ";
    sentenceTextEl.textContent = sentence;
  });

  btnBackspace.addEventListener("click", () => {
    sentence = sentence.slice(0, -1);
    sentenceTextEl.textContent = sentence || "—";
    if (!sentence) sentenceTextEl.textContent = "";
  });

  btnClear.addEventListener("click", () => {
    sentence = "";
    sentenceTextEl.textContent = "";
  });

  btnCopy.addEventListener("click", () => {
    if (!sentence) return;
    navigator.clipboard.writeText(sentence);
    btnCopy.textContent = "Copied!";
    setTimeout(() => { btnCopy.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg> Copy`; }, 1500);
  });

  // Text-to-Speech
  btnSpeak.addEventListener("click", () => {
    if (!sentence.trim()) return;
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(sentence);
      utterance.rate = 0.9;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    } else {
      alert("Text-to-Speech is not supported in this browser.");
    }
  });

  // Snapshot frame
  snapshotBtn.addEventListener("click", () => {
    if (!isDetecting) return;
    const a = document.createElement("a");
    a.href = canvasEl.toDataURL("image/jpeg");
    a.download = `sign_detection_${Date.now()}.jpg`;
    a.click();
  });

  // Practice & Quiz Trainer Controls
  function pickNewPracticeTarget() {
    const pool = PRACTICE_TARGETS.filter(t => t !== currentPracticeTarget);
    currentPracticeTarget = pool[Math.floor(Math.random() * pool.length)];
    practiceTargetEl.textContent = currentPracticeTarget;
    practiceFeedbackEl.innerHTML = `Show sign <strong>${currentPracticeTarget}</strong> to the camera!`;
    practiceFeedbackEl.style.color = "var(--text-muted)";
  }

  if (btnTogglePractice) {
    btnTogglePractice.addEventListener("click", () => {
      practiceActive = !practiceActive;
      if (practiceActive) {
        practicePanel.classList.remove("d-none");
        btnTogglePractice.textContent = "Stop Trainer";
        btnTogglePractice.classList.add("btn-danger");
        btnTogglePractice.classList.remove("btn-secondary");
        if (!isDetecting) startDetection();
        pickNewPracticeTarget();
      } else {
        practicePanel.classList.add("d-none");
        btnTogglePractice.textContent = "Start Trainer";
        btnTogglePractice.classList.add("btn-secondary");
        btnTogglePractice.classList.remove("btn-danger");
        currentPracticeTarget = "";
      }
    });
  }

  if (btnSkipPractice) {
    btnSkipPractice.addEventListener("click", pickNewPracticeTarget);
  }

  // Keyboard Shortcuts for Rapid Translation
  window.addEventListener("keydown", (e) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;

    if (e.code === "Space") {
      e.preventDefault();
      sentence += " ";
      sentenceTextEl.textContent = sentence;
    } else if (e.code === "Backspace") {
      e.preventDefault();
      sentence = sentence.slice(0, -1);
      sentenceTextEl.textContent = sentence;
    } else if (e.key === "s" || e.key === "S") {
      e.preventDefault();
      btnSpeak.click();
    } else if (e.key === "c" || e.key === "C") {
      e.preventDefault();
      btnCopy.click();
    } else if (e.code === "Escape") {
      if (isDetecting) stopDetection();
    }
  });


  // ==========================================
  // TAB 2: Image Upload & Analysis
  // ==========================================
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const uploadConfSlider = document.getElementById("upload-conf-slider");
  const uploadConfVal = document.getElementById("upload-conf-val");
  const uploadIouSlider = document.getElementById("upload-iou-slider");
  const uploadIouVal = document.getElementById("upload-iou-val");

  const resultsArea = document.getElementById("upload-results-area");
  const imgAnnotated = document.getElementById("result-annotated");
  const imgOriginal = document.getElementById("result-original");
  const btnViewAnnotated = document.getElementById("btn-view-annotated");
  const btnViewOriginal = document.getElementById("btn-view-original");
  const btnDownloadResult = document.getElementById("btn-download-result");
  const btnTestAnother = document.getElementById("btn-test-another");

  const uploadLatency = document.getElementById("upload-latency");
  const uploadCount = document.getElementById("upload-count");
  const uploadDim = document.getElementById("upload-dim");
  const detTableBody = document.getElementById("det-table-body");
  const samplesScroll = document.getElementById("samples-scroll");

  let currentAnnotatedDataUrl = "";
  let lastUploadedFile = null;

  uploadConfSlider.addEventListener("input", (e) => {
    uploadConfVal.textContent = `${Math.round(e.target.value * 100)}%`;
    if (lastUploadedFile) processImageUpload(lastUploadedFile);
  });

  uploadIouSlider.addEventListener("input", (e) => {
    uploadIouVal.textContent = `${Math.round(e.target.value * 100)}%`;
    if (lastUploadedFile) processImageUpload(lastUploadedFile);
  });

  dropZone.addEventListener("click", () => fileInput.click());

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please upload an image file (JPG, PNG, WEBP).");
      return;
    }
    lastUploadedFile = file;
    processImageUpload(file);
  }

  async function processImageUpload(fileOrUrl) {
    const formData = new FormData();
    formData.append("confidence", uploadConfSlider.value);
    formData.append("iou", uploadIouSlider.value);

    if (typeof fileOrUrl === "string") {
      formData.append("sample_path", fileOrUrl);
    } else {
      formData.append("file", fileOrUrl);
    }

    dropZone.classList.add("d-none");
    resultsArea.classList.remove("d-none");
    resultsArea.scrollIntoView({ behavior: "smooth", block: "nearest" });

    try {
      const res = await fetch("/api/detect_image", {
        method: "POST",
        body: formData
      });

      const data = await res.json();
      if (data.status === "success") {
        renderImageResults(data);
      } else {
        alert("Detection failed: " + data.message);
      }
    } catch (err) {
      console.error("Upload error:", err);
      alert("An error occurred while uploading image.");
    }
  }

  function renderImageResults(data) {
    currentAnnotatedDataUrl = data.annotated_image;
    imgAnnotated.src = data.annotated_image;
    imgOriginal.src = data.original_image;

    // View toggling
    imgAnnotated.classList.remove("d-none");
    imgOriginal.classList.add("d-none");
    btnViewAnnotated.classList.add("btn-primary");
    btnViewAnnotated.classList.remove("btn-secondary");
    btnViewOriginal.classList.add("btn-secondary");
    btnViewOriginal.classList.remove("btn-primary");

    // Stats
    uploadLatency.textContent = `${data.latency_ms} ms`;
    uploadCount.textContent = data.count;
    uploadDim.textContent = `${data.dimensions.width} × ${data.dimensions.height}`;

    // Table rows
    detTableBody.innerHTML = "";
    if (data.detections.length === 0) {
      detTableBody.innerHTML = `<tr><td colspan="4" style="text-align:center; color:var(--text-muted); padding:1.5rem;">No signs detected with confidence ≥ ${Math.round(uploadConfSlider.value * 100)}%. Try lowering the confidence slider.</td></tr>`;
    } else {
      data.detections.forEach((det, idx) => {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td>
            <span class="class-badge" style="background:${det.color}22; color:${det.color}; border:1px solid ${det.color}55;">
              ● ${det.name}
            </span>
          </td>
          <td>
            <span style="font-family:monospace; font-weight:600;">${det.confidence_pct}%</span>
          </td>
          <td style="font-family:monospace; color:var(--text-muted); font-size:0.75rem;">
            [${det.xmin}, ${det.ymin}, ${det.xmax}, ${det.ymax}]
          </td>
          <td style="font-family:monospace; color:var(--text-muted); font-size:0.75rem;">
            ${det.width} × ${det.height}
          </td>
        `;
        detTableBody.appendChild(row);
      });
    }
  }

  // View toggle buttons
  btnViewAnnotated.addEventListener("click", () => {
    imgAnnotated.classList.remove("d-none");
    imgOriginal.classList.add("d-none");
    btnViewAnnotated.classList.add("btn-primary");
    btnViewAnnotated.classList.remove("btn-secondary");
    btnViewOriginal.classList.add("btn-secondary");
    btnViewOriginal.classList.remove("btn-primary");
  });

  btnViewOriginal.addEventListener("click", () => {
    imgOriginal.classList.remove("d-none");
    imgAnnotated.classList.add("d-none");
    btnViewOriginal.classList.add("btn-primary");
    btnViewOriginal.classList.remove("btn-secondary");
    btnViewAnnotated.classList.add("btn-secondary");
    btnViewAnnotated.classList.remove("btn-primary");
  });

  btnDownloadResult.addEventListener("click", () => {
    if (!currentAnnotatedDataUrl) return;
    const a = document.createElement("a");
    a.href = currentAnnotatedDataUrl;
    a.download = `annotated_sign_${Date.now()}.jpg`;
    a.click();
  });

  btnTestAnother.addEventListener("click", () => {
    resultsArea.classList.add("d-none");
    dropZone.classList.remove("d-none");
    fileInput.value = "";
    lastUploadedFile = null;
  });

  // Load Built-in Samples Gallery
  async function loadSamples() {
    try {
      const res = await fetch("/api/samples");
      const data = await res.json();
      if (data.status === "success" && data.samples.length > 0) {
        samplesScroll.innerHTML = "";
        data.samples.forEach(sample => {
          const chip = document.createElement("div");
          chip.className = "sample-chip";
          chip.innerHTML = `
            <img src="${sample.url}" alt="${sample.class_name}" loading="lazy">
            <span>${sample.class_name}</span>
          `;
          chip.addEventListener("click", () => {
            lastUploadedFile = sample.url;
            processImageUpload(sample.url);
          });
          samplesScroll.appendChild(chip);
        });
      }
    } catch (e) {
      console.error("Failed to load samples:", e);
    }
  }
  loadSamples();


  // ==========================================
  // TAB 3: Metrics & Class Guide
  // ==========================================
  const modalOverlay = document.getElementById("zoom-modal");
  const modalImg = document.getElementById("modal-img");
  const modalClose = document.getElementById("modal-close");
  const classSearchInput = document.getElementById("class-search");
  const classCardsGrid = document.getElementById("class-cards-grid");

  // Chart Zoom Modal
  document.querySelectorAll(".metric-chart-card").forEach(card => {
    card.addEventListener("click", () => {
      const img = card.querySelector("img");
      if (img) {
        modalImg.src = img.src;
        modalOverlay.classList.add("active");
      }
    });
  });

  modalClose.addEventListener("click", () => modalOverlay.classList.remove("active"));
  modalOverlay.addEventListener("click", (e) => {
    if (e.target === modalOverlay) modalOverlay.classList.remove("active");
  });

  // Class Reference Guide Search & Click-to-Test
  if (classCardsGrid) {
    classCardsGrid.querySelectorAll(".class-guide-chip").forEach(chip => {
      chip.addEventListener("click", () => {
        const url = chip.dataset.imgUrl;
        if (url) {
          // Switch to Upload tab
          document.querySelector('.tab-btn[data-tab="tab-upload"]').click();
          lastUploadedFile = url;
          processImageUpload(url);
        }
      });
    });
  }

  if (classSearchInput && classCardsGrid) {
    classSearchInput.addEventListener("input", (e) => {
      const query = e.target.value.toLowerCase().trim();
      const chips = classCardsGrid.querySelectorAll(".class-guide-chip");
      chips.forEach(chip => {
        const name = chip.dataset.className.toLowerCase();
        if (name.includes(query)) {
          chip.style.display = "block";
        } else {
          chip.style.display = "none";
        }
      });
    });
  }
});
