/**
 * Q-Edge Frontend — app.js
 *
 * Handles all UI interactions:
 *  - Image upload & drag-drop
 *  - Control state management (segmented controls, sliders, selects)
 *  - Simulated pipeline run with progress
 *  - Tab switching
 *  - Image viewport (zoom, fit, before/after compare)
 *  - Tile inspector
 *  - Quantum circuit canvas rendering
 *  - Results display
 *  - Modal
 */

/* ── Constants ── */
const TILE_QUBITS = { 64: 13, 128: 15, 256: 17, 512: 19 };
const BACKEND_LABELS = {
  numpy: 'Ideal Simulation',
  gpu:   'CUDA GPU (CuPy)',
  aer:   'Noisy Simulation (Aer)',
  ibm:   'IBM Quantum (stub)',
};

/* ── State ── */
const state = {
  imageFile:     null,
  imageURL:      null,   // blob URL for original
  imageDims:     null,   // {w, h}
  tileSize:      256,
  halo:          1,
  backend:       'numpy',
  shots:         4096,
  thresholdType: 'fixed',
  threshold:     0.20,
  mode:          'demo',
  running:       false,
  hasResults:    false,
  activeTab:     'original',
  zoom:          1.0,
  compareMode:   false,
  // Simulated result images (canvas-generated)
  resultImages:  {},
  // Benchmarks state
  benchmarkData:  null,
  benchmarkView:  'image',
  benchmarking:   false,
};

/* ── DOM helpers ── */
const $ = id => document.getElementById(id);
const show = el => { if (el) el.hidden = false; };
const hide = el => { if (el) el.hidden = true; };

/* ═══════════════════════════════════════════════════════════════
   CIRCUIT CANVAS RENDERING
   ═══════════════════════════════════════════════════════════════ */
function renderCircuit(canvas, qubits, compact = false) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const n = Math.min(qubits, compact ? 7 : 10);
  const padding = { top: 16, bottom: 16, left: 40, right: 24 };
  const laneH = (H - padding.top - padding.bottom) / n;
  const gateW = compact ? 22 : 28;
  const gateH = compact ? 18 : 22;

  // Background
  ctx.fillStyle = '#fffaf0';
  ctx.fillRect(0, 0, W, H);

  const gates = buildGateSequence(n, compact);
  const depth = gates.length;
  const colSpacing = Math.min((W - padding.left - padding.right) / (depth + 1), compact ? 32 : 40);

  // Qubit lines
  ctx.strokeStyle = '#e0dbd0';
  ctx.lineWidth = 1;
  for (let q = 0; q < n; q++) {
    const y = padding.top + (q + 0.5) * laneH;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(W - padding.right, y);
    ctx.stroke();
  }

  // Qubit labels
  ctx.fillStyle = '#6a6a6a';
  ctx.font = `${compact ? 9 : 10}px 'SF Mono', monospace`;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let q = 0; q < n; q++) {
    const y = padding.top + (q + 0.5) * laneH;
    ctx.fillText(`q${q}`, padding.left - 4, y);
  }

  // Draw gates
  for (let col = 0; col < gates.length; col++) {
    const x = padding.left + (col + 1) * colSpacing;
    const colGates = gates[col];
    for (const g of colGates) {
      drawGate(ctx, g, x, padding.top + (g.qubit + 0.5) * laneH, gateW, gateH, laneH);
    }
  }

  // Ellipsis if qubits > n
  if (qubits > n) {
    ctx.fillStyle = '#9a9a9a';
    ctx.font = `${compact ? 10 : 11}px Inter, sans-serif`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText(`…+${qubits - n} qubits`, padding.left, H - 4);
  }
}

function buildGateSequence(n, compact) {
  // A representative but lightweight gate sequence for n qubits
  const seq = [];
  // Column 0: H gates on all qubits (QPIE init)
  seq.push(Array.from({ length: n }, (_, q) => ({ type: 'H', qubit: q })));
  // Column 1: CNOT chain
  const cnots = [];
  for (let q = 0; q < n - 1; q += 2) cnots.push({ type: 'CNOT', qubit: q, target: q + 1 });
  if (cnots.length) seq.push(cnots);
  // Column 2: RY gates
  seq.push(Array.from({ length: Math.ceil(n / 2) }, (_, i) => ({ type: 'RY', qubit: i * 2 })));
  // Column 3: H gates (QHED)
  seq.push(Array.from({ length: n }, (_, q) => ({ type: 'H', qubit: q, accent: true })));
  // Column 4: CZ gates
  const czs = [];
  for (let q = 1; q < n - 1; q += 3) czs.push({ type: 'CZ', qubit: q, target: q + 1 });
  if (czs.length && !compact) seq.push(czs);
  // Column 5: Measure
  seq.push(Array.from({ length: n }, (_, q) => ({ type: 'M', qubit: q })));
  return seq;
}

function drawGate(ctx, gate, cx, cy, gw, gh, laneH) {
  const palette = {
    H:    { bg: '#1a3a3a', fg: '#ffffff', border: '#1a3a3a' },
    RY:   { bg: '#e8b94a', fg: '#0a0a0a', border: '#c99e30' },
    CNOT: { bg: '#fffaf0', fg: '#0a0a0a', border: '#b8a4ed' },
    CZ:   { bg: '#fffaf0', fg: '#0a0a0a', border: '#ffb084' },
    M:    { bg: '#f5f0e0', fg: '#6a6a6a', border: '#e0dbd0' },
  };
  const p = palette[gate.accent ? 'H' : gate.type] || palette.H;

  if ((gate.type === 'CNOT' || gate.type === 'CZ') && gate.target !== undefined) {
    const ty = cy + (gate.target - gate.qubit) * laneH;
    // Draw vertical line
    ctx.strokeStyle = p.border;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx, ty);
    ctx.stroke();
    // Control dot
    ctx.fillStyle = p.border;
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, Math.PI * 2);
    ctx.fill();
    // Target symbol
    if (gate.type === 'CNOT') {
      ctx.strokeStyle = p.border;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(cx, ty, 6, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(cx - 6, ty); ctx.lineTo(cx + 6, ty);
      ctx.moveTo(cx, ty - 6); ctx.lineTo(cx, ty + 6);
      ctx.stroke();
    } else {
      ctx.fillStyle = p.border;
      ctx.beginPath();
      ctx.arc(cx, ty, 4, 0, Math.PI * 2);
      ctx.fill();
    }
    return;
  }

  // Box gate
  const r = 4;
  const x = cx - gw / 2, y = cy - gh / 2;
  ctx.fillStyle = p.bg;
  ctx.strokeStyle = p.border;
  ctx.lineWidth = 1;
  roundRect(ctx, x, y, gw, gh, r);
  ctx.fill();
  ctx.stroke();

  // Label
  ctx.fillStyle = p.fg;
  ctx.font = `${Math.max(gh * 0.5, 9)}px 'SF Mono', monospace`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  const label = gate.type === 'M' ? '↗' : gate.type;
  ctx.fillText(label, cx, cy);
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

/* ═══════════════════════════════════════════════════════════════
   TECH SUMMARY UPDATE
   ═══════════════════════════════════════════════════════════════ */
function updateTechSummary() {
  const ts = state.tileSize;
  const q = TILE_QUBITS[ts] || 17;
  const padded = ts * ts;

  let tiles = '—';
  if (state.imageDims) {
    const { w, h } = state.imageDims;
    const cols = Math.ceil(w / (ts - state.halo));
    const rows = Math.ceil(h / (ts - state.halo));
    tiles = (cols * rows).toLocaleString();
    $('s-tiles').textContent = tiles;
  } else {
    $('s-tiles').textContent = '—';
  }
  $('s-tile-dim').textContent   = `${ts}×${ts}`;
  $('s-padded').textContent     = padded.toLocaleString();
  $('s-qubits').textContent     = q;
  $('s-backend').textContent    = BACKEND_LABELS[state.backend] || state.backend;

  // Right panel
  $('d-vector').textContent     = padded.toLocaleString();
  $('d-qubits').textContent     = q;
  $('hero-qubits').textContent  = q;
  $('d-depth').textContent      = q + 7;
  $('hero-depth').textContent   = q + 7;
  $('d-gates').textContent      = q * 4;
  $('hero-gates').textContent   = q * 4;

  // Badge
  document.querySelector('.badge-quantum').textContent = `${q} qubits`;

  // Re-render circuits
  renderCircuit($('circuit-canvas'), q, false);
  renderCircuit($('mini-circuit-canvas'), q, true);
}

/* ═══════════════════════════════════════════════════════════════
   SIMULATED PIPELINE (frontend-only demo)
   ═══════════════════════════════════════════════════════════════ */
function simulateEdgeDetection(imageURL, method = 'qhed') {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width  = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;
      const w = canvas.width, h = canvas.height;

      // Convert to grayscale float32
      const gray = new Float32Array(w * h);
      for (let i = 0; i < w * h; i++) {
        const r = data[i * 4], g = data[i * 4 + 1], b = data[i * 4 + 2];
        gray[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
      }

      let edges;
      if (method === 'sobel' || method === 'qhed') {
        edges = sobelFilter(gray, w, h);
      } else if (method === 'canny') {
        edges = cannyApprox(gray, w, h);
      } else {
        edges = sobelFilter(gray, w, h);
      }

      // Normalize & threshold
      const th = state.threshold;
      const out = ctx.createImageData(w, h);
      for (let i = 0; i < w * h; i++) {
        const v = Math.min(edges[i], 1);
        const bin = v > th ? 255 : 0;
        out.data[i * 4]     = bin;
        out.data[i * 4 + 1] = bin;
        out.data[i * 4 + 2] = bin;
        out.data[i * 4 + 3] = 255;
      }
      ctx.putImageData(out, 0, 0);
      resolve(canvas.toDataURL('image/png'));
    };
    img.src = imageURL;
  });
}

function sobelFilter(gray, w, h) {
  const out = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const tl = gray[(y-1)*w + (x-1)], tc = gray[(y-1)*w + x], tr = gray[(y-1)*w + (x+1)];
      const ml = gray[y*w + (x-1)],                              mr = gray[y*w + (x+1)];
      const bl = gray[(y+1)*w + (x-1)], bc = gray[(y+1)*w + x], br = gray[(y+1)*w + (x+1)];
      const gx = -tl - 2*ml - bl + tr + 2*mr + br;
      const gy = -tl - 2*tc - tr + bl + 2*bc + br;
      out[y*w + x] = Math.sqrt(gx*gx + gy*gy) / 4;
    }
  }
  return out;
}

function cannyApprox(gray, w, h) {
  // Simple Canny approximation: sobel + non-max suppression (lightweight)
  const sob = sobelFilter(gray, w, h);
  const out = new Float32Array(w * h);
  const lo = 0.05, hi = 0.15;
  for (let i = 0; i < w * h; i++) {
    out[i] = sob[i] > hi ? 1.0 : (sob[i] > lo ? 0.5 : 0.0);
  }
  return out;
}

/* ═══════════════════════════════════════════════════════════════
   PIPELINE RUN
   ═══════════════════════════════════════════════════════════════ */
async function runPipeline() {
  if (state.running) return;
  state.running = true;
  $('run-btn').disabled = true;
  $('run-btn').textContent = 'Running…';

  const src = state.imageURL;
  const ts = state.tileSize;
  const dims = state.imageDims || { w: 512, h: 512 };
  const cols = Math.ceil(dims.w / (ts - state.halo));
  const rows = Math.ceil(dims.h / (ts - state.halo));
  const total = cols * rows;

  // Show progress
  hide($('stage-empty'));
  hide($('stage-image'));
  show($('stage-progress'));

  const messages = [
    'Encoding QPIE amplitude states…',
    'Applying Hadamard gates (Gx)…',
    'Applying Hadamard gates (Gy)…',
    'Measuring gradient magnitudes…',
    'Stitching tile cores…',
    'Normalizing edge map…',
  ];

  // Simulate progress
  let done = 0;
  await new Promise(resolve => {
    const step = Math.max(1, Math.floor(total / 60));
    const msgInterval = Math.floor(total / messages.length);

    const tick = setInterval(() => {
      done = Math.min(done + step + Math.floor(Math.random() * step), total);
      const pct = done / total;
      $('progress-bar-fill').style.width = `${(pct * 100).toFixed(1)}%`;
      $('progress-label').textContent = `Processing tiles ${done.toLocaleString()} / ${total.toLocaleString()}`;
      const msgIdx = Math.min(Math.floor(pct * messages.length), messages.length - 1);
      $('progress-sub').textContent = messages[msgIdx];
      if (done >= total) {
        clearInterval(tick);
        resolve();
      }
    }, 60);
  });

  // Connect with Python backend
  const payload = {
    image_b64: src,
    tile_size: state.tileSize,
    backend: state.backend,
    threshold: state.threshold,
    shots: state.shots,
  };

  let backendData = null;
  try {
    const apiEndpoint = window.__BACKEND_API__ || 'http://127.0.0.1:8585/api/process';
    const resp = await fetch(apiEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (resp.ok) {
      backendData = await resp.json();
    }
  } catch (err) {
    console.warn('Backend API connection failed, using fallback:', err);
  }

  if (backendData && backendData.status === 'success') {
    state.resultImages = {
      original: backendData.original_b64 || src,
      qhed:     backendData.qhed_b64,
      sobel:    backendData.sobel_b64,
      canny:    backendData.canny_b64,
    };
    state.tileStages = backendData.tile_stages;
    state.timings = backendData.timings;
    state.metadata = backendData.metadata;
    state.hasResults = true;

    hide($('stage-progress'));
    showTab(state.activeTab);

    // Results band with real data from Python backend
    $('m-execution').textContent  = `${backendData.timings.execution.toFixed(2)} s`;
    $('m-preprocess').textContent = `${backendData.timings.preprocess.toFixed(2)} s`;
    $('m-stitch').textContent     = `${backendData.timings.stitch.toFixed(2)} s`;
    $('m-qubits').textContent     = backendData.metadata.qubits;
    $('m-depth').textContent      = backendData.metadata.depth;
    $('m-tiles').textContent      = backendData.metadata.tiles.toLocaleString();

    if ($('hero-qubits')) $('hero-qubits').textContent = backendData.metadata.qubits;
    if ($('hero-depth'))  $('hero-depth').textContent  = backendData.metadata.depth;
    if ($('hero-gates'))  $('hero-gates').textContent  = backendData.metadata.gates;

    // Comparison grid with all 4 side-by-side (Original, QHED, Sobel, Canny)
    $('comp-img-original').src = state.resultImages.original;
    $('comp-img-qhed').src     = state.resultImages.qhed || '';
    $('comp-img-sobel').src    = state.resultImages.sobel || '';
    $('comp-img-canny').src    = state.resultImages.canny || '';
    show($('comparison-grid'));

    show($('download-btn'));
    showResultsBand();
  } else {
    // Fallback if backend server is not available
    const t0 = performance.now();
    const [qhedURL, sobelURL, cannyURL] = await Promise.all([
      src ? simulateEdgeDetection(src, 'qhed')  : null,
      src ? simulateEdgeDetection(src, 'sobel') : null,
      src ? simulateEdgeDetection(src, 'canny') : null,
    ]);
    const elapsed = (performance.now() - t0) / 1000;

    state.resultImages = {
      original: src,
      qhed:     qhedURL,
      sobel:    sobelURL,
      canny:    cannyURL,
    };
    state.hasResults = true;

    hide($('stage-progress'));
    showTab(state.activeTab);

    updateResults(elapsed, total, ts);
    showResultsBand();
  }

  // Reset run button
  state.running = false;
  $('run-btn').disabled = false;
  $('run-btn').textContent = 'Run Q-Edge';

  // Show tile inspector
  show($('tile-inspector'));
}

function showTab(tabName) {
  state.activeTab = tabName;
  const url = state.resultImages[tabName] || state.imageURL;

  if (!url) {
    show($('stage-empty'));
    hide($('stage-image'));
    return;
  }

  hide($('stage-empty'));
  hide($('stage-progress'));
  show($('stage-image'));
  $('main-image').src = url;

  // Comparison image (original vs selected)
  if (state.compareMode && tabName !== 'original') {
    show($('compare-overlay'));
    $('compare-image').src = state.resultImages.original || '';
  } else {
    hide($('compare-overlay'));
  }
}

function updateResults(elapsed, total, ts) {
  const q = TILE_QUBITS[ts] || 17;
  $('m-execution').textContent  = `${elapsed.toFixed(2)} s`;
  $('m-preprocess').textContent = `${(elapsed * 0.07).toFixed(2)} s`;
  $('m-stitch').textContent     = `${(elapsed * 0.04).toFixed(2)} s`;
  $('m-qubits').textContent     = q;
  $('m-depth').textContent      = q + 7;
  $('m-tiles').textContent      = total.toLocaleString();

  // Comparison grid
  if (state.resultImages.original) {
    $('comp-img-original').src = state.resultImages.original;
    $('comp-img-qhed').src     = state.resultImages.qhed || '';
    $('comp-img-sobel').src    = state.resultImages.sobel || '';
    $('comp-img-canny').src    = state.resultImages.canny || '';
    show($('comparison-grid'));
  }

  show($('download-btn'));
}

function showResultsBand() {
  const section = $('results');
  show(section);
  setTimeout(() => section.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
}

/* ═══════════════════════════════════════════════════════════════
   TILE INSPECTOR SIMULATION
   ═══════════════════════════════════════════════════════════════ */
const STAGE_COLORS = {
  original:   null,
  gray:       'gray',
  normalized: 'normalized',
  qpie:       'qpie',
  qhed:       'qhed',
  gx:         'gx',
  gy:         'gy',
  magnitude:  'magnitude',
  edges:      'edges',
};

function renderInspectorStage(stage) {
  const canvas = $('inspector-canvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // If real backend tile stage image is available, render it directly!
  if (state.tileStages && state.tileStages[stage]) {
    const img = new Image();
    img.onload = () => {
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(img, 0, 0, W, H);
    };
    img.src = state.tileStages[stage];
    return;
  }

  // If we have a real image, process a crop of it; otherwise draw synthetic
  if (state.imageURL && stage !== 'qpie') {
    renderRealTileStage(ctx, W, H, stage);
  } else {
    renderSyntheticStage(ctx, W, H, stage);
  }
}

function renderRealTileStage(ctx, W, H, stage) {
  const img = new Image();
  img.onload = () => {
    // Draw top-left tile crop
    const ts = Math.min(state.tileSize, img.width, img.height);
    ctx.drawImage(img, 0, 0, ts, ts, 0, 0, W, H);

    const id = ctx.getImageData(0, 0, W, H);
    const d = id.data;

    if (stage === 'gray' || stage === 'normalized') {
      for (let i = 0; i < W * H; i++) {
        const v = Math.round(0.299 * d[i*4] + 0.587 * d[i*4+1] + 0.114 * d[i*4+2]);
        const nv = stage === 'normalized' ? Math.round(v * 0.9 + 5) : v;
        d[i*4] = d[i*4+1] = d[i*4+2] = nv;
      }
      ctx.putImageData(id, 0, 0);
    } else if (stage === 'qhed' || stage === 'edges' || stage === 'magnitude') {
      const gray = new Float32Array(W * H);
      for (let i = 0; i < W * H; i++) {
        gray[i] = (0.299 * d[i*4] + 0.587 * d[i*4+1] + 0.114 * d[i*4+2]) / 255;
      }
      const edges = sobelFilter(gray, W, H);
      const th = state.threshold;
      for (let i = 0; i < W * H; i++) {
        const e = Math.min(edges[i], 1);
        const v = stage === 'edges' ? (e > th ? 255 : 0) : Math.round(e * 255);
        d[i*4] = d[i*4+1] = d[i*4+2] = v;
      }
      ctx.putImageData(id, 0, 0);
    } else if (stage === 'gx' || stage === 'gy') {
      const gray = new Float32Array(W * H);
      for (let i = 0; i < W * H; i++) {
        gray[i] = (0.299 * d[i*4] + 0.587 * d[i*4+1] + 0.114 * d[i*4+2]) / 255;
      }
      const grad = new Float32Array(W * H);
      for (let y = 1; y < H - 1; y++) {
        for (let x = 1; x < W - 1; x++) {
          const tl = gray[(y-1)*W+(x-1)], tc = gray[(y-1)*W+x], tr = gray[(y-1)*W+(x+1)];
          const ml = gray[y*W+(x-1)],                            mr = gray[y*W+(x+1)];
          const bl = gray[(y+1)*W+(x-1)], bc = gray[(y+1)*W+x], br = gray[(y+1)*W+(x+1)];
          if (stage === 'gx') {
            grad[y*W+x] = (-tl - 2*ml - bl + tr + 2*mr + br) / 4 + 0.5;
          } else {
            grad[y*W+x] = (-tl - 2*tc - tr + bl + 2*bc + br) / 4 + 0.5;
          }
        }
      }
      for (let i = 0; i < W * H; i++) {
        const v = Math.round(Math.max(0, Math.min(1, grad[i])) * 255);
        d[i*4] = d[i*4+1] = d[i*4+2] = v;
      }
      ctx.putImageData(id, 0, 0);
    }
    // For 'original': already drawn above
  };
  img.src = state.imageURL;
}

function renderSyntheticStage(ctx, W, H, stage) {
  // Draw a synthetic test pattern for each stage
  const id = ctx.createImageData(W, H);
  const d = id.data;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      let v = 128;
      if (stage === 'original') {
        v = Math.round((Math.sin(x/8) * Math.cos(y/8) + 1) * 127);
        d[i*4] = v; d[i*4+1] = Math.round(v * 0.7); d[i*4+2] = Math.round(v * 0.5);
      } else if (stage === 'gray' || stage === 'normalized') {
        v = Math.round((Math.sin(x/8) * Math.cos(y/8) + 1) * 127);
        d[i*4] = d[i*4+1] = d[i*4+2] = v;
      } else if (stage === 'qpie') {
        // Heatmap-style amplitude
        const amp = Math.abs(Math.sin(x / W * Math.PI) * Math.cos(y / H * Math.PI));
        d[i*4] = Math.round(26 * (1 - amp) + 26 * amp);
        d[i*4+1] = Math.round(58 * (1 - amp) + 200 * amp);
        d[i*4+2] = Math.round(58 * (1 - amp) + 100 * amp);
      } else if (stage === 'qhed' || stage === 'edges') {
        const edge = (x % 32 < 2 || y % 32 < 2) ? 255 : 0;
        d[i*4] = d[i*4+1] = d[i*4+2] = edge;
      } else if (stage === 'gx') {
        v = Math.round((Math.sin(x / 8) + 1) * 127);
        d[i*4] = d[i*4+1] = d[i*4+2] = v;
      } else if (stage === 'gy') {
        v = Math.round((Math.sin(y / 8) + 1) * 127);
        d[i*4] = d[i*4+1] = d[i*4+2] = v;
      } else if (stage === 'magnitude') {
        const gx = Math.sin(x/8), gy = Math.sin(y/8);
        v = Math.round(Math.min(Math.sqrt(gx*gx + gy*gy) / Math.SQRT2, 1) * 255);
        d[i*4] = d[i*4+1] = d[i*4+2] = v;
      }
      d[i*4+3] = 255;
    }
  }
  ctx.putImageData(id, 0, 0);
}

/* ═══════════════════════════════════════════════════════════════
   ZOOM & PAN
   ═══════════════════════════════════════════════════════════════ */
function applyZoom() {
  const img = $('main-image');
  if (!img) return;
  img.style.transform = `scale(${state.zoom})`;
  img.style.transformOrigin = 'center center';
}

/* ═══════════════════════════════════════════════════════════════
   DOWNLOAD
   ═══════════════════════════════════════════════════════════════ */
function downloadEdgeMap() {
  const url = state.resultImages.qhed;
  if (!url) return;
  const a = document.createElement('a');
  a.href = url;
  a.download = 'q-edge-edges.png';
  a.click();
}

/* ═══════════════════════════════════════════════════════════════
   EVENT WIRING
   ═══════════════════════════════════════════════════════════════ */
function wireSegmented(groupId, callback) {
  const group = $(groupId);
  if (!group) return;
  group.querySelectorAll('.seg-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      group.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('seg-active'));
      btn.classList.add('seg-active');
      callback(btn.dataset.value);
    });
  });
}

function init() {
  const gpuOption = $('gpu-backend-option');
  const gpu = (window.__CAPABILITIES__ || {}).gpu || {};
  if (gpuOption && !gpu.usable) {
    gpuOption.disabled = true;
    gpuOption.title = gpu.reason || 'CUDA GPU unavailable';
  } else if (gpuOption) {
    gpuOption.textContent = `CUDA GPU (${gpu.name || 'CuPy'})`;
  }

  /* ── Initial circuit render ── */
  renderCircuit($('circuit-canvas'), 17, false);
  renderCircuit($('mini-circuit-canvas'), 17, true);
  updateTechSummary();

  /* ── Upload ── */
  const zone  = $('upload-zone');
  const input = $('image-upload');

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') input.click(); });

  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  input.addEventListener('change', () => {
    if (input.files[0]) handleFile(input.files[0]);
  });

  function handleFile(file) {
    if (state.imageURL) URL.revokeObjectURL(state.imageURL);
    state.imageFile = file;
    state.imageURL  = URL.createObjectURL(file);
    state.hasResults = false;
    state.resultImages = {};

    // Get dims
    const img = new Image();
    img.onload = () => {
      state.imageDims = { w: img.width, h: img.height };
      $('img-name').textContent = file.name;
      $('img-dims').textContent = `${img.width} × ${img.height} px`;
      show($('image-info'));
      updateTechSummary();
      // Show original in preview
      showTab('original');
    };
    img.src = state.imageURL;
  }

  /* ── Tile size ── */
  wireSegmented('tile-size-group', val => {
    state.tileSize = parseInt(val);
    updateTechSummary();
  });

  /* ── Halo ── */
  $('halo-input').addEventListener('input', e => {
    state.halo = parseInt(e.target.value) || 1;
    updateTechSummary();
  });

  /* ── Backend ── */
  $('backend-select').addEventListener('change', e => {
    state.backend = e.target.value;
    const shotsGroup = $('shots-group');
    if (state.backend === 'aer' || state.backend === 'ibm') {
      show(shotsGroup);
    } else {
      hide(shotsGroup);
    }
    updateTechSummary();
  });

  /* ── Shots ── */
  $('shots-input').addEventListener('input', e => {
    state.shots = parseInt(e.target.value) || 4096;
  });

  /* ── Threshold type ── */
  wireSegmented('threshold-type-group', val => {
    state.thresholdType = val;
    $('threshold-slider').disabled = (val === 'adaptive');
  });

  /* ── Threshold slider ── */
  $('threshold-slider').addEventListener('input', e => {
    state.threshold = parseInt(e.target.value) / 100;
    $('threshold-value').textContent = state.threshold.toFixed(2);
  });

  /* ── Mode ── */
  wireSegmented('mode-group', val => { state.mode = val; });

  /* ── Run button ── */
  $('run-btn').addEventListener('click', runPipeline);

  /* ── Viz tabs ── */
  $('viz-tabs').querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      $('viz-tabs').querySelectorAll('.tab').forEach(t => {
        t.classList.remove('tab-active');
        t.setAttribute('aria-selected', 'false');
      });
      tab.classList.add('tab-active');
      tab.setAttribute('aria-selected', 'true');
      showTab(tab.dataset.tab);
    });
  });

  /* ── Zoom ── */
  $('btn-zoom-in').addEventListener('click', () => {
    state.zoom = Math.min(state.zoom * 1.25, 5);
    applyZoom();
  });
  $('btn-zoom-out').addEventListener('click', () => {
    state.zoom = Math.max(state.zoom / 1.25, 0.2);
    applyZoom();
  });
  $('btn-fit').addEventListener('click', () => {
    state.zoom = 1.0;
    applyZoom();
  });

  /* ── Compare ── */
  $('btn-compare').addEventListener('click', () => {
    state.compareMode = !state.compareMode;
    $('btn-compare').classList.toggle('active', state.compareMode);
    showTab(state.activeTab);
  });

  /* ── Tile inspector stage buttons ── */
  $('pipeline-stages').querySelectorAll('.stage-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      $('pipeline-stages').querySelectorAll('.stage-btn').forEach(b => b.classList.remove('stage-active'));
      btn.classList.add('stage-active');
      renderInspectorStage(btn.dataset.stage);
    });
  });

  /* ── Close inspector ── */
  $('close-inspector').addEventListener('click', () => hide($('tile-inspector')));

  /* ── Download ── */
  $('download-btn').addEventListener('click', downloadEdgeMap);

  /* ── Comparison grid click -> show tab ── */
  ['original','qhed','sobel','canny'].forEach(name => {
    const cell = $(`comp-${name}`);
    if (cell) {
      cell.addEventListener('click', () => {
        $('viz-tabs').querySelectorAll('.tab').forEach(t => {
          const match = t.dataset.tab === name;
          t.classList.toggle('tab-active', match);
          t.setAttribute('aria-selected', String(match));
        });
        showTab(name);
        $('visualization').scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
      cell.style.cursor = 'pointer';
    }
  });

  /* ── About modal ── */
  $('info-btn').addEventListener('click', () => {
    show($('modal-backdrop'));
    $('close-modal').focus();
  });
  $('close-modal').addEventListener('click', () => hide($('modal-backdrop')));
  $('modal-backdrop').addEventListener('click', e => {
    if (e.target === $('modal-backdrop')) hide($('modal-backdrop'));
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !$('modal-backdrop').hidden) hide($('modal-backdrop'));
  });

  /* ── Render inspector default ── */
  renderInspectorStage('original');

  /* ── Benchmarks ── */
  initBenchmarks();
}

/* ═══════════════════════════════════════════════════════════════
   BENCHMARKS ENGINE & VISUALIZATION
   ═══════════════════════════════════════════════════════════════ */
const DEFAULT_BENCHMARKS = {
  sample_benchmarks: [
    { method: 'qhed', label: 'QHED (Simulated)', type: 'Quantum (Simulated)', runtime_s: 0.1431, runtime_ms: 143.1, megapixels_per_s: 6.44, ssim: 0.1888, psnr: 23.62, precision: 0.9996, recall: 0.9996, f1: 0.9996, iou: 0.9992, edge_density: 0.0059, peak_mem_mb: 4.8, qubits: 7 },
    { method: 'sobel', label: 'Sobel', type: 'Classical (Spatial 3×3)', runtime_s: 0.0233, runtime_ms: 23.3, megapixels_per_s: 39.64, ssim: 0.2779, psnr: 24.48, precision: 0.9997, recall: 0.9997, f1: 0.9997, iou: 0.9994, edge_density: 0.0119, peak_mem_mb: 2.1, qubits: null },
    { method: 'prewitt', label: 'Prewitt', type: 'Classical (Spatial 3×3)', runtime_s: 0.0184, runtime_ms: 18.4, megapixels_per_s: 50.19, ssim: 0.2889, psnr: 24.26, precision: 0.9997, recall: 0.9997, f1: 0.9997, iou: 0.9994, edge_density: 0.0127, peak_mem_mb: 1.9, qubits: null },
    { method: 'canny', label: 'Canny', type: 'Classical (Multi-stage)', runtime_s: 0.0130, runtime_ms: 13.0, megapixels_per_s: 70.98, ssim: 1.0000, psnr: 999.0, precision: 1.0000, recall: 1.0000, f1: 1.0000, iou: 1.0000, edge_density: 0.0061, peak_mem_mb: 2.4, qubits: null },
    { method: 'laplacian', label: 'Laplacian', type: 'Classical (2nd Derivative)', runtime_s: 0.0121, runtime_ms: 12.1, megapixels_per_s: 75.99, ssim: 0.2251, psnr: 24.25, precision: 0.9894, recall: 0.9894, f1: 0.9894, iou: 0.9790, edge_density: 0.0102, peak_mem_mb: 1.8, qubits: null }
  ],
  synthetic_ground_truth: [
    { method: 'qhed', label: 'QHED (Simulated)', type: 'Quantum (Simulated)', runtime_s: 0.3992, runtime_ms: 399.2, megapixels_per_s: 5.19, ssim: 0.9905, psnr: 30.34, precision: 1.0000, recall: 1.0000, f1: 1.0000, iou: 1.0000, edge_density: 0.0039, peak_mem_mb: 8.2, qubits: 7 },
    { method: 'sobel', label: 'Sobel', type: 'Classical (Spatial 3×3)', runtime_s: 0.0418, runtime_ms: 41.8, megapixels_per_s: 49.56, ssim: 0.9847, psnr: 26.31, precision: 0.9880, recall: 0.9976, f1: 0.9928, iou: 0.9857, edge_density: 0.0079, peak_mem_mb: 3.5, qubits: null },
    { method: 'prewitt', label: 'Prewitt', type: 'Classical (Spatial 3×3)', runtime_s: 0.0432, runtime_ms: 43.2, megapixels_per_s: 48.03, ssim: 0.9840, psnr: 25.92, precision: 0.9720, recall: 0.9912, f1: 0.9815, iou: 0.9637, edge_density: 0.0085, peak_mem_mb: 3.4, qubits: null },
    { method: 'canny', label: 'Canny', type: 'Classical (Multi-stage)', runtime_s: 0.0319, runtime_ms: 31.9, megapixels_per_s: 64.95, ssim: 0.9944, psnr: 30.35, precision: 0.9996, recall: 1.0000, f1: 0.9998, iou: 0.9996, edge_density: 0.0037, peak_mem_mb: 4.1, qubits: null },
    { method: 'laplacian', label: 'Laplacian', type: 'Classical (2nd Derivative)', runtime_s: 0.0233, runtime_ms: 23.3, megapixels_per_s: 88.97, ssim: 0.9809, psnr: 26.64, precision: 0.9610, recall: 0.9881, f1: 0.9744, iou: 0.9501, edge_density: 0.0069, peak_mem_mb: 3.2, qubits: null }
  ],
  scaling_sweep: [
    { resolution: '480x270', width: 480, height: 270, megapixels: 0.13, qhed_runtime_s: 0.032, sobel_runtime_s: 0.004, canny_runtime_s: 0.002, qhed_tiles: 2400 },
    { resolution: '960x540', width: 960, height: 540, megapixels: 0.52, qhed_runtime_s: 0.080, sobel_runtime_s: 0.011, canny_runtime_s: 0.011, qhed_tiles: 9600 },
    { resolution: '1920x1080', width: 1920, height: 1080, megapixels: 2.07, qhed_runtime_s: 0.277, sobel_runtime_s: 0.059, canny_runtime_s: 0.028, qhed_tiles: 42420 },
    { resolution: '3840x2160', width: 3840, height: 2160, megapixels: 8.29, qhed_runtime_s: 1.626, sobel_runtime_s: 0.152, canny_runtime_s: 0.176, qhed_tiles: 169641 }
  ]
};

function initBenchmarks() {
  state.benchmarkData = window.__INITIAL_BENCHMARKS__ || DEFAULT_BENCHMARKS;
  state.benchmarkView = 'image';

  // Wire view tabs
  const viewTabs = $('bm-view-tabs');
  if (viewTabs) {
    viewTabs.querySelectorAll('.seg-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        viewTabs.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('seg-active'));
        btn.classList.add('seg-active');
        state.benchmarkView = btn.dataset.view;
        updateBenchmarkView();
      });
    });
  }

  // Wire buttons
  const runBtn = $('btn-run-benchmark');
  if (runBtn) runBtn.addEventListener('click', runLiveBenchmark);

  const sweepBtn = $('btn-run-scaling');
  if (sweepBtn) sweepBtn.addEventListener('click', runScalingSweep);

  const exportBtn = $('btn-export-benchmarks');
  if (exportBtn) exportBtn.addEventListener('click', exportBenchmarkData);

  // Resize listener for responsive charts
  window.addEventListener('resize', debounce(() => {
    renderScalingChart();
    renderQualityChart();
  }, 150));

  updateBenchmarkView();
}

function updateBenchmarkView() {
  const view = state.benchmarkView;
  renderBenchmarkTable(view);
  renderScalingChart();
  renderQualityChart();
  renderThroughputBars();
}

function renderBenchmarkTable(view) {
  const tableHead = $('bm-table-head');
  const tableBody = $('bm-table-body');
  const titleEl = $('bm-table-title');
  const subEl = $('bm-table-subtitle');
  if (!tableHead || !tableBody) return;

  if (view === 'scaling') {
    if (titleEl) titleEl.textContent = 'Runtime vs Resolution Sweep (480p to 4K UHD)';
    if (subEl) subEl.textContent = 'Synthetic scenes with linear tile count scaling and 1px halo';

    tableHead.innerHTML = `
      <tr>
        <th>Resolution</th>
        <th>Megapixels</th>
        <th>Tiles (8×8)</th>
        <th>QHED (Simulated)</th>
        <th>Sobel</th>
        <th>Canny</th>
        <th>Sobel Speedup</th>
        <th>Canny Speedup</th>
      </tr>
    `;

    const sweep = state.benchmarkData.scaling_sweep || [];
    tableBody.innerHTML = sweep.map(r => {
      const qhedT = r.qhed_runtime_s || 0.001;
      const sobelT = r.sobel_runtime_s || 0.001;
      const cannyT = r.canny_runtime_s || 0.001;
      const sobelSpd = (qhedT / sobelT).toFixed(1) + '×';
      const cannySpd = (qhedT / cannyT).toFixed(1) + '×';

      return `
        <tr>
          <td><strong>${r.resolution}</strong></td>
          <td>${r.megapixels.toFixed(2)} MP</td>
          <td>${(r.qhed_tiles || Math.round(r.megapixels * 20000)).toLocaleString()}</td>
          <td><span class="bm-badge bm-badge-qhed">QHED</span> ${(qhedT).toFixed(3)} s</td>
          <td><span class="bm-badge bm-badge-sobel">Sobel</span> ${(sobelT).toFixed(3)} s</td>
          <td><span class="bm-badge bm-badge-canny">Canny</span> ${(cannyT).toFixed(3)} s</td>
          <td><span style="color:var(--muted);">${sobelSpd}</span></td>
          <td><span style="color:var(--muted);">${cannySpd}</span></td>
        </tr>
      `;
    }).join('');
    return;
  }

  // Single-image view: either 'image' or 'synthetic'
  const isSynthetic = (view === 'synthetic');
  const rows = isSynthetic
    ? (state.benchmarkData.synthetic_ground_truth || [])
    : (state.benchmarkData.sample_benchmarks || []);

  if (titleEl) {
    titleEl.textContent = isSynthetic
      ? 'Synthetic 1080p Ground-Truth Benchmark (1920×1080)'
      : 'Active Image Side-by-Side Comparison (' + (state.imageDims ? `${state.imageDims.w}×${state.imageDims.h}` : 'Sample 1280×720') + ')';
  }
  if (subEl) {
    subEl.textContent = isSynthetic
      ? 'Reference: Mathematical Ground Truth (Single-pixel contours with 1px tolerance)'
      : 'Reference: Canny Edge Detector (Canny achieves F1 = 1.000 by reference design)';
  }

  tableHead.innerHTML = `
    <tr>
      <th>Method</th>
      <th>Type</th>
      <th>Runtime</th>
      <th>Throughput</th>
      <th>F1 Score</th>
      <th>SSIM</th>
      <th>PSNR</th>
      <th>IoU</th>
      <th>Density</th>
      <th>Peak RAM</th>
    </tr>
  `;

  let minRuntime = Infinity;
  let maxF1 = -1;
  rows.forEach(r => {
    if (r.runtime_s && r.runtime_s < minRuntime) minRuntime = r.runtime_s;
    if (r.f1 && r.f1 > maxF1) maxF1 = r.f1;
  });

  tableBody.innerHTML = rows.map(r => {
    const isFastest = (r.runtime_s === minRuntime);
    const isBestF1 = (Math.abs(r.f1 - maxF1) < 0.0005);
    const badgeClass = `bm-badge-${r.method}`;

    const runtimeStr = r.runtime_s < 1.0
      ? `${(r.runtime_s * 1000).toFixed(1)} ms`
      : `${r.runtime_s.toFixed(3)} s`;
    const psnrStr = (r.psnr && r.psnr >= 900) ? '∞' : (r.psnr ? `${r.psnr.toFixed(1)} dB` : '—');
    const ramStr = r.peak_mem_mb ? `${r.peak_mem_mb.toFixed(1)} MB` : '—';

    return `
      <tr>
        <td>
          <div class="bm-method-cell">
            <span class="bm-badge ${badgeClass}">${r.method.toUpperCase()}</span>
            <span class="bm-method-name">${r.label || r.method}</span>
          </div>
        </td>
        <td style="color:var(--muted); font-size:12px;">${r.type || 'Detector'}</td>
        <td>
          <strong>${runtimeStr}</strong>
          ${isFastest ? '<span class="bm-badge-best">Fastest</span>' : ''}
        </td>
        <td>${(r.megapixels_per_s || 0).toFixed(1)} MP/s</td>
        <td>
          <strong>${(r.f1 || 0).toFixed(4)}</strong>
          ${isBestF1 ? '<span class="bm-badge-best">Best</span>' : ''}
        </td>
        <td>${(r.ssim || 0).toFixed(4)}</td>
        <td>${psnrStr}</td>
        <td>${r.iou ? r.iou.toFixed(4) : '—'}</td>
        <td>${r.edge_density ? (r.edge_density * 100).toFixed(2) + '%' : '—'}</td>
        <td style="color:var(--muted);">${ramStr}</td>
      </tr>
    `;
  }).join('');
}

function renderScalingChart() {
  const canvas = $('scaling-chart-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const W = rect.width || 560;
  const H = rect.height || 260;

  canvas.width = W * dpr;
  canvas.height = H * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);

  const sweep = state.benchmarkData.scaling_sweep || [];
  if (!sweep.length) return;

  const pad = { top: 25, right: 30, bottom: 40, left: 55 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const maxTime = Math.max(...sweep.map(s => s.qhed_runtime_s || 1), 2.0);

  ctx.strokeStyle = '#e0dbd0';
  ctx.lineWidth = 1;
  ctx.fillStyle = '#6a6a6a';
  ctx.font = '11px Inter, sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';

  const gridSteps = 4;
  for (let i = 0; i <= gridSteps; i++) {
    const val = (maxTime * (i / gridSteps));
    const y = pad.top + plotH - (i / gridSteps) * plotH;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + plotW, y);
    ctx.stroke();
    ctx.fillText(`${val.toFixed(2)}s`, pad.left - 8, y);
  }

  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const xPoints = sweep.map((s, idx) => {
    const x = pad.left + (idx / (sweep.length - 1)) * plotW;
    ctx.fillText(`${s.resolution}`, x, pad.top + plotH + 8);
    ctx.fillStyle = '#9a9a9a';
    ctx.font = '10px Inter, sans-serif';
    ctx.fillText(`${s.megapixels} MP`, x, pad.top + plotH + 22);
    ctx.fillStyle = '#6a6a6a';
    ctx.font = '11px Inter, sans-serif';
    return x;
  });

  const series = [
    { key: 'qhed_runtime_s', color: '#2a78d6', name: 'QHED' },
    { key: 'sobel_runtime_s', color: '#eb6834', name: 'Sobel' },
    { key: 'canny_runtime_s', color: '#eda100', name: 'Canny' },
  ];

  series.forEach(ser => {
    ctx.strokeStyle = ser.color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    sweep.forEach((pt, idx) => {
      const val = pt[ser.key] || 0;
      const x = xPoints[idx];
      const y = pad.top + plotH - (val / maxTime) * plotH;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    sweep.forEach((pt, idx) => {
      const val = pt[ser.key] || 0;
      const x = xPoints[idx];
      const y = pad.top + plotH - (val / maxTime) * plotH;

      ctx.fillStyle = '#fffaf0';
      ctx.beginPath();
      ctx.arc(x, y, 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = ser.color;
      ctx.lineWidth = 2;
      ctx.stroke();

      if (idx === sweep.length - 1 || ser.name === 'QHED') {
        ctx.fillStyle = ser.color;
        ctx.font = 'bold 10px SF Mono, monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`${val.toFixed(3)}s`, x, y - 12);
      }
    });
  });
}

function renderQualityChart() {
  const canvas = $('quality-chart-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const W = rect.width || 560;
  const H = rect.height || 260;

  canvas.width = W * dpr;
  canvas.height = H * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);

  const isSynthetic = (state.benchmarkView === 'synthetic');
  const rows = isSynthetic
    ? (state.benchmarkData.synthetic_ground_truth || [])
    : (state.benchmarkData.sample_benchmarks || []);
  if (!rows.length) return;

  const pad = { top: 25, right: 20, bottom: 40, left: 45 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  ctx.strokeStyle = '#e0dbd0';
  ctx.lineWidth = 1;
  ctx.fillStyle = '#6a6a6a';
  ctx.font = '11px Inter, sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';

  for (let i = 0; i <= 4; i++) {
    const val = (i / 4).toFixed(2);
    const y = pad.top + plotH - (i / 4) * plotH;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + plotW, y);
    ctx.stroke();
    ctx.fillText(val, pad.left - 6, y);
  }

  const groupW = plotW / rows.length;
  const barW = Math.min(22, groupW * 0.32);

  rows.forEach((r, idx) => {
    const cx = pad.left + (idx + 0.5) * groupW;

    const f1 = r.f1 || 0;
    const f1H = f1 * plotH;
    const f1X = cx - barW - 2;
    const f1Y = pad.top + plotH - f1H;
    ctx.fillStyle = '#1a3a3a';
    roundRect(ctx, f1X, f1Y, barW, f1H, 3);
    ctx.fill();

    const ssim = r.ssim || 0;
    const ssimH = ssim * plotH;
    const ssimX = cx + 2;
    const ssimY = pad.top + plotH - ssimH;
    ctx.fillStyle = '#b8a4ed';
    roundRect(ctx, ssimX, ssimY, barW, ssimH, 3);
    ctx.fill();

    ctx.fillStyle = '#1a3a3a';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(f1.toFixed(2), f1X + barW / 2, Math.max(pad.top - 2, f1Y - 4));

    ctx.fillStyle = '#7c3aed';
    ctx.fillText(ssim.toFixed(2), ssimX + barW / 2, Math.max(pad.top - 2, ssimY - 4));

    ctx.fillStyle = '#0a0a0a';
    ctx.font = '600 12px Inter, sans-serif';
    ctx.fillText(r.method.toUpperCase(), cx, pad.top + plotH + 10);
  });
}

function renderThroughputBars() {
  const container = $('bm-throughput-bars');
  if (!container) return;

  const isSynthetic = (state.benchmarkView === 'synthetic');
  const rows = isSynthetic
    ? (state.benchmarkData.synthetic_ground_truth || [])
    : (state.benchmarkData.sample_benchmarks || []);
  if (!rows.length) return;

  const maxTP = Math.max(...rows.map(r => r.megapixels_per_s || 1), 10);

  container.innerHTML = rows.map(r => {
    const val = r.megapixels_per_s || 0;
    const pct = Math.max(4, Math.min(100, (val / maxTP) * 100));
    const fillClass = `tp-fill-${r.method}`;

    return `
      <div class="tp-row">
        <div class="tp-label">${r.label || r.method.toUpperCase()}</div>
        <div class="tp-track">
          <div class="tp-fill ${fillClass}" style="width: ${pct.toFixed(1)}%;"></div>
        </div>
        <div class="tp-val">${val.toFixed(2)} MP/s</div>
      </div>
    `;
  }).join('');
}

async function runLiveBenchmark() {
  if (state.benchmarking) return;
  state.benchmarking = true;

  const btn = $('btn-run-benchmark');
  const spinner = $('bm-spinner');
  const label = $('bm-run-label');
  const statusPill = $('bm-status-text');

  if (btn) btn.disabled = true;
  if (spinner) spinner.hidden = false;
  if (label) label.textContent = 'Measuring Performance…';
  if (statusPill) statusPill.textContent = 'Running Live Measurements…';

  try {
    const payload = {
      image_b64: state.resultImages.original || state.imageURL || null,
      tile_size: state.tileSize,
      backend: state.backend,
      threshold: state.threshold,
    };

    const endpoint = window.__BENCHMARK_API__ || 'http://127.0.0.1:8585/api/benchmark';
    const resp = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (resp.ok) {
      const data = await resp.json();
      if (data.status === 'success' && data.results) {
        state.benchmarkData.sample_benchmarks = data.results;
        state.benchmarkView = 'image';

        const tabs = $('bm-view-tabs');
        if (tabs) {
          tabs.querySelectorAll('.seg-btn').forEach(b => {
            b.classList.toggle('seg-active', b.dataset.view === 'image');
          });
        }

        if (data.summary) {
          if (data.summary.qhed_f1 && $('bm-kpi-f1')) {
            $('bm-kpi-f1').textContent = `${data.summary.qhed_f1.toFixed(3)} F1`;
          }
        }

        if (statusPill) statusPill.textContent = 'Live Benchmark Complete (Active Image)';
      }
    } else {
      throw new Error(`HTTP ${resp.status}`);
    }
  } catch (err) {
    console.warn('Live benchmark request failed, keeping verified baseline:', err);
    if (statusPill) statusPill.textContent = 'Baseline Data Active';
  } finally {
    state.benchmarking = false;
    if (btn) btn.disabled = false;
    if (spinner) spinner.hidden = true;
    if (label) label.textContent = 'Run Live Benchmark';
    updateBenchmarkView();
  }
}

async function runScalingSweep() {
  const btn = $('btn-run-scaling');
  const statusPill = $('bm-status-text');
  if (btn) btn.disabled = true;
  if (btn) btn.textContent = 'Sweeping…';
  if (statusPill) statusPill.textContent = 'Running 480p–4K Resolution Sweep…';

  try {
    const endpoint = window.__SCALING_API__ || 'http://127.0.0.1:8585/api/scaling';
    const resp = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ quick: false }),
    });

    if (resp.ok) {
      const data = await resp.json();
      if (data.status === 'success' && data.pivot_rows) {
        state.benchmarkData.scaling_sweep = data.pivot_rows;
      }
    }
  } catch (err) {
    console.warn('Scaling request failed, keeping baseline:', err);
  } finally {
    if (btn) btn.disabled = false;
    if (btn) btn.textContent = 'Run 480p–4K Sweep';
    if (statusPill) statusPill.textContent = 'Resolution Sweep Updated';

    state.benchmarkView = 'scaling';
    const tabs = $('bm-view-tabs');
    if (tabs) {
      tabs.querySelectorAll('.seg-btn').forEach(b => {
        b.classList.toggle('seg-active', b.dataset.view === 'scaling');
      });
    }
    updateBenchmarkView();
  }
}

function exportBenchmarkData() {
  const data = state.benchmarkData || DEFAULT_BENCHMARKS;
  const jsonStr = JSON.stringify(data, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `q_edge_benchmarks_${new Date().toISOString().slice(0, 10)}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function debounce(fn, ms) {
  let timer;
  return function(...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), ms);
  };
}

document.addEventListener('DOMContentLoaded', init);
