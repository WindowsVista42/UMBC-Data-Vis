import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── Constants ────────────────────────────────────────────────────────────────
const SIDEBAR_W    = 270;
const DRAG_THRESH  = 5;          // px — below this is a click, above is a drag
const DBL_CLICK_MS = 350;        // ms — max gap for double-click

const PALETTE = [
  '#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f',
  '#edc948','#b07aa1','#ff9da7','#9c755f','#bab0ac',
  '#17becf','#bcbd22','#7f7f7f','#d62728','#2ca02c',
  '#1f77b4','#ff7f0e','#aec7e8','#ffbb78','#98df8a',
];
const NOISE_HEX = '#737373';  // cluster -1

// ── Vertex shader ────────────────────────────────────────────────────────────
const VERT = /* glsl */`
  attribute vec3  aColor;
  attribute float aAlpha;
  varying   vec3  vColor;
  varying   float vAlpha;
  uniform   float uPointSize;

  void main() {
    vec4  mvPos   = modelViewMatrix * vec4(position, 1.0);
    gl_Position   = projectionMatrix * mvPos;
    float depth   = max(-mvPos.z, 0.001);
    gl_PointSize  = (uPointSize * 5.0) / depth;
    vColor = aColor;
    vAlpha = aAlpha;
  }
`;

// ── Fragment shader ──────────────────────────────────────────────────────────
const FRAG = /* glsl */`
  varying vec3  vColor;
  varying float vAlpha;
  uniform float uOutline;

  void main() {
    if (vAlpha < 0.5) discard;
    vec2  uv   = gl_PointCoord - vec2(0.5);
    float dist = length(uv);
    if (dist > 0.5) discard;
    vec3 col = vColor;
    if (uOutline > 0.5 && dist > 0.38) {
      float t = smoothstep(0.38, 0.48, dist);
      col = mix(col, vec3(0.0), t * 0.85);
    }
    gl_FragColor = vec4(col, 1.0);
  }
`;

// ── Module state ─────────────────────────────────────────────────────────────
let N = 0, loadedCount = 0;
let points = [];          // full point objects [{id,x,y,z,cluster,title,abstract,doi}]
let metaData = null;      // parsed meta.json

// Pre-allocated GPU buffers (sized to N after meta.json loads)
let posArr, colArr, alphaArr, baseR, baseG, baseB;
let clusterArr;           // Int32Array — cluster id per point

// Three.js objects
let renderer, scene, camera, controls;
let geometry, pointsMesh, posAttr, colAttr, alphaAttr;
let uniforms;
const raycaster = new THREE.Raycaster();
const mouse     = new THREE.Vector2();

// Interaction state
let lockedIdx      = -1;   // index of locked tooltip point (-1 = none)
let pointerIsDown  = false;
let mouseDownX     = 0, mouseDownY = 0;
let lastClickTime  = 0, lastClickCluster = null;
let camAnim        = null; // animation callback or null

// Sidebar / visibility state
let hiddenClusters  = new Set();   // Set<number>
let highlightedSet  = null;        // Set<number> | null
let clusterDisplayData = [];       // [{cid, name, count, colorHex, unclustered}]
let sortMode        = 'default';
let searchQuery     = '';
let lastSBClickIdx  = -1;          // last clicked row index (for shift-select)
let selectedClusters = new Set();  // Set<number> — sidebar row selection

// ── Color helpers ─────────────────────────────────────────────────────────────
function hexToRgbNorm(hex) {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  return [r, g, b];
}

function clusterColorRgb(cid) {
  if (cid === -1) return hexToRgbNorm(NOISE_HEX);
  return hexToRgbNorm(PALETTE[((cid % PALETTE.length) + PALETTE.length) % PALETTE.length]);
}

function clusterColorHex(cid) {
  if (cid === -1) return NOISE_HEX;
  return PALETTE[((cid % PALETTE.length) + PALETTE.length) % PALETTE.length];
}

// ── Buffer allocation ─────────────────────────────────────────────────────────
function allocateBuffers(n) {
  posArr   = new Float32Array(n * 3);
  colArr   = new Float32Array(n * 3);
  alphaArr = new Float32Array(n);     // 0 = transparent (discard in shader)
  baseR    = new Float32Array(n);
  baseG    = new Float32Array(n);
  baseB    = new Float32Array(n);
  clusterArr = new Int32Array(n);
}

// ── Scene initialisation ──────────────────────────────────────────────────────
function initScene() {
  const canvas = document.getElementById('three-canvas');
  const w = window.innerWidth - SIDEBAR_W;
  const h = window.innerHeight;

  renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(w, h);

  scene  = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(60, w / h, 0.001, 1000);
  camera.position.set(0.8, 0.6, 2.0);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping  = true;
  controls.dampingFactor  = 0.08;
  controls.minDistance    = 0.05;
  controls.maxDistance    = 50;

  // Geometry
  geometry = new THREE.BufferGeometry();
  // Prevent Three.js from frustum-culling the cloud before bounds are computed
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 1000);

  posAttr   = new THREE.BufferAttribute(posArr,   3);
  colAttr   = new THREE.BufferAttribute(colArr,   3);
  alphaAttr = new THREE.BufferAttribute(alphaArr, 1);

  posAttr.usage   = THREE.DynamicDrawUsage;
  colAttr.usage   = THREE.DynamicDrawUsage;
  alphaAttr.usage = THREE.DynamicDrawUsage;

  geometry.setAttribute('position', posAttr);
  geometry.setAttribute('aColor',   colAttr);
  geometry.setAttribute('aAlpha',   alphaAttr);
  geometry.setDrawRange(0, 0);

  uniforms = {
    uPointSize: { value: 1.0 },
    uOutline:   { value: 1.0 },
  };

  const mat = new THREE.ShaderMaterial({
    vertexShader:   VERT,
    fragmentShader: FRAG,
    uniforms,
    depthTest:  true,
    depthWrite: true,
  });

  pointsMesh = new THREE.Points(geometry, mat);
  scene.add(pointsMesh);

  // ── Controls UI wiring ──
  document.getElementById('fc-reset').addEventListener('click', () => setCameraPreset('reset'));
  document.getElementById('fc-top').addEventListener('click',   () => setCameraPreset('top'));
  document.getElementById('fc-front').addEventListener('click', () => setCameraPreset('front'));
  document.getElementById('fc-side').addEventListener('click',  () => setCameraPreset('side'));

  const sizeSlider = document.getElementById('fc-size');
  const sizeVal    = document.getElementById('fc-size-val');
  sizeSlider.addEventListener('input', () => {
    uniforms.uPointSize.value = parseFloat(sizeSlider.value);
    sizeVal.textContent = sizeSlider.value;
  });

  const outlineBtn = document.getElementById('fc-outline');
  outlineBtn.addEventListener('click', () => {
    const on = outlineBtn.classList.toggle('active');
    uniforms.uOutline.value = on ? 1.0 : 0.0;
  });

  // ── Abstract toggle ──
  document.getElementById('ht-abstract-toggle').addEventListener('click', () => {
    const absEl = document.getElementById('ht-abstract');
    const togEl = document.getElementById('ht-abstract-toggle');
    const open  = absEl.style.display === 'block';
    if (open) {
      absEl.style.display = 'none';
      togEl.innerHTML = 'Abstract &#9660;';
    } else {
      absEl.style.display = 'block';
      togEl.innerHTML = 'Abstract &#9650;';
    }
  });

  // ── Pointer events ──
  window.addEventListener('resize', onResize);
  renderer.domElement.addEventListener('pointerdown',  onPointerDown);
  renderer.domElement.addEventListener('pointermove',  onPointerMove);
  renderer.domElement.addEventListener('pointerup',    onPointerUp);
  renderer.domElement.addEventListener('pointerleave', onPointerLeave);

  animate();
}

// ── Decompression helpers ─────────────────────────────────────────────────────

/** Decompress a gzip ArrayBuffer → ArrayBuffer. */
async function decompressBuffer(ab) {
  const stream = new Blob([ab]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).arrayBuffer();
}

/** Decompress a gzip ArrayBuffer → parsed JSON value. */
async function decompressJson(ab) {
  const stream = new Blob([ab]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).json();
}

// ── Chunk ingestion ───────────────────────────────────────────────────────────

/**
 * Ingest one chunk's worth of points.
 *
 * @param {ArrayBuffer} binRaw  — decompressed binary: float32[N×3] positions
 *                                followed by int32[N] cluster IDs (little-endian)
 * @param {Array}       textData — [{id, title, abstract, doi}, …] length N
 */
function ingestChunk(binRaw, textData) {
  const n    = textData.length;
  const prev = loadedCount;

  // Float32Array view of the first N*12 bytes (xyz per point)
  const srcPos = new Float32Array(binRaw, 0, n * 3);
  // Int32Array view of the next N*4 bytes (cluster per point)
  const srcClu = new Int32Array(binRaw, n * 12, n);

  for (let j = 0; j < n; j++) {
    const i  = points.length;
    const i3 = i * 3;
    points.push(textData[j]);

    posArr[i3]   = srcPos[j * 3];
    posArr[i3+1] = srcPos[j * 3 + 1];
    posArr[i3+2] = srcPos[j * 3 + 2];

    const cid = srcClu[j];
    const [r, g, b] = clusterColorRgb(cid);
    colArr[i3]   = baseR[i] = r;
    colArr[i3+1] = baseG[i] = g;
    colArr[i3+2] = baseB[i] = b;
    clusterArr[i] = cid;
    alphaArr[i]   = 1.0;
  }
  loadedCount = points.length;

  // Only re-upload the newly added portion
  const newCount = loadedCount - prev;
  posAttr.addUpdateRange(prev * 3, newCount * 3);
  colAttr.addUpdateRange(prev * 3, newCount * 3);
  alphaAttr.addUpdateRange(prev, newCount);

  posAttr.needsUpdate   = true;
  colAttr.needsUpdate   = true;
  alphaAttr.needsUpdate = true;
  geometry.setDrawRange(0, loadedCount);
}

// ── Color management ──────────────────────────────────────────────────────────
function applyColors() {
  for (let i = 0; i < loadedCount; i++) {
    const cid = clusterArr[i];
    const i3  = i * 3;
    if (hiddenClusters.has(cid)) {
      alphaArr[i] = 0.0;
      continue;
    }
    alphaArr[i] = 1.0;
    if (highlightedSet !== null && !highlightedSet.has(cid)) {
      // Dim non-highlighted points
      colArr[i3]   = baseR[i] * 0.12 + 0.03;
      colArr[i3+1] = baseG[i] * 0.12 + 0.03;
      colArr[i3+2] = baseB[i] * 0.12 + 0.03;
    } else {
      colArr[i3]   = baseR[i];
      colArr[i3+1] = baseG[i];
      colArr[i3+2] = baseB[i];
    }
  }
  colAttr.clearUpdateRanges();
  alphaAttr.clearUpdateRanges();
  colAttr.needsUpdate   = true;
  alphaAttr.needsUpdate = true;
}

function setHighlight(set) {
  highlightedSet = set;
  applyColors();
  renderSidebarList();
  updateStats();
}

// ── Progress bar ──────────────────────────────────────────────────────────────
function showProgress(pct, label) {
  document.getElementById('progress-wrap').style.display  = 'block';
  document.getElementById('progress-label').style.display = 'block';
  document.getElementById('progress-bar').style.width     = `${pct}%`;
  document.getElementById('progress-label').textContent   = label;
}

function hideProgress() {
  document.getElementById('progress-wrap').style.display  = 'none';
  document.getElementById('progress-label').style.display = 'none';
}

// ── Tooltip ───────────────────────────────────────────────────────────────────
function worldToScreen(x, y, z) {
  const v = new THREE.Vector3(x, y, z).project(camera);
  const w = window.innerWidth - SIDEBAR_W;
  const h = window.innerHeight;
  return { x: (v.x + 1) / 2 * w, y: (-v.y + 1) / 2 * h };
}

function positionTooltip(idx) {
  const tip = document.getElementById('hover-tip');
  if (!tip || tip.style.display === 'none') return;
  const i3 = idx * 3;
  const sp = worldToScreen(posArr[i3], posArr[i3+1], posArr[i3+2]);

  const tw      = tip.offsetWidth  || 280;
  const th      = tip.offsetHeight || 80;
  const canvasW = window.innerWidth - SIDEBAR_W;
  const margin  = 16;

  let left, arrowSide;
  if (sp.x + margin + tw + 20 <= canvasW) {
    left      = sp.x + margin;
    arrowSide = 'left';
  } else {
    left      = sp.x - margin - tw;
    arrowSide = 'right';
  }
  left = Math.max(8, Math.min(left, canvasW - tw - 8));

  let top = sp.y - th / 2;
  top = Math.max(8, Math.min(top, window.innerHeight - th - 8));

  tip.style.left = `${Math.round(left)}px`;
  tip.style.top  = `${Math.round(top)}px`;

  const arrow   = document.getElementById('hover-tip-arrow');
  const arrowY  = Math.max(8, Math.min(sp.y - top - 7, th - 20));
  arrow.style.top = `${Math.round(arrowY)}px`;
  arrow.className = `arrow-${arrowSide}`;
  if (arrowSide === 'left') {
    arrow.style.left = '-7px'; arrow.style.right = 'auto';
  } else {
    arrow.style.right = '-7px'; arrow.style.left = 'auto';
  }
}

function showTooltip(idx, locked) {
  if (idx < 0 || idx >= loadedCount) return;
  const pt  = points[idx];
  const tip = document.getElementById('hover-tip');

  tip.querySelector('.ht-title').textContent = pt.title || pt.id;

  const linkEl = tip.querySelector('.ht-link');
  linkEl.href  = pt.doi || `https://arxiv.org/abs/${pt.id}`;

  const cid   = clusterArr[idx];
  const name  = metaData?.cluster_names?.[String(cid)] ?? `Cluster ${cid}`;
  const badge = tip.querySelector('.ht-cluster-badge');
  badge.textContent  = name;
  badge.style.color  = clusterColorHex(cid);

  if (locked) {
    document.getElementById('ht-abstract').textContent = pt.abstract || '(no abstract)';
    tip.classList.add('locked');
  } else {
    // Reset abstract state for hover mode
    document.getElementById('ht-abstract').style.display = 'none';
    document.getElementById('ht-abstract-toggle').innerHTML = 'Abstract &#9660;';
    tip.classList.remove('locked');
  }

  tip.style.display = 'block';
  positionTooltip(idx);
}

function hideTooltip() {
  const tip = document.getElementById('hover-tip');
  tip.style.display = 'none';
  tip.classList.remove('locked');
  document.getElementById('ht-abstract').style.display = 'none';
  document.getElementById('ht-abstract-toggle').innerHTML = 'Abstract &#9660;';
  lockedIdx = -1;
}

// ── Raycasting ────────────────────────────────────────────────────────────────
function raycastBest(e) {
  const rc    = renderer.domElement.getBoundingClientRect();
  mouse.x     =  ((e.clientX - rc.left) / rc.width)  * 2 - 1;
  mouse.y     = -((e.clientY - rc.top)  / rc.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);

  const dist = camera.position.distanceTo(controls.target);
  raycaster.params.Points = { threshold: Math.max(0.005, dist * 0.015) };

  const hits = raycaster.intersectObject(pointsMesh);
  for (const hit of hits) {
    const idx = hit.index;
    if (idx < loadedCount && alphaArr[idx] > 0.5) return idx;
  }
  return -1;
}

// ── Pointer events ────────────────────────────────────────────────────────────
function onPointerDown(e) {
  pointerIsDown = true;
  mouseDownX    = e.clientX;
  mouseDownY    = e.clientY;
}

function onPointerMove(e) {
  if (lockedIdx !== -1) return;  // locked tooltip: don't update hover
  if (pointerIsDown) {
    const d = Math.hypot(e.clientX - mouseDownX, e.clientY - mouseDownY);
    if (d > DRAG_THRESH) {
      // User is dragging — hide hover (but not if locked)
      const tip = document.getElementById('hover-tip');
      if (!tip.classList.contains('locked')) tip.style.display = 'none';
      return;
    }
  }
  const idx = raycastBest(e);
  if (idx >= 0) {
    showTooltip(idx, false);
  } else {
    const tip = document.getElementById('hover-tip');
    if (!tip.classList.contains('locked')) tip.style.display = 'none';
  }
}

function onPointerUp(e) {
  pointerIsDown = false;
  const d = Math.hypot(e.clientX - mouseDownX, e.clientY - mouseDownY);
  if (d > DRAG_THRESH) return;  // was a drag — ignore click

  const idx = raycastBest(e);
  const now = Date.now();

  // Double-click detection
  if (now - lastClickTime < DBL_CLICK_MS && lastClickTime > 0) {
    const prevTime    = lastClickTime;
    lastClickTime     = 0;
    lastClickCluster  = null;
    void prevTime;

    hideTooltip();
    if (idx >= 0) {
      const cid    = clusterArr[idx];
      const already = highlightedSet?.size === 1 && highlightedSet.has(cid);
      if (already) {
        setHighlight(null);
        selectedClusters.clear();
      } else {
        setHighlight(new Set([cid]));
        selectedClusters = new Set([cid]);
      }
    } else {
      setHighlight(null);
      selectedClusters.clear();
    }
    renderSidebarList();
    return;
  }

  lastClickTime    = now;
  lastClickCluster = idx >= 0 ? clusterArr[idx] : null;

  if (idx >= 0) {
    lockedIdx = idx;
    showTooltip(idx, true);
  } else {
    hideTooltip();
  }
}

function onPointerLeave() {
  pointerIsDown = false;
  if (lockedIdx < 0) {
    const tip = document.getElementById('hover-tip');
    if (!tip.classList.contains('locked')) tip.style.display = 'none';
  }
}

// ── Camera presets ────────────────────────────────────────────────────────────
function setCameraPreset(preset) {
  const target = new THREE.Vector3(0, 0, 0);
  let pos;
  switch (preset) {
    case 'top':   pos = new THREE.Vector3(0,   2.2, 0.001); break;
    case 'front': pos = new THREE.Vector3(0,   0,   2.5);   break;
    case 'side':  pos = new THREE.Vector3(2.5, 0,   0);     break;
    default:      pos = new THREE.Vector3(0.8, 0.6, 2.0);   break; // reset
  }
  animateCameraTo(pos, target);
}

function animateCameraTo(targetPos, targetLookAt, duration = 650) {
  const startPos    = camera.position.clone();
  const startTarget = controls.target.clone();
  const startTime   = performance.now();

  camAnim = (now) => {
    const raw = Math.min((now - startTime) / duration, 1);
    const t   = raw < 0.5 ? 2 * raw * raw : -1 + (4 - 2 * raw) * raw; // ease-in-out
    camera.position.lerpVectors(startPos, targetPos, t);
    controls.target.lerpVectors(startTarget, targetLookAt, t);
    controls.update();
    return raw < 1; // return false when done
  };
}

// ── Resize ────────────────────────────────────────────────────────────────────
function onResize() {
  const w = window.innerWidth - SIDEBAR_W;
  const h = window.innerHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  if (lockedIdx >= 0) positionTooltip(lockedIdx);
}

// ── Render loop ───────────────────────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  if (camAnim) {
    if (!camAnim(performance.now())) camAnim = null;
  } else {
    controls.update();
  }
  renderer.render(scene, camera);
  if (lockedIdx >= 0) positionTooltip(lockedIdx);
}

// ── Sidebar ───────────────────────────────────────────────────────────────────
function initSidebar() {
  // Count points per cluster
  const counts = {};
  for (let i = 0; i < loadedCount; i++) {
    const cid = clusterArr[i];
    counts[cid] = (counts[cid] || 0) + 1;
  }

  clusterDisplayData = Object.entries(metaData.cluster_names).map(([cidStr, name]) => {
    const cid = parseInt(cidStr, 10);
    return {
      cid,
      name,
      count:       counts[cid] || 0,
      colorHex:    clusterColorHex(cid),
      unclustered: cid === -1,
    };
  }).sort((a, b) => {
    if (a.unclustered) return -1;
    if (b.unclustered) return 1;
    return a.cid - b.cid;
  });

  // Hide noise cluster by default (but only if it has points)
  if (counts[-1] > 0) {
    hiddenClusters.add(-1);
    applyColors();
  }

  // Sidebar button events
  document.getElementById('sb-show-all').addEventListener('click', () => {
    hiddenClusters.clear();
    applyColors();
    renderSidebarList();
    updateStats();
  });

  document.getElementById('sb-isolate').addEventListener('click', () => {
    if (!highlightedSet || highlightedSet.size === 0) return;
    hiddenClusters.clear();
    for (const d of clusterDisplayData) {
      if (!highlightedSet.has(d.cid)) hiddenClusters.add(d.cid);
    }
    applyColors();
    renderSidebarList();
    updateStats();
  });

  document.getElementById('sb-clear').addEventListener('click', () => {
    setHighlight(null);
    selectedClusters.clear();
    renderSidebarList();
  });

  // Sort buttons
  ['sort-default', 'sort-alpha', 'sort-size'].forEach(id => {
    document.getElementById(id).addEventListener('click', () => {
      document.querySelectorAll('.sort-btn').forEach(b => b.classList.remove('active'));
      document.getElementById(id).classList.add('active');
      sortMode = id.replace('sort-', '');
      renderSidebarList();
    });
  });

  // Search
  const searchEl = document.getElementById('sidebar-search');
  document.getElementById('sidebar-search-clear').addEventListener('click', () => {
    searchEl.value = '';
    searchQuery = '';
    setHighlight(null);
    selectedClusters.clear();
  });
  searchEl.addEventListener('input', () => {
    searchQuery = searchEl.value.toLowerCase();
    if (searchQuery) {
      const matching = new Set(
        clusterDisplayData
          .filter(d => d.name.toLowerCase().includes(searchQuery))
          .map(d => d.cid)
      );
      setHighlight(matching.size > 0 ? matching : null);
    } else {
      setHighlight(null);
      selectedClusters.clear();
    }
  });

  updateStats();
  renderSidebarList();
}

function getSortedClusters() {
  let data = [...clusterDisplayData];
  if (searchQuery) {
    data = data.filter(d => d.name.toLowerCase().includes(searchQuery));
  }
  if (sortMode === 'alpha') {
    data.sort((a, b) => {
      if (a.unclustered) return -1;
      if (b.unclustered) return 1;
      return a.name.localeCompare(b.name);
    });
  } else if (sortMode === 'size') {
    data.sort((a, b) => {
      if (a.unclustered) return -1;
      if (b.unclustered) return 1;
      return b.count - a.count;
    });
  }
  return data;
}

function renderSidebarList() {
  const list   = document.getElementById('sidebar-list');
  const data   = getSortedClusters();
  list.innerHTML = '';

  data.forEach((d, visIdx) => {
    const row = document.createElement('div');
    row.className = 'cluster-row';
    if (selectedClusters.has(d.cid)) row.classList.add('selected');

    const isHidden    = hiddenClusters.has(d.cid);
    const isHighlight = highlightedSet?.has(d.cid);

    // Eye toggle
    const eye = document.createElement('span');
    eye.className   = 'cluster-eye' + (isHidden ? ' hidden-eye' : '');
    eye.textContent = isHidden ? '○' : '●';
    eye.title       = isHidden ? 'Show cluster' : 'Hide cluster';

    // Color dot
    const dot = document.createElement('span');
    dot.className        = 'cluster-dot';
    dot.style.background = d.colorHex;

    // Name (with search highlight)
    const nameEl = document.createElement('span');
    nameEl.className = 'cluster-name' + (isHighlight ? ' highlighted' : '');
    if (searchQuery) {
      const lo = d.name.toLowerCase().indexOf(searchQuery);
      if (lo >= 0) {
        nameEl.innerHTML =
          esc(d.name.slice(0, lo)) +
          `<span class="match-hl">${esc(d.name.slice(lo, lo + searchQuery.length))}</span>` +
          esc(d.name.slice(lo + searchQuery.length));
      } else {
        nameEl.textContent = d.name;
      }
    } else {
      nameEl.textContent = d.name;
    }

    // Count
    const countEl = document.createElement('span');
    countEl.className   = 'cluster-count';
    countEl.textContent = d.count.toLocaleString();

    row.appendChild(eye);
    row.appendChild(dot);
    row.appendChild(nameEl);
    row.appendChild(countEl);

    // Eye click — toggle visibility
    eye.addEventListener('click', e => {
      e.stopPropagation();
      if (hiddenClusters.has(d.cid)) {
        hiddenClusters.delete(d.cid);
      } else {
        hiddenClusters.add(d.cid);
        // Unlock tooltip if it was showing a point in this cluster
        if (lockedIdx >= 0 && clusterArr[lockedIdx] === d.cid) hideTooltip();
      }
      applyColors();
      renderSidebarList();
      updateStats();
    });

    // Row click — highlight cluster(s)
    row.addEventListener('click', e => {
      const visData = getSortedClusters();
      if (e.shiftKey && lastSBClickIdx >= 0) {
        // Range select
        const lo = Math.min(lastSBClickIdx, visIdx);
        const hi = Math.max(lastSBClickIdx, visIdx);
        for (let i = lo; i <= hi; i++) selectedClusters.add(visData[i].cid);
      } else if (e.ctrlKey || e.metaKey) {
        // Toggle individual
        if (selectedClusters.has(d.cid)) selectedClusters.delete(d.cid);
        else                              selectedClusters.add(d.cid);
        lastSBClickIdx = visIdx;
      } else {
        // Single select
        selectedClusters.clear();
        selectedClusters.add(d.cid);
        lastSBClickIdx = visIdx;
      }
      setHighlight(selectedClusters.size > 0 ? new Set(selectedClusters) : null);
    });

    list.appendChild(row);
  });
}

function updateStats() {
  const total     = loadedCount;
  const nClusters = clusterDisplayData.filter(d => !d.unclustered).length;
  const visible   = clusterDisplayData
    .filter(d => !hiddenClusters.has(d.cid))
    .reduce((s, d) => s + d.count, 0);

  document.getElementById('sidebar-stats').textContent =
    `${total.toLocaleString()} points · ${nClusters} clusters · ${visible.toLocaleString()} visible`;
}

function esc(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ── Boot ──────────────────────────────────────────────────────────────────────
async function boot() {
  showProgress(0, 'Loading…');

  let meta;
  try {
    const resp = await fetch('meta.json');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    meta = await resp.json();
  } catch (e) {
    document.getElementById('progress-label').textContent = `Error: could not load meta.json (${e.message})`;
    document.getElementById('progress-label').style.display = 'block';
    return;
  }

  metaData = meta;
  N        = meta.total;

  allocateBuffers(N);
  initScene();

  const nChunks = meta.chunks;
  const pad     = (i) => String(i).padStart(6, '0');

  for (let i = 0; i < nChunks; i++) {
    const binName  = `chunks/chunk_${pad(i)}.bin.gz`;
    const textName = `chunks/chunk_${pad(i)}.json.gz`;
    try {
      const [binResp, textResp] = await Promise.all([fetch(binName), fetch(textName)]);
      if (!binResp.ok)  throw new Error(`HTTP ${binResp.status} for ${binName}`);
      if (!textResp.ok) throw new Error(`HTTP ${textResp.status} for ${textName}`);

      const [binRaw, textData] = await Promise.all([
        binResp.arrayBuffer().then(decompressBuffer),
        textResp.arrayBuffer().then(decompressJson),
      ]);
      ingestChunk(binRaw, textData);
    } catch (e) {
      console.error(`Failed to load chunk ${i}: ${e.message}`);
      continue;
    }
    const pct       = Math.round(((i + 1) / nChunks) * 100);
    const remaining = N - loadedCount;
    showProgress(pct, `Loading… ${loadedCount.toLocaleString()} / ${N.toLocaleString()} points  (${remaining.toLocaleString()} remaining)`);
  }

  hideProgress();
  initSidebar();
}

boot();
