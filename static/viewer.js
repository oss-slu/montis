import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';

const DEFAULT_MAX_SLOPE = 55;
const DEFAULT_RISK_POW  = 2.0;

// ---------- DOM ----------
const canvas     = document.getElementById('canvas');
const overlayEl  = document.getElementById('overlay');
const btnFast    = document.getElementById('btnFast');
const btnSafe    = document.getElementById('btnSafe');
const btnClear   = document.getElementById('btnClear');
const startText  = document.getElementById('startText');
const goalText   = document.getElementById('goalText');
const routeText  = document.getElementById('routeText');
const hud        = document.getElementById('hud');
const panel      = document.getElementById('panel');
const btnHide    = document.getElementById('btnHide');
const btnToggle  = document.getElementById('toggle');

// Auth elements
const usernameEl = document.getElementById('username');
const passwordEl = document.getElementById('password');
const btnLogin   = document.getElementById('btnLogin');
const btnSignup  = document.getElementById('btnSignup');
const btnLogout  = document.getElementById('btnLogout');
const authStatus = document.getElementById('authStatus');

let currentUser = null;

function updateAuthUI() {
  if (currentUser) {
    btnLogin.style.display = 'none';
    btnSignup.style.display = 'none';
    btnLogout.style.display = '';
    usernameEl.style.display = 'none';
    passwordEl.style.display = 'none';
    authStatus.textContent = `Logged in as ${currentUser.username}`;
  } else {
    btnLogin.style.display = '';
    btnSignup.style.display = '';
    btnLogout.style.display = 'none';
    usernameEl.style.display = '';
    passwordEl.style.display = '';
    authStatus.textContent = '';
  }
}

async function loginOrSignup(endpoint) {
  const username = usernameEl.value.trim();
  const password = passwordEl.value;
  if (!username || !password) { authStatus.textContent = 'Enter username and password.'; return; }
  authStatus.textContent = 'Processing...';
  try {
    const r = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });
    const res = await r.json();
    if (res.error) { authStatus.textContent = res.error; return; }
    currentUser = { id: res.id, username: res.username };
    updateAuthUI();
    authStatus.textContent = `Logged in as ${currentUser.username}`;
  } catch (e) {
    authStatus.textContent = 'Auth error.';
  }
}

btnLogin.addEventListener('click', () => loginOrSignup('/api/login'));
btnSignup.addEventListener('click', () => loginOrSignup('/api/signup'));
btnLogout.addEventListener('click', async () => {
  await fetch('/api/logout', { method: 'POST' });
  currentUser = null;
  updateAuthUI();
  authStatus.textContent = '';
});
updateAuthUI();

// Create UI elements
const toolsDiv = document.createElement('div');
toolsDiv.className = 'row';
toolsDiv.innerHTML = `
  <select id="tool">
    <option value="none" selected>Tool: None</option>
    <option value="probe">Tool: Probe</option>
    <option value="area">Tool: Area select</option>
    <option value="pin">Tool: Add hazard</option>
    <option value="segment">Tool: Segment paint</option>
    <option value="draw">Tool: Draw route</option>
  </select>
`;
panel.appendChild(toolsDiv);

const areaDiv = document.createElement('div');
areaDiv.className = 'row';
areaDiv.innerHTML = `
  <button id="btnAreaStats" class="ghost">Compute Area Stats</button>
  <button id="btnClearSel" class="ghost">Clear Selection</button>
  <span id="areaInfo" class="muted">0 selected</span>
`;
panel.appendChild(areaDiv);

const hazDiv = document.createElement('div');
hazDiv.className = 'row';
hazDiv.innerHTML = `<button id="btnReloadHaz" class="ghost">Reload Hazards</button>`;
panel.appendChild(hazDiv);

const hazSizeDiv = document.createElement('div');
hazSizeDiv.className = 'row';
hazSizeDiv.innerHTML = `
  <label style="min-width:auto">Marker size</label>
  <input id="hazSize" type="range" min="0.5" max="3" step="0.1" value="1.6">
`;
panel.appendChild(hazSizeDiv);

const classDiv = document.createElement('div');
classDiv.className = 'row';
classDiv.innerHTML = `
  <select id="clsType" title="Classify current polygon (Area select + Shift+Click)">
    <option value="crevasse">Classify as: Crevasse</option>
    <option value="glacier">Classify as: Glacier</option>
    <option value="snow">Classify as: Snow</option>
    <option value="rock">Classify as: Rock</option>
    <option value="ice">Classify as: Ice</option>
  </select>
  <button id="btnApplyClass"  class="ghost">Apply</button>
  <button id="btnShowClasses" class="ghost">Show</button>
  <button id="btnHideClasses" class="ghost">Hide</button>
`;
panel.appendChild(classDiv);

const customDiv = document.createElement('div');
customDiv.className = 'row';
customDiv.innerHTML = `
  <button id="btnCustomUndo"  class="ghost">Undo</button>
  <button id="btnCustomClear" class="ghost">Clear</button>
  <button id="btnCustomSave"  class="ghost">Save Custom Route</button>
`;
panel.appendChild(customDiv);

const customInfoDiv = document.createElement('div');
customInfoDiv.className = 'row';
customInfoDiv.innerHTML = `<span id="customInfo" class="muted">0 pts</span>`;
panel.appendChild(customInfoDiv);

const routesDiv = document.createElement('div');
routesDiv.className = 'row';
routesDiv.innerHTML = `
  <button id="btnSaveRoute" class="ghost">Save Route</button>
  <button id="btnShowRoutes" class="ghost">Show Saved</button>
`;
panel.appendChild(routesDiv);

const segDiv = document.createElement('div');
segDiv.className = 'row';
segDiv.innerHTML = `
  <button id="btnRunSeg" class="ghost">Run K-means Seg</button>
  <button id="btnToggleSeg" class="ghost">Toggle Seg</button>
`;
panel.appendChild(segDiv);

const areaInfo   = document.getElementById('areaInfo');
const customInfo = document.getElementById('customInfo');

// Ensure the overlay select has "heatmap"
(() => {
  if (overlayEl && !Array.from(overlayEl.options).some(o => o.value === 'heatmap')) {
    const opt = document.createElement('option');
    opt.value = 'heatmap';
    opt.textContent = 'Overlay: Route heatmap';
    overlayEl.appendChild(opt);
  }
})();

// ---------- THREE ----------
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x0b1020, 1);
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 1e7);
scene.add(camera);
scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dir = new THREE.DirectionalLight(0xffffff, 1.2); dir.position.set(300,300,600); scene.add(dir);

const controls = new OrbitControls(camera, renderer.domElement);
function renderOnce() {
  renderer.render(scene, camera);
}
controls.enableDamping = true; controls.dampingFactor = 0.12;
controls.addEventListener('change', renderOnce);

window.addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
  renderOnce();
});

// ---------- State ----------
let modelId    = null;
let object3D   = null;
let geom       = null;
let positions  = null, indices = null, normals = null, slope = null, count = 0;
let startIdx   = null, goalIdx = null;
let routeLine  = null;
let mouse = new THREE.Vector2();
let raycaster = new THREE.Raycaster();
let hazardGroup = new THREE.Group(); scene.add(hazardGroup);
let hazards = [];
let selection = new Set();
let selectionVis = null;
let customPath = [];
let customLine = null;
let customMarkers = new THREE.Group(); scene.add(customMarkers);
let segLabels = null;
let segMesh = null;
let segVisible = false;
let classFilter = null;
let modelBounds = null;
let hazardSizeScale = 1.0;
let ID_TO_CLASS = {};
let CLASS_COLOR = {};
let CLASS_TO_ID = {};

const toolEl = document.getElementById('tool');
let currentTool = toolEl.value;

function updateSelectionVis(){
  if (selectionVis){ scene.remove(selectionVis); selectionVis.geometry.dispose(); selectionVis.material.dispose(); selectionVis=null; }
  if (selection.size===0) { areaInfo.textContent = "0 selected"; renderOnce(); return; }
  const idxs = Array.from(selection);
  const pts = new Float32Array(idxs.length*3);
  for (let i=0;i<idxs.length;i++){ const j=idxs[i]; pts[i*3]=positions[j*3]; pts[i*3+1]=positions[j*3+1]; pts[i*3+2]=positions[j*3+2]; }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pts,3));
  const m = new THREE.PointsMaterial({ color: 0xffd400, size: 6, sizeAttenuation: true, depthTest:false });
  selectionVis = new THREE.Points(g, m); scene.add(selectionVis);
  areaInfo.textContent = `${selection.size} selected`;
  renderOnce();
}

function drawRoute(path){
  if (routeLine){ scene.remove(routeLine); routeLine.geometry.dispose(); routeLine.material.dispose(); routeLine=null; }
  if (!path || path.length<2) { routeText.textContent='—'; renderOnce(); return; }
  const pts = new Float32Array(path.length*3);
  for (let k=0;k<path.length;k++){
    const i = path[k];
    pts[k*3+0]=positions[i*3+0];
    pts[k*3+1]=positions[i*3+1];
    pts[k*3+2]=positions[i*3+2];
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pts,3));
  const m = new THREE.LineBasicMaterial({ color: 0x00b4ff });
  routeLine = new THREE.Line(g,m);
  scene.add(routeLine);
  renderOnce();
}

function markerRadius(){
  const b = modelBounds;
  if (!b) return 1;
  const spanX = b.max.x - b.min.x;
  const spanY = b.max.y - b.min.y;
  const spanZ = b.max.z - b.min.z;
  const span  = Math.max(spanX, spanY, spanZ);
  return Math.max(span * 0.012 * hazardSizeScale, span * 0.002);
}

function reloadHazards(){
  if (!modelId) return;
  fetch(`/api/hazards/${modelId}`).then(r=>r.json()).then(list=>{
    hazards = list;
    hazardGroup.clear();

    const r = markerRadius();
    const geomS = new THREE.SphereGeometry(r, 16, 16);

    for (const h of hazards){
      const color = (h.kind === 'rockfall' || h.kind === 'avalanche') ? 0xff6a00 : 0xff3b3b;
      const matS = new THREE.MeshBasicMaterial({ color, depthTest:false, transparent:true, opacity:0.95 });
      const s = new THREE.Mesh(geomS, matS);
      s.renderOrder = 999;
      s.position.set(h.x, h.y, h.z);
      s.userData = { id:h.id, kind:h.kind, severity:h.severity, note:h.note, x:h.x, y:h.y, z:h.z };
      hazardGroup.add(s);
    }
    renderOnce();
  });
}

function computeCustomStats(idxArr){
  if (!idxArr || idxArr.length < 2) return { lengthKm: 0, gainM: 0 };
  let length = 0, gain = 0;
  for (let k=1;k<idxArr.length;k++){
    const i = idxArr[k-1], j = idxArr[k];
    const dx = positions[j*3]   - positions[i*3];
    const dy = positions[j*3+1] - positions[i*3+1];
    const dz = positions[j*3+2] - positions[i*3+2];
    length += Math.hypot(dx,dy,dz);
    if (dz > 0) gain += dz;
  }
  return { lengthKm: length/1000, gainM: gain };
}

function updateCustomVis(){
  if (customLine){ scene.remove(customLine); customLine.geometry.dispose(); customLine.material.dispose(); customLine=null; }
  customMarkers.clear();

  const r = Math.max(markerRadius() * 0.6, 0.5);
  const gS = new THREE.SphereGeometry(r, 12, 12);
  const mS = new THREE.MeshBasicMaterial({ color: 0x00ffa6, depthTest:false, transparent:true, opacity:0.95 });
  for (const i of customPath){
    const s = new THREE.Mesh(gS, mS.clone());
    s.position.set(positions[i*3], positions[i*3+1], positions[i*3+2]);
    s.renderOrder = 998;
    customMarkers.add(s);
  }

  if (customPath.length >= 2){
    const pts = new Float32Array(customPath.length*3);
    for (let k=0;k<customPath.length;k++){
      const i = customPath[k];
      pts[k*3+0]=positions[i*3+0];
      pts[k*3+1]=positions[i*3+1];
      pts[k*3+2]=positions[i*3+2];
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pts,3));
    const m = new THREE.LineBasicMaterial({ color: 0x00ffa6, transparent:true, opacity:0.95 });
    customLine = new THREE.Line(g,m);
    customLine.renderOrder = 997;
    scene.add(customLine);
  }

  const stats = computeCustomStats(customPath);
  customInfo.textContent = `${customPath.length} pts · ${stats.lengthKm.toFixed(2)} km · +${stats.gainM.toFixed(0)} m`;
  renderOnce();
}

function clearCustom(){
  customPath = [];
  updateCustomVis();
}

async function ensureSegmentation(){
  if (!modelId) return false;
  try{
    const r = await fetch(`/api/segment/${modelId}`);
    if (r.ok) return true;
  }catch{}
  const r2 = await fetch(`/api/segment/${modelId}`, {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ k:1, features:["elev"] })
  });
  return r2.ok;
}

function loadSegLabels(redraw=false){
  if (!modelId) return;
  fetch(`/api/segment/${modelId}`).then(r=>r.json()).then(res=>{
    if (res.error){ alert(res.error); return; }
    segLabels = new Int32Array(res.labels);
    if (redraw) drawSegOverlay();
  });
}

function drawSegOverlay(){
  if (!segLabels) return;

  if (segMesh){ scene.remove(segMesh); segMesh.geometry?.dispose?.(); segMesh.material?.dispose?.(); segMesh=null; }

  const colors = new Float32Array(count*3);
  const colorForLabel = (lab) => {
    if (lab === -1 || lab === 0) return CLASS_COLOR['none']; // white for 'none' (unclassified)
    const cls = ID_TO_CLASS[lab];
    if (cls && CLASS_COLOR[cls]) return CLASS_COLOR[cls];
    // If label is not mapped to a class, show as white
    return CLASS_COLOR['none'];
  };
  for (let i=0;i<count;i++){
    const lab = segLabels[i];
    let c = colorForLabel(lab);
    if (classFilter){
      if (ID_TO_CLASS[lab] === classFilter) {
        // keep bright
      } else {
        // dim others to grey
        c = new THREE.Color(0x666666);
      }
    }
    colors[i*3+0]=c.r; colors[i*3+1]=c.g; colors[i*3+2]=c.b;
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions,3));
  if (indices) g.setIndex(new THREE.BufferAttribute(indices,1));
  g.setAttribute('color', new THREE.BufferAttribute(colors,3));

  if (indices){
    const m = new THREE.MeshLambertMaterial({ vertexColors:true, side:THREE.DoubleSide, opacity:0.9, transparent:true });
    segMesh = new THREE.Mesh(g,m);
  } else {
    const m = new THREE.PointsMaterial({ size: 3, sizeAttenuation: true, vertexColors:true, opacity:0.9, transparent:true });
    segMesh = new THREE.Points(g,m);
  }
  scene.add(segMesh);
  renderOnce();
}

function orderSelectionAsPolygonXY(){
  const idxs = Array.from(selection);
  if (idxs.length < 3) return null;
  let cx=0, cy=0;
  for (const i of idxs){ cx += positions[i*3]; cy += positions[i*3+1]; }
  cx /= idxs.length; cy /= idxs.length;
  const withAngle = idxs.map(i => {
    const x = positions[i*3] - cx;
    const y = positions[i*3+1] - cy;
    return { i, ang: Math.atan2(y, x) };
  });
  withAngle.sort((a,b)=>a.ang-b.ang);
  const poly = withAngle.map(o => [positions[o.i*3], positions[o.i*3+1]]);
  return poly;
}

function pointsInPolygonXY(poly){
  const px = poly.map(p=>p[0]);
  const py = poly.map(p=>p[1]);
  const m = px.length;
  if (m < 3) return [];
  const minx = Math.min(...px), maxx = Math.max(...px);
  const miny = Math.min(...py), maxy = Math.max(...py);

  const inside = [];
  for (let n=0; n<count; n++){
    const x = positions[n*3], y = positions[n*3+1];
    if (x < minx || x > maxx || y < miny || y > maxy) continue;
    let c = false;
    for (let i=0, j=m-1; i<m; j=i++){
      const xi = px[i], yi = py[i];
      const xj = px[j], yj = py[j];
      const intersect = ((yi>y) !== (yj>y)) &&
                        (x < (xj - xi) * (y - yi) / ((yj - yi) || 1e-12) + xi);
      if (intersect) c = !c;
    }
    if (c) inside.push(n);
  }
  return inside;
}

// ---------- Events ----------
renderer.domElement.addEventListener('pointerdown', (ev) => {
  if (!object3D) return;
  const rect = renderer.domElement.getBoundingClientRect();
  const mx=((ev.clientX-rect.left)/rect.width)*2-1;
  const my=-((ev.clientY-rect.top)/rect.height)*2+1;
  mouse.set(mx,my);
  raycaster.setFromCamera(mouse,camera);
  
  // Click hazard markers → show details
  const hitsHaz = raycaster.intersectObjects(hazardGroup.children, true);
  if (hitsHaz.length) {
    const d = (hitsHaz[0].object.userData || {});
    const fmt = (n) => (typeof n === 'number' && isFinite(n)) ? n.toFixed(2) : '—';
    alert(
      `⚠ Hazard\n` +
      `Kind: ${d.kind ?? 'hazard'}\n` +
      `Severity: ${d.severity ?? '—'}\n` +
      `Note: ${d.note ? String(d.note) : '—'}\n` +
      `Position: x ${fmt(d.x)}, y ${fmt(d.y)}, z ${fmt(d.z)}`
    );
    return;
  }
  
  const hit = raycaster.intersectObject(object3D,true)[0];
  if (!hit) return;
  let idx = 0;
  if (hit.index !== undefined) idx = hit.index;
  else if (hit.face) {
    const hp = hit.point;
    const a=hit.face.a, b=hit.face.b, c=hit.face.c;
    const va = new THREE.Vector3(positions[a*3],positions[a*3+1],positions[a*3+2]);
    const vb = new THREE.Vector3(positions[b*3],positions[b*3+1],positions[b*3+2]);
    const vc = new THREE.Vector3(positions[c*3],positions[c*3+1],positions[c*3+2]);
    const da=hp.distanceTo(va), db=hp.distanceTo(vb), dc=hp.distanceTo(vc);
    idx = (da<db && da<dc)? a : (db<dc? b : c);
  }
  
  if (currentTool !== 'none') {
    const z = positions[idx*3+2];
    const s = slope ? slope[idx] : 0;
    hud.innerHTML = `⛰️ <b>Probe</b><br/>#${idx} · z=${z.toFixed(2)} m · slope=${s.toFixed(1)}°`;
  }
  if (currentTool === 'none') return;
  if (currentTool === 'probe') {
    if (ev.altKey){ startIdx = idx; startText.textContent = '#'+idx; }
    if (ev.ctrlKey){
      goalIdx = idx; goalText.textContent = '#'+idx;
      if (modelId && startIdx!=null && indices){
        const mode = btnSafe.classList.contains('active') ? 'safe' : 'fast';
        const maxSlope = DEFAULT_MAX_SLOPE;
        const riskPow  = DEFAULT_RISK_POW;
        fetch(`/api/route/${modelId}`, {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({ start:startIdx, goal:goalIdx, mode, maxSlope, riskPow })
        }).then(r => r.json()).then(res => {
          if (res.error){ routeText.textContent = res.error; alert(res.error); drawRoute(null); return; }
          routeText.textContent = `${res.lengthKm.toFixed(2)} km · +${res.gainM.toFixed(0)} m`;
          drawRoute(res.path);
        }).catch(e => { routeText.textContent='route error'; console.error(e); });
      }
    }
    return;
  }
  if (currentTool === 'area') {
    if (ev.shiftKey){ selection.add(idx); updateSelectionVis(); }
    return;
  }
  if (currentTool === 'pin') {
    if (!currentUser) { alert('Login required to add hazards.'); return; }
    const kind = prompt('Hazard kind (e.g., rockfall, crevasse, avalanche):','rockfall') || 'hazard';
    const sev  = parseInt(prompt('Severity 1-5:','3') || '3', 10);
    const note = prompt('Note:','') || '';
    fetch(`/api/hazards/${modelId}`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ index: idx, kind, severity: sev, note })
    }).then(()=>reloadHazards());
    return;
  }
  if (currentTool === 'segment') {
    if (ev.shiftKey && segLabels){
      const labStr = prompt('Label ID (integer, 0..k-1):','0');
      if (labStr==null) return;
      const lab = parseInt(labStr||'0',10);
      fetch(`/api/segment/${modelId}/paint`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ indices:[idx], label: lab })
      }).then(()=>loadSegLabels(true));
    }
    return;
  }
  if (currentTool === 'draw') {
    if (ev.altKey) {
      if (customPath.length) customPath.pop();
    } else {
      customPath.push(idx);
    }
    updateCustomVis();
    return;
  }
});

// Event listeners
document.getElementById('btnAreaStats').addEventListener('click', () => {
  if (selection.size === 0) { alert('No area selected.'); return; }
  let totalArea = 0;
  let avgSlope = 0;
  let avgElev = 0;
  for (const idx of selection) {
    if (slope) avgSlope += slope[idx];
    if (positions) avgElev += positions[idx*3+2];
  }
  avgSlope /= selection.size;
  avgElev /= selection.size;
  alert(`Area Stats:\nVertices: ${selection.size}\nAvg Elevation: ${avgElev.toFixed(2)} m\nAvg Slope: ${avgSlope.toFixed(1)}°`);
});

document.getElementById('btnClearSel').addEventListener('click', () => {
  selection.clear();
  updateSelectionVis();
});

document.getElementById('btnReloadHaz').addEventListener('click', reloadHazards);

document.getElementById('hazSize').addEventListener('input', (e) => {
  hazardSizeScale = parseFloat(e.target.value);
  reloadHazards();
});

toolEl.addEventListener('change', (e) => {
  currentTool = e.target.value;
});

document.getElementById('btnShowRoutes').addEventListener('click', () => {
  if (!modelId) return;
  fetch(`/api/routes/${modelId}`).then(r=>r.json()).then(list=>{
    scene.traverse(obj => {
      if (obj.userData && obj.userData._savedRoute) { scene.remove(obj); obj.geometry?.dispose?.(); obj.material?.dispose?.(); }
    });
    for (const r of list){
      const arr = r.path;
      if (!arr || arr.length<2) continue;
      const pts = new Float32Array(arr.length*3);
      for (let i=0;i<arr.length;i++){
        const j = arr[i];
        pts[i*3+0]=positions[j*3+0]; pts[i*3+1]=positions[j*3+1]; pts[i*3+2]=positions[j*3+2];
      }
      const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pts,3));
      const m = new THREE.LineBasicMaterial({ color: 0xffffff, opacity: 0.35, transparent:true });
      const l = new THREE.Line(g,m); l.userData = {_savedRoute:true};
      scene.add(l);
    }
    renderOnce();
  });
});

document.getElementById('btnSaveRoute').addEventListener('click', () => {
  if (!modelId || !routeLine) { alert('No route to save.'); return; }
  const name = prompt('Route name:', 'my route') || 'my route';
  fetch(`/api/route/${modelId}/save`, {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ name, path: [], tags:['saved'] })
  }).then(r=>r.json()).then(saved=>{
    if (saved && saved.id) alert('Saved route #' + saved.id);
  }).catch(e=>alert('Save failed: ' + e));
});

document.getElementById('btnApplyClass').addEventListener('click', async () => {
  if (!modelId){ alert('Shared model is always used.'); return; }
  if (selection.size < 3){ alert('Select at least 3 vertices with Shift+Click (Tool: Area select).'); return; }

  const ok = await ensureSegmentation();
  if (!ok){ alert('Could not prepare label storage.'); return; }

  const poly = orderSelectionAsPolygonXY();
  if (!poly){ alert('Polygon ordering failed.'); return; }

  const inside = pointsInPolygonXY(poly);
  if (!inside.length){ alert('No vertices fell inside the polygon. Try a larger area.'); return; }

  const cls = document.getElementById('clsType').value;
  const label = CLASS_TO_ID[cls];

  const CH = 25000;
  for (let s=0; s<inside.length; s+=CH){
    const chunk = inside.slice(s, s+CH);
    await fetch(`/api/segment/${modelId}/paint`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ indices: chunk, label })
    });
  }

  alert(`Applied "${cls}" to ${inside.length} vertices inside polygon.`);
  // Clear selection after applying class
  selection.clear();
  updateSelectionVis();
  loadSegLabels(true);
});

document.getElementById('btnShowClasses').addEventListener('click', () => {
  segVisible = true; classFilter = null;
  if (!segLabels) loadSegLabels(true); else drawSegOverlay();
});

document.getElementById('btnHideClasses').addEventListener('click', () => {
  segVisible = false; classFilter = null;
  if (segMesh){ scene.remove(segMesh); segMesh.geometry.dispose(); segMesh.material.dispose(); segMesh=null; renderOnce(); }
});

document.getElementById('btnCustomUndo').addEventListener('click', () => {
  if (customPath.length) customPath.pop();
  updateCustomVis();
});
document.getElementById('btnCustomClear').addEventListener('click', clearCustom);
document.getElementById('btnCustomSave').addEventListener('click', () => {
  if (!modelId || customPath.length < 2){ alert('Add at least 2 points in "Draw route" mode.'); return; }
  const name = prompt('Custom route name:', 'my path') || 'my path';
  fetch(`/api/route/${modelId}/save`, {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ name, path: customPath, tags:['custom'] })
  }).then(r=>r.json()).then(saved=>{
    if (saved && saved.id) alert('Saved custom route #' + saved.id);
  }).catch(e=>alert('Save failed: ' + e));
});

document.getElementById('btnRunSeg').addEventListener('click', () => {
  if (!modelId) return;
  fetch(`/api/segment/${modelId}`, {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ k:4, features:["elev","slope","nx","ny","nz"] })
  }).then(()=>loadSegLabels(true));
});
document.getElementById('btnToggleSeg').addEventListener('click', () => {
  segVisible = !segVisible; classFilter = null;
  if (segVisible){ if (!segLabels) loadSegLabels(true); else drawSegOverlay(); }
  else { if (segMesh){ scene.remove(segMesh); segMesh.geometry.dispose(); segMesh.material.dispose(); segMesh=null; renderOnce(); } }
});

btnHide.addEventListener('click', () => { panel.style.display = 'none'; btnToggle.style.display = 'block'; });
btnToggle.addEventListener('click', () => { panel.style.display = 'block'; btnToggle.style.display = 'none'; });

// Auto-load shared model for all users
async function loadDefaultModel() {
  let defaultModelId = null;
  try {
    const res = await fetch('/');
    const text = await res.text();
    const match = text.match(/default_model_id\s*=\s*['\"]([\w-]+)['\"]/);
    if (match) defaultModelId = match[1];
  } catch {}
  if (!defaultModelId) defaultModelId = 'default-check-ply';
  
  // Fetch model meta first
  let meta;
  try {
    meta = await fetch(`/api/meta/${defaultModelId}`).then(r=>r.json());
    if (meta.error) {
      console.error('Meta error:', meta.error);
      alert(meta.error);
      return;
    }
  } catch (e) {
    console.error('Could not fetch meta:', e);
    alert('Failed to load model meta');
    return;
  }

  console.log('Meta response:', meta);
  modelId = defaultModelId;
  modelBounds = meta.bounds;
  count = meta.count || 0;

  // Load positions (required)
  try {
    const posRes = await fetch(`/api/positions/${defaultModelId}`);
    if (posRes.ok) {
      const posBuffer = await posRes.arrayBuffer();
      if (posBuffer.byteLength > 0) {
        positions = new Float32Array(posBuffer);
        count = positions.length / 3;
        console.log('Loaded positions:', count, 'vertices');
      }
    } else {
      throw new Error('Failed to load positions');
    }
  } catch (e) {
    console.error('Could not load positions:', e);
    alert('Failed to load model geometry');
    return;
  }

  // Load indices (if available)
  if (meta.has_indices) {
    try {
      const idxRes = await fetch(`/api/indices/${defaultModelId}`);
      if (idxRes.ok) {
        const idxBuffer = await idxRes.arrayBuffer();
        if (idxBuffer.byteLength > 0) {
          indices = new Uint32Array(idxBuffer);
          console.log('Loaded indices:', indices.length);
        }
      }
    } catch (e) {
      console.log('Could not load indices:', e);
    }
  }

  // Load normals (optional)
  try {
    const normalRes = await fetch(`/api/normals/${defaultModelId}`);
    if (normalRes.ok) {
      const normalBuffer = await normalRes.arrayBuffer();
      if (normalBuffer.byteLength > 0) {
        normals = new Float32Array(normalBuffer);
        console.log('Loaded normals');
      }
    }
  } catch (e) {
    console.log('Could not load normals:', e);
  }

  // Load slope data (optional)
  try {
    const slopeRes = await fetch(`/api/slope/${defaultModelId}`);
    if (slopeRes.ok) {
      const slopeBuffer = await slopeRes.arrayBuffer();
      if (slopeBuffer.byteLength > 0) {
        slope = new Float32Array(slopeBuffer);
        console.log('Loaded slope data');
      }
    }
  } catch (e) {
    console.log('Could not load slope data:', e);
  }

  // Create the 3D object
  if (positions && positions.length > 0) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    if (normals && normals.length > 0) {
      geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    } else {
      geometry.computeVertexNormals();
      console.log('Computed vertex normals');
    }

    if (indices && indices.length > 0) {
      geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    }

    const material = indices ?
      new THREE.MeshLambertMaterial({ color: 0x888888, side: THREE.DoubleSide, wireframe: false }) :
      new THREE.PointsMaterial({ color: 0x888888, size: 3, sizeAttenuation: true });

    object3D = indices ? new THREE.Mesh(geometry, material) : new THREE.Points(geometry, material);
    scene.add(object3D);
    console.log('Added 3D object to scene');

    // Position camera to view the model
    if (modelBounds) {
      const center = new THREE.Vector3(
        (modelBounds.min.x + modelBounds.max.x) / 2,
        (modelBounds.min.y + modelBounds.max.y) / 2,
        (modelBounds.min.z + modelBounds.max.z) / 2
      );
      const size = Math.max(
        modelBounds.max.x - modelBounds.min.x,
        modelBounds.max.y - modelBounds.min.y,
        modelBounds.max.z - modelBounds.min.z
      );

      camera.position.set(center.x, center.y, center.z + size * 2);
      controls.target.copy(center);
      controls.update();
      console.log('Positioned camera at bounds:', modelBounds);
    }

    renderOnce();
  } else {
    console.error('No position data loaded');
    alert('Failed to load model geometry - no position data');
    return;
  }

  // Load hazards and initialize
  reloadHazards();
  initializeClassMappings();
}

// Initialize class mappings
function initializeClassMappings() {
  CLASS_TO_ID = {
    'none': 0,
    'crevasse': 1,
    'glacier': 2,
    'snow': 3,
    'rock': 4,
    'ice': 5,
    'other': 6
  };
  ID_TO_CLASS = {};
  for (const [cls, id] of Object.entries(CLASS_TO_ID)) {
    ID_TO_CLASS[id] = cls;
  }
  CLASS_COLOR = {
    'none': new THREE.Color(0xffff00), // yellow
    'crevasse': new THREE.Color(0xff69b4), // pink
    'glacier': new THREE.Color(0x00ffff),
    'snow': new THREE.Color(0xffffff),
    'rock': new THREE.Color(0x8b4513),
    'ice': new THREE.Color(0x87ceeb),
    'other': new THREE.Color(0x808080)
  };
}

// Start the application

// --- Add classification color legend overlay ---
function addClassLegend() {
  if (document.getElementById('classLegend')) return;
  const legend = document.createElement('div');
  legend.id = 'classLegend';
  legend.style.position = 'fixed';
  legend.style.right = '18px';
  legend.style.bottom = '18px';
  legend.style.background = '#0b1226cc';
  legend.style.border = '1px solid #1a2340';
  legend.style.borderRadius = '10px';
  legend.style.padding = '10px 16px 10px 12px';
  legend.style.fontSize = '13px';
  legend.style.zIndex = 1002;
  legend.style.color = '#e7eefb';
  legend.style.boxShadow = '0 2px 8px #0008';
  legend.innerHTML = `
    <b style="font-size:14px; color:#cfe5ff;">Legend</b><br>
    <div style="display:flex; flex-direction:column; gap:4px; margin-top:6px;">
      <span><span style="display:inline-block;width:18px;height:12px;background:#ffff00;border-radius:3px;margin-right:7px;border:1px solid #bbb;"></span>None</span>
      <span><span style="display:inline-block;width:18px;height:12px;background:#ff69b4;border-radius:3px;margin-right:7px;"></span>Crevasse</span>
      <span><span style="display:inline-block;width:18px;height:12px;background:#00ffff;border-radius:3px;margin-right:7px;"></span>Glacier</span>
      <span><span style="display:inline-block;width:18px;height:12px;background:#fff;border-radius:3px;margin-right:7px;border:1px solid #bbb;"></span>Snow</span>
      <span><span style="display:inline-block;width:18px;height:12px;background:#8b4513;border-radius:3px;margin-right:7px;"></span>Rock</span>
      <span><span style="display:inline-block;width:18px;height:12px;background:#87ceeb;border-radius:3px;margin-right:7px;"></span>Ice</span>
      <span><span style="display:inline-block;width:18px;height:12px;background:#808080;border-radius:3px;margin-right:7px;"></span>Other</span>
    </div>
  `;
  document.body.appendChild(legend);
}

addClassLegend();

loadDefaultModel();

// --- Overlay logic for Slope and Route Heatmap ---
let overlayMesh = null;

function clearOverlay() {
  if (overlayMesh) {
    scene.remove(overlayMesh);
    overlayMesh.geometry?.dispose?.();
    overlayMesh.material?.dispose?.();
    overlayMesh = null;
    renderOnce();
  }
}

async function showSlopeOverlay() {
  if (!positions || !slope) return;
  clearOverlay();
  // Color vertices by slope (blue=low, red=high)
  const colors = new Float32Array(count * 3);
  let minS = 90, maxS = 0;
  for (let i = 0; i < count; i++) {
    if (slope[i] < minS) minS = slope[i];
    if (slope[i] > maxS) maxS = slope[i];
  }
  for (let i = 0; i < count; i++) {
    const t = (slope[i] - minS) / (maxS - minS + 1e-6);
    // blue (low) to red (high)
    colors[i*3+0] = t;
    colors[i*3+1] = 0.2;
    colors[i*3+2] = 1.0-t;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  if (indices && indices.length > 0) g.setIndex(new THREE.BufferAttribute(indices, 1));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  let m;
  if (indices && indices.length > 0) {
    m = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide, opacity: 0.85, transparent: true });
    overlayMesh = new THREE.Mesh(g, m);
  } else {
    m = new THREE.PointsMaterial({ size: 3, sizeAttenuation: true, vertexColors: true, opacity: 0.85, transparent: true });
    overlayMesh = new THREE.Points(g, m);
  }
  overlayMesh.renderOrder = 1000;
  scene.add(overlayMesh);
  renderOnce();
}

async function showHeatmapOverlay() {
  if (!positions || !modelId) return;
  clearOverlay();
  // Fetch heatmap data
  let indicesArr = [], countsArr = [];
  try {
    const res = await fetch(`/api/routes/${modelId}/heatmap`);
    const data = await res.json();
    indicesArr = data.indices || [];
    countsArr = data.counts || [];
  } catch (e) {
    alert('Failed to load route heatmap');
    return;
  }
  if (!indicesArr.length) {
    alert('No heatmap data');
    return;
  }
  // Normalize counts for color
  let minC = Math.min(...countsArr), maxC = Math.max(...countsArr);
  const colors = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    colors[i*3+0] = 0.5; // default gray
    colors[i*3+1] = 0.5;
    colors[i*3+2] = 0.5;
  }
  for (let k = 0; k < indicesArr.length; k++) {
    const i = indicesArr[k];
    const t = (countsArr[k] - minC) / (maxC - minC + 1e-6);
    // yellow (high) to blue (low)
    colors[i*3+0] = 1.0;
    colors[i*3+1] = 1.0 - t;
    colors[i*3+2] = 0.2 + 0.8 * (1.0 - t);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  if (indices && indices.length > 0) g.setIndex(new THREE.BufferAttribute(indices, 1));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  let m;
  if (indices && indices.length > 0) {
    m = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide, opacity: 0.85, transparent: true });
    overlayMesh = new THREE.Mesh(g, m);
  } else {
    m = new THREE.PointsMaterial({ size: 3, sizeAttenuation: true, vertexColors: true, opacity: 0.85, transparent: true });
    overlayMesh = new THREE.Points(g, m);
  }
  overlayMesh.renderOrder = 1000;
  scene.add(overlayMesh);
  renderOnce();
}

// Overlay dropdown event
if (overlayEl) {
  overlayEl.addEventListener('change', async (e) => {
    clearOverlay();
    if (!overlayEl.value || overlayEl.value === 'none') return;
    if (overlayEl.value === 'slope') {
      await showSlopeOverlay();
    } else if (overlayEl.value === 'heatmap') {
      await showHeatmapOverlay();
    }
  });
}