from __future__ import annotations

import os, uuid, json, sqlite3, typing as t
import numpy as np
import trimesh
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from scipy.spatial import cKDTree, ConvexHull

app = Flask(__name__)
BASE_DIR = os.path.dirname(__file__)
app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


# In-memory geometry store (file -> arrays). Metadata persists via SQLite.
MODELS: dict[str, dict] = {}  # mid -> {P, idx, normals, slope, graph, bounds, path}
DEFAULT_MODEL_ID = None

DB_PATH = os.path.join(BASE_DIR, "peaks3d.db")

# --------------------------- SQLite helpers --------------------------- #
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
        conn = db()
        cur = conn.cursor()
        # Users table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""")
        # Hazards table (add user_id)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS hazards(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL,
            ix INTEGER,
            x REAL, y REAL, z REAL,
            kind TEXT, severity INTEGER, note TEXT,
            user_id INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""")
        # Routes table (add user_id)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS routes(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL,
            name TEXT,
            path TEXT NOT NULL,        -- JSON array of indices
            tags TEXT,                 -- JSON array of strings
            user_id INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS segments(
            model_id TEXT PRIMARY KEY,
            k INTEGER NOT NULL,
            labels_path TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""")
        conn.commit(); conn.close()
# ------------------------------- Routes ------------------------------ #
from flask import session
from werkzeug.security import generate_password_hash, check_password_hash
app.secret_key = os.environ.get("SECRET_KEY", "peaks3d_secret")

# ------------------------------- Auth ------------------------------- #
@app.post("/api/signup")
def api_signup():
    data = request.get_json(force=True) or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", ""))
    if not username or not password:
        return jsonify(error="username and password required"), 400
    conn = db(); cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username=?", (username,))
    if cur.fetchone():
        conn.close()
        return jsonify(error="username already exists"), 400
    pw_hash = generate_password_hash(password)
    cur.execute("INSERT INTO users(username, password_hash) VALUES (?, ?)", (username, pw_hash))
    conn.commit(); uid = cur.lastrowid; conn.close()
    session["user_id"] = uid
    session["username"] = username
    return jsonify(id=uid, username=username)

@app.post("/api/login")
def api_login():
    data = request.get_json(force=True) or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", ""))
    conn = db(); cur = conn.cursor()
    row = cur.execute("SELECT id, password_hash FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    if not row or not check_password_hash(row["password_hash"], password):
        return jsonify(error="invalid credentials"), 401
    session["user_id"] = row["id"]
    session["username"] = username
    return jsonify(id=row["id"], username=username)

@app.post("/api/logout")
def api_logout():
    session.clear()
    return jsonify(ok=True)


def load_default_model():
    global DEFAULT_MODEL_ID
    model_path = os.path.join(app.config["UPLOAD_FOLDER"], "check.ply")
    if not os.path.exists(model_path):
        print(f"Default model file not found: {model_path}")
        return
    try:
        P, idx, normals, bounds = load_mesh_to_arrays(model_path)
    except Exception as e:
        print(f"Failed to load default model: {e}")
        return
    slope = slope_from_normals(normals)
    graph = build_graph(P.shape[0], idx, P=P)
    mid = "default-check-ply"
    MODELS[mid] = dict(P=P, idx=idx, normals=normals, slope=slope, graph=graph, bounds=bounds, path=model_path)
    DEFAULT_MODEL_ID = mid
    print(f"Loaded default model: {model_path} as id {mid}")



# ------------------------- Geometry utilities ------------------------- #
def _as_mesh(scene_or_mesh):
    """Convert a trimesh.Scene to a single Trimesh by concatenation."""
    if isinstance(scene_or_mesh, trimesh.Scene):
        geoms = list(scene_or_mesh.geometry.values())
        if not geoms:
            raise ValueError("scene has no geometry")
        return trimesh.util.concatenate(geoms)
    return scene_or_mesh

def load_mesh_to_arrays(path: str):
    """
    Load .ply or .obj (and friends) into arrays.
    Returns (P[N,3], idx[3T] or None, normals[N,3], bounds dict)
    """
    m = trimesh.load(path, process=False)
    m = _as_mesh(m)

    if isinstance(m, trimesh.points.PointCloud):
        P = np.asarray(m.vertices, dtype=np.float32)
        idx = None
        normals = estimate_normals(P)
    else:
        if not hasattr(m, "vertices"):
            m = trimesh.load_mesh(path, process=False)
        P = np.asarray(m.vertices, dtype=np.float32)
        idx = None
        if hasattr(m, "faces") and m.faces is not None and len(m.faces) > 0:
            idx = np.asarray(m.faces, dtype=np.uint32).reshape(-1)
        normals = None
        if hasattr(m, "vertex_normals") and m.vertex_normals is not None and len(m.vertex_normals) == len(P):
            normals = np.asarray(m.vertex_normals, dtype=np.float32)
        if normals is None:
            if idx is not None:
                _ = m.vertex_normals  # triggers compute
                normals = np.asarray(m.vertex_normals, dtype=np.float32)
            else:
                normals = estimate_normals(P)

    bounds = dict(
        min=dict(x=float(P[:,0].min()), y=float(P[:,1].min()), z=float(P[:,2].min())),
        max=dict(x=float(P[:,0].max()), y=float(P[:,1].max()), z=float(P[:,2].max())),
    )
    return P, idx, normals, bounds

def estimate_normals(P: np.ndarray, k: int = 16) -> np.ndarray:
    tree = cKDTree(P)
    normals = np.empty_like(P, dtype=np.float32)
    k = min(k, len(P))
    for i in range(P.shape[0]):
        _, idx = tree.query(P[i], k=k)
        pts = P[idx]
        C = np.cov(pts.T)
        _, v = np.linalg.eigh(C)
        n = v[:, 0]
        if n[2] < 0: n = -n
        normals[i] = n
    return normals

def slope_from_normals(normals: np.ndarray) -> np.ndarray:
    nz = np.clip(normals[:,2], -1.0, 1.0)
    return np.degrees(np.arccos(nz)).astype(np.float32)

def risk_from_slope(s: t.Union[np.ndarray,float]) -> np.ndarray:
    s = np.clip(s, 0, 90)
    lo = 1/(1+np.exp(-(s-22)/3))
    hi = 1/(1+np.exp(-(40-s)/3))
    return np.clip(lo*hi, 0, 1)

def build_graph(count: int, indices: np.ndarray|None, P: np.ndarray|None=None,
                knn: int=16, max_nodes_knn: int=300_000) -> list[list[int]]|None:
    if indices is not None:
        G: list[list[int]] = [[] for _ in range(count)]
        tris = indices.reshape(-1,3)
        for a,b,c in tris:
            G[a].extend([b,c]); G[b].extend([a,c]); G[c].extend([a,b])
        for i in range(count): G[i] = sorted(set(G[i]))
        return G
    if P is None or count==0: return None
    if count > max_nodes_knn:
        app.logger.warning("kNN graph skipped: %d > %d", count, max_nodes_knn)
        return None
    tree = cKDTree(P)
    kq = min(knn+1, count)
    nbrs = tree.query(P, k=kq)[1]
    if nbrs.ndim==1: nbrs = nbrs.reshape(-1,1)
    nbrs = nbrs[:,1:]
    return [sorted(set(row.tolist())) for row in nbrs]

def snap_to_feasible(P: np.ndarray, S: np.ndarray, ix: int, maxSlope: float) -> int:
    if 0 <= ix < len(S) and S[ix] <= maxSlope: return int(ix)
    good = np.where(S <= maxSlope)[0]
    if good.size == 0: return int(ix)
    tree = cKDTree(P[good]); _, j = tree.query(P[int(ix)])
    return int(good[int(j)])

def dijkstra_route(
    G: list[list[int]],
    P: np.ndarray,
    S: np.ndarray,
    start: int,
    goal: int,
    mode: str,
    maxSlope: float,
    riskPow: float,
    soft: bool = True,
):
    """
    FAST  => pure geometric shortest path (no slope gating / penalties)
    SAFE  => geometric distance * (1 + risk penalty), with 'soft' slope penalty above maxSlope
    """
    import heapq

    N = len(G)
    if N == 0: 
        return None

    # Clamp indices
    start = int(np.clip(start, 0, N - 1))
    goal  = int(np.clip(goal,  0, N - 1))

    dist = np.full(N, np.inf)
    prev = np.full(N, -1, dtype=np.int32)
    seen = np.zeros(N, bool)
    pq: list[tuple[float, int]] = []

    def edge_cost(u: int, v: int) -> float:
        x1, y1, z1 = P[u]
        x2, y2, z2 = P[v]
        dx, dy, dz = (x2 - x1), (y2 - y1), (z2 - z1)
        d3 = float(np.sqrt(dx * dx + dy * dy + dz * dz))

        if mode == "fast":
            # pure geometric shortest path
            return d3

        # SAFE mode
        s1, s2 = float(S[u]), float(S[v])
        if not soft and (s1 > maxSlope or s2 > maxSlope):
            return np.inf

        # risk term (0..1) then ^riskPow to accentuate
        r = float(risk_from_slope(np.array([s1, s2])).mean()) ** max(riskPow, 1e-6)

        # soft penalty for exceeding maxSlope (smoothly increases cost)
        over = max(0.0, max(s1, s2) - maxSlope)
        pen = 1.0
        if soft and over > 0:
            frac = over / max(1.0, 90.0 - maxSlope)
            pen *= (1.0 + 9.0 * (frac ** 2))  # up to 10x

        return d3 * (1.0 + 4.0 * r) * pen

    dist[start] = 0.0
    heapq.heappush(pq, (0.0, start))

    while pq:
        du, u = heapq.heappop(pq)
        if seen[u]:
            continue
        seen[u] = True
        if u == goal:
            break
        for v in G[u]:
            c = edge_cost(u, v)
            if not np.isfinite(c):
                continue
            nd = du + c
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if not np.isfinite(dist[goal]):
        return None

    # Reconstruct & stats
    path: list[int] = []
    v = goal
    while v != -1:
        path.append(int(v))
        v = int(prev[v])
    path.reverse()

    length = 0.0
    gain = 0.0
    for i in range(1, len(path)):
        a, b = path[i - 1], path[i]
        dx, dy, dz = (P[b] - P[a]).tolist()
        length += float(np.sqrt(dx * dx + dy * dy + dz * dz))
        if dz > 0:
            gain += dz

    return dict(path=path, lengthKm=length / 1000.0, gainM=gain)



# ------------------------------ K-means ------------------------------ #
def kmeans_numpy(X: np.ndarray, k: int, iters: int=20, seed: int=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n==0: return np.zeros(0, dtype=np.int32), np.zeros((k,X.shape[1]))
    init_idx = rng.choice(n, size=min(k,n), replace=False)
    C = X[init_idx].copy()
    L = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        d2 = ((X[:,None,:]-C[None,:,:])**2).sum(-1)
        L_new = d2.argmin(axis=1).astype(np.int32)
        if np.array_equal(L_new, L): break
        L = L_new
        for j in range(k):
            pts = X[L==j]
            if len(pts)>0: C[j]=pts.mean(axis=0)
    return L, C

# ------------------------------ Presets ------------------------------ #
# Only two options are exposed to the client: 'fast' and 'safe'.
# We map them to fixed parameters and ignore any client-provided maxSlope/riskPow.
ROUTE_PRESETS = {
    "fast":   dict(mode="fast", maxSlope=75.0, riskPow=1.0),
    "fastest":dict(mode="fast", maxSlope=75.0, riskPow=1.0),  # alias
    "safe":   dict(mode="safe", maxSlope=40.0, riskPow=3.0),
}

# ------------------------------- Routes ------------------------------ #

@app.get("/")
def index():
    # Pass default model id to frontend
    return render_template("index.html", default_model_id=DEFAULT_MODEL_ID)


# Remove upload endpoint (disable model upload for all users)
@app.post("/api/upload")
def api_upload():
    return jsonify(error="Model upload disabled. Shared model is always used."), 403


# Always use the default model for all API endpoints
def get_default_model():
    return MODELS.get(DEFAULT_MODEL_ID)

@app.post("/api/route/<mid>")
def api_route(mid: str):
    m = get_default_model()
    if not m:
        return jsonify(error="model not found"), 404
    data = request.get_json(force=True) or {}
    try:
        start = int(data.get("start", -1))
        goal  = int(data.get("goal",  -1))
    except Exception:
        return jsonify(error="start/goal required"), 400
    if start < 0 or goal < 0:
        return jsonify(error="start/goal required"), 400
    mode     = str(data.get("mode", "fast"))
    maxSlope = float(data.get("maxSlope", 55))
    riskPow  = float(data.get("riskPow", 2.0))
    G, P, S = m["graph"], m["P"], m["slope"]
    if G is None:
        return jsonify(error="routing requires a mesh or kNN graph"), 400
    s0, g0 = start, goal
    start = snap_to_feasible(P, S, start, maxSlope)
    goal  = snap_to_feasible(P, S, goal,  maxSlope)
    if mode == "fast":
        res = dijkstra_route(G, P, S, start, goal, mode="fast", maxSlope=90, riskPow=0.0, soft=True)
        if res is None:
            return jsonify(error="no route; graph disconnected"), 400
        res.update(start=start, goal=goal, snapped=bool((s0 != start) or (g0 != goal)), modeUsed="fast")
        return jsonify(res)
    res = dijkstra_route(G, P, S, start, goal, mode="safe", maxSlope=maxSlope, riskPow=riskPow, soft=True)
    if res is None:
        for relax in (5, 10, 15, 20, 25):
            res = dijkstra_route(G, P, S, start, goal, mode="safe",
                                 maxSlope=min(90.0, maxSlope + relax),
                                 riskPow=riskPow, soft=True)
            if res is not None:
                res.update(relaxedTo=min(90.0, maxSlope + relax))
                break
    if res is None:
        res = dijkstra_route(G, P, S, start, goal, mode="fast", maxSlope=90, riskPow=0.0, soft=True)
        if res is None:
            return jsonify(error="no route; graph disconnected"), 400
        res.update(fallbackFast=True)
    res.update(start=start, goal=goal, snapped=bool((s0 != start) or (g0 != goal)), modeUsed="safe")
    return jsonify(res)




@app.post("/api/route/<mid>/save")
def api_route_save(mid: str):
    m = get_default_model()
    if not m: return jsonify(error="model not found"), 404
    data = request.get_json(force=True) or {}
    name = str(data.get("name","route"))
    path = data.get("path")
    tags = data.get("tags", [])
    user_id = session.get("user_id")
    if not user_id:
        return jsonify(error="login required"), 401
    if not isinstance(path, list) or len(path)<2: return jsonify(error="path required"), 400
    conn = db(); cur = conn.cursor()
    cur.execute("INSERT INTO routes(model_id,name,path,tags,user_id) VALUES(?,?,?,?,?)",
                (DEFAULT_MODEL_ID, name, json.dumps(path), json.dumps(tags), user_id))
    conn.commit(); rid = cur.lastrowid; conn.close()
    return jsonify(id=int(rid))


@app.get("/api/routes/<mid>")
def api_routes(mid: str):
    conn = db(); cur = conn.cursor()
    rows = cur.execute("SELECT id,name,path,tags,created_at FROM routes WHERE model_id=? ORDER BY id DESC", (DEFAULT_MODEL_ID,)).fetchall()
    conn.close()
    out=[]
    for r in rows:
        out.append(dict(id=r["id"], name=r["name"], path=json.loads(r["path"]), tags=json.loads(r["tags"] or "[]"), created_at=r["created_at"]))
    return jsonify(out)


@app.get("/api/routes/<mid>/heatmap")
def api_routes_heatmap(mid: str):
    m = get_default_model()
    if not m: return jsonify(error="model not found"), 404
    N = m["P"].shape[0]
    counts = np.zeros(N, dtype=np.int32)
    conn = db(); cur = conn.cursor()
    for r in cur.execute("SELECT path FROM routes WHERE model_id=?", (DEFAULT_MODEL_ID,)):
        arr = json.loads(r["path"])
        for ix in arr: 
            if 0 <= ix < N: counts[ix]+=1
    conn.close()
    nz = counts.nonzero()[0].tolist()
    vals = counts[counts>0].tolist()
    return jsonify(indices=nz, counts=vals)

# ------------------------------- Hazards ----------------------------- #

@app.get("/api/hazards/<mid>")
def api_hazards(mid: str):
    conn = db(); cur = conn.cursor()
    rows = cur.execute("""SELECT id, ix, x,y,z, kind, severity, note, created_at
                          FROM hazards WHERE model_id=? ORDER BY id DESC""", (DEFAULT_MODEL_ID,)).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.post("/api/hazards/<mid>")
def api_hazards_add(mid: str):
    m = get_default_model()
    if not m: return jsonify(error="model not found"), 404
    user_id = session.get("user_id")
    if not user_id:
        return jsonify(error="login required"), 401
    P = m["P"]
    data = request.get_json(force=True) or {}
    ix = data.get("index")
    if ix is not None:
        ix = int(ix)
        x,y,z = map(float, P[ix])
    else:
        x = float(data.get("x")); y = float(data.get("y")); z = float(data.get("z"))
        ix = int(cKDTree(P).query([x,y,z])[1])
    kind = str(data.get("kind","hazard"))
    sev  = int(data.get("severity", 3))
    note = str(data.get("note",""))
    conn = db(); cur = conn.cursor()
    cur.execute("""INSERT INTO hazards(model_id,ix,x,y,z,kind,severity,note,user_id)
                   VALUES(?,?,?,?,?,?,?,?,?)""", (DEFAULT_MODEL_ID, ix, x,y,z, kind, sev, note, user_id))
    conn.commit(); hid = cur.lastrowid; conn.close()
    return jsonify(id=int(hid))


@app.delete("/api/hazards/<mid>/<int:hid>")
def api_hazards_del(mid: str, hid: int):
    conn = db(); cur = conn.cursor()
    cur.execute("DELETE FROM hazards WHERE model_id=? AND id=?", (DEFAULT_MODEL_ID,hid))
    conn.commit(); conn.close()
    return jsonify(ok=True)

# ----------------------------- Area Stats ---------------------------- #

@app.post("/api/area_stats/<mid>")
def api_area_stats(mid: str):
    m = get_default_model()
    if not m: return jsonify(error="model not found"), 404
    data = request.get_json(force=True) or {}
    inds = data.get("indices")
    if not isinstance(inds, list) or len(inds)==0:
        return jsonify(error="indices required"), 400
    inds = np.array(inds, dtype=np.int64)
    P, S, idx = m["P"], m["slope"], m["idx"]
    pts = P[inds]
    elev = pts[:,2]
    elev_min, elev_max = float(elev.min()), float(elev.max())
    slope_vals = S[inds]
    s_avg = float(slope_vals.mean()); s_p95 = float(np.percentile(slope_vals,95))
    area_m2 = 0.0
    if idx is not None and len(idx)>0:
        sel = set(int(i) for i in inds.tolist())
        tris = idx.reshape(-1,3)
        mask = np.array([a in sel and b in sel and c in sel for a,b,c in tris])
        tsel = tris[mask]
        if len(tsel)>0:
            v1 = P[tsel[:,1]] - P[tsel[:,0]]
            v2 = P[tsel[:,2]] - P[tsel[:,0]]
            area_m2 = float(0.5*np.linalg.norm(np.cross(v1,v2), axis=1).sum())
    else:
        if len(pts)>=3:
            hull = ConvexHull(pts[:,:2])
            area_m2 = float(hull.volume)  # 2D hull area
    return jsonify(count=int(len(inds)), elevMin=elev_min, elevMax=elev_max,
                   slopeAvg=s_avg, slopeP95=s_p95, areaM2=area_m2)

# ----------------------------- Segmentation -------------------------- #
@app.post("/api/segment/<mid>")
def api_segment(mid: str):
    m = MODELS.get(mid)
    if not m: return jsonify(error="model not found"), 404
    data = request.get_json(force=True) or {}
    k = int(data.get("k", 4))
    use = data.get("features", ["elev","slope","nx","ny","nz"])
    P, N, S = m["P"], m["normals"], m["slope"]
    feats = []
    if "elev" in use: feats.append(P[:,2:3])
    if "slope" in use: feats.append(S.reshape(-1,1))
    if "nx" in use or "ny" in use or "nz" in use:
        take = [i for i,f in enumerate(["nx","ny","nz"]) if f in use]
        feats.append(N[:, take])
    X = np.concatenate(feats, axis=1).astype(np.float32)
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True)+1e-6)
    L,_ = kmeans_numpy(X, k=k, iters=30)
    lab_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{mid}_seg.npy")
    np.save(lab_path, L)
    conn = db(); cur = conn.cursor()
    cur.execute("INSERT INTO segments(model_id,k,labels_path) VALUES(?,?,?) ON CONFLICT(model_id) DO UPDATE SET k=excluded.k, labels_path=excluded.labels_path",
                (mid, k, lab_path))
    conn.commit(); conn.close()
    return jsonify(k=k, ok=True)

@app.get("/api/segment/<mid>")
def api_segment_get(mid: str):
    conn = db(); cur = conn.cursor()
    row = cur.execute("SELECT k, labels_path FROM segments WHERE model_id=?", (mid,)).fetchone()
    conn.close()
    if not row: return jsonify(error="no segmentation"), 404
    L = np.load(row["labels_path"]).astype(int).tolist()
    return jsonify(k=int(row["k"]), labels=L)
# Add these endpoints to your app.py file

@app.get("/api/positions/<mid>")
def api_positions(mid: str):
    """Return vertex positions as binary data"""
    m = get_default_model()
    if not m:
        return jsonify(error="model not found"), 404
    positions = m["P"].astype(np.float32)
    return positions.tobytes(), 200, {'Content-Type': 'application/octet-stream'}

@app.get("/api/indices/<mid>")
def api_indices(mid: str):
    """Return face indices as binary data"""
    m = get_default_model()
    if not m:
        return jsonify(error="model not found"), 404
    if m["idx"] is None:
        return b'', 200, {'Content-Type': 'application/octet-stream'}
    indices = m["idx"].astype(np.uint32)
    return indices.tobytes(), 200, {'Content-Type': 'application/octet-stream'}

@app.get("/api/normals/<mid>")
def api_normals(mid: str):
    """Return vertex normals as binary data"""
    m = get_default_model()
    if not m:
        return jsonify(error="model not found"), 404
    if m["normals"] is None:
        return b'', 200, {'Content-Type': 'application/octet-stream'}
    normals = m["normals"].astype(np.float32)
    return normals.tobytes(), 200, {'Content-Type': 'application/octet-stream'}

@app.get("/api/slope/<mid>")
def api_slope(mid: str):
    """Return slope data as binary data"""
    m = get_default_model()
    if not m:
        return jsonify(error="model not found"), 404
    if m["slope"] is None:
        return b'', 200, {'Content-Type': 'application/octet-stream'}
    slope = m["slope"].astype(np.float32)
    return slope.tobytes(), 200, {'Content-Type': 'application/octet-stream'}

@app.get("/api/geometry/<mid>")
def api_geometry(mid: str):
    """Return all geometry data as JSON (alternative to binary endpoints)"""
    m = get_default_model()
    if not m:
        return jsonify(error="model not found"), 404
    
    result = {
        "positions": m["P"].astype(np.float32).tolist(),
        "count": int(m["P"].shape[0]),
        "bounds": m["bounds"]
    }
    
    if m["idx"] is not None:
        result["indices"] = m["idx"].astype(np.uint32).tolist()
    
    if m["normals"] is not None:
        result["normals"] = m["normals"].astype(np.float32).tolist()
    
    if m["slope"] is not None:
        result["slope"] = m["slope"].astype(np.float32).tolist()
    
    return jsonify(result)
@app.post("/api/segment/<mid>/paint")
def api_segment_paint(mid: str):
    data = request.get_json(force=True) or {}
    inds = data.get("indices"); lab = int(data.get("label",0))
    if not isinstance(inds,list) or len(inds)==0: return jsonify(error="indices required"), 400
    conn = db(); cur = conn.cursor()
    row = cur.execute("SELECT labels_path FROM segments WHERE model_id=?", (mid,)).fetchone()
    conn.close()
    if not row: return jsonify(error="no segmentation"), 404
    L = np.load(row["labels_path"])
    inds = np.array(inds, dtype=np.int64)
    L[inds] = lab
    np.save(row["labels_path"], L)
    return jsonify(ok=True)

@app.post("/api/segment/<mid>/paint_polygon")
def api_segment_paint_polygon(mid: str):
    m = MODELS.get(mid)
    if not m:
        return jsonify(error="model not found"), 404

    data = request.get_json(force=True) or {}
    poly = data.get("polygon")
    lab  = int(data.get("label", 0))
    if not poly or len(poly) < 3:
        return jsonify(error="polygon with 3+ points required"), 400

    import numpy as np
    from matplotlib.path import Path as MplPath

    poly = np.asarray(poly, float)
    path = MplPath(poly)

    P   = m["P"]             # (N,3) vertices
    idx = m["idx"]           # (3*T,) faces or None

    # --- load or init labels ---
    conn = db(); cur = conn.cursor()
    row = cur.execute("SELECT labels_path FROM segments WHERE model_id=?", (mid,)).fetchone()
    if row:
        lab_path = row["labels_path"]
        L = np.load(lab_path)
        if L.shape[0] != P.shape[0]:
            L = np.resize(L, P.shape[0])
    else:
        lab_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{mid}_seg.npy")
        L = np.zeros(P.shape[0], dtype=np.int32)

    painted_vertices = 0
    painted_faces = 0

    if idx is not None and len(idx) > 0:
        # ---------- TRIANGLE-MESH: paint by face centroid inside polygon ----------
        tris = idx.reshape(-1, 3)
        cent_xy = P[tris].mean(axis=1)[:, :2]               # (T, 2)
        inside_face = path.contains_points(cent_xy)         # (T,)
        faces_sel = tris[inside_face]
        if faces_sel.size > 0:
            verts_sel = np.unique(faces_sel.reshape(-1))
            L[verts_sel] = lab
            painted_vertices = int(verts_sel.size)
            painted_faces = int(faces_sel.shape[0])
    else:
        # ---------- POINT CLOUD: paint only "surface" points (top-of-cell) ----------
        xy = P[:, :2]
        inside = path.contains_points(xy)                   # (N,)
        if inside.any():
            minx, miny = xy.min(axis=0)
            maxx, maxy = xy.max(axis=0)
            spanx = float(maxx - minx)
            spany = float(maxy - miny)
            longest = max(spanx, spany)

            # target ~128 cells across the longest axis, clamped
            target_cells = 128
            nx = max(32, min(512, int(np.ceil(spanx / (longest / target_cells + 1e-9)))))
            ny = max(32, min(512, int(np.ceil(spany / (longest / target_cells + 1e-9)))))
            nx = max(nx, 1); ny = max(ny, 1)

            cellx = spanx / nx if nx > 0 else 1.0
            celly = spany / ny if ny > 0 else 1.0
            cellx = cellx if cellx > 0 else 1.0
            celly = celly if celly > 0 else 1.0

            ix = np.clip(((xy[:, 0] - minx) / cellx).astype(np.int32), 0, nx - 1)
            iy = np.clip(((xy[:, 1] - miny) / celly).astype(np.int32), 0, ny - 1)
            bin_id = ix + nx * iy

            z = P[:, 2]
            inside_idx = np.where(inside)[0]
            if inside_idx.size > 0:
                bi = bin_id[inside_idx]

                order = np.argsort(bi, kind="stable")
                bi_sorted = bi[order]
                idx_sorted = inside_idx[order]

                best = {}
                cur_bin = None
                best_i = None
                best_z = -np.inf
                for b, i in zip(bi_sorted, idx_sorted):
                    zi = z[i]
                    if cur_bin is None:
                        cur_bin, best_i, best_z = b, i, zi
                    elif b != cur_bin:
                        best[cur_bin] = best_i
                        cur_bin, best_i, best_z = b, i, zi
                    else:
                        if zi > best_z:
                            best_i, best_z = i, zi
                if cur_bin is not None:
                    best[cur_bin] = best_i

                top_idx = np.fromiter(best.values(), dtype=np.int64)
                if top_idx.size > 0:
                    L[top_idx] = lab
                    painted_vertices = int(top_idx.size)

    np.save(lab_path, L)
    if not row:
        cur.execute(
            "INSERT INTO segments(model_id,k,labels_path) VALUES(?,?,?)",
            (mid, int(L.max() + 1), lab_path)
        )
    conn.commit(); conn.close()

    return jsonify(ok=True, paintedVertices=painted_vertices, paintedFaces=painted_faces)

# ------------------------------- Meta -------------------------------- #
@app.get("/api/meta/<mid>")
def api_meta(mid: str):
    m = MODELS.get(mid)
    if not m: return jsonify(error="not found"), 404
    return jsonify(count=int(m["P"].shape[0]), has_indices=bool(m["idx"] is not None), bounds=m["bounds"])


# ------------------------------- Run -------------------------------- #
init_db()
load_default_model()
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
