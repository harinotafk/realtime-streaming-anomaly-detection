"""
Real-Time Streaming Anomaly Detection - single-file edition
===========================================================
Dataset : Kaggle "Credit Card Fraud Detection" (mlg-ulb/creditcardfraud)
Model   : Isolation Forest + sliding window + adaptive threshold +
          background retraining (concept-drift handling)

Layers in this one file
  1. Data      : Kaggle loader / synthetic fallback / feature prep
  2. Model     : StreamingDetector
  3. Pipeline  : producer thread -> bounded queue -> micro-batch consumer
  4. Dashboard : stdlib HTTP + Server-Sent Events + embedded HTML/JS UI
  5. CLI       : `dashboard` (default) or `headless` report mode

Usage
  python stream_anomaly.py                          # dashboard, credit-card fraud via kagglehub
  python stream_anomaly.py --synthetic              # offline demo
  python stream_anomaly.py --csv my_data.csv        # ANY CSV (columns auto-detected)
  python stream_anomaly.py --csv my.csv --label status --anomaly-value fault \\
                           --time-col timestamp --features temp,vibration,pressure
  python stream_anomaly.py --kaggle owner/dataset --kaggle-file data.csv
  python stream_anomaly.py --csv nab/data/realKnownCause/machine_temperature_system_failure.csv \\
                           --labels-json nab/labels/combined_windows.json --contamination 0.01
  python stream_anomaly.py --headless --csv my.csv  # console report only
"""
import argparse
import json
import os
import queue
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# 1-3. Data, model, pipeline
# --------------------------------------------------------------------------- #
SENTINEL = None


# --------------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------------- #
LABEL_NAMES = ["Class", "class", "label", "Label", "is_anomaly", "anomaly", "Anomaly",
               "target", "Target", "is_fraud", "isFraud", "fraud", "Fraud", "outlier", "y"]
TIME_NAMES = ["Time", "time", "timestamp", "Timestamp", "datetime", "DateTime",
              "date", "Date", "ts"]
POSITIVE = {"1", "1.0", "true", "yes", "anomaly", "anomalous", "fraud", "attack",
            "outlier", "abnormal", "bad", "malicious"}


def load_data(args):
    """Load any CSV: local --csv, a Kaggle dataset via --kaggle owner/name,
    or (default) the credit-card fraud dataset."""
    if args.synthetic:
        return make_synthetic()
    path = args.csv
    if path is None and not args.kaggle:
        for p in ("data/creditcard.csv", "creditcard.csv"):
            if os.path.exists(p):
                path = p
                break
    if path is None:
        slug = args.kaggle or "mlg-ulb/creditcardfraud"
        try:
            import kagglehub  # pip install kagglehub (needs Kaggle credentials)
            folder = kagglehub.dataset_download(slug)
        except Exception as e:
            raise SystemExit(
                f"Could not download '{slug}' ({e}).\nDownload the CSV manually "
                "from Kaggle and run: python stream_anomaly.py --csv path/to/file.csv")
        csvs = [os.path.join(r, f) for r, _, fs in os.walk(folder)
                for f in fs if f.lower().endswith(".csv")]
        if not csvs:
            raise SystemExit(f"No CSV files found in {folder}")
        if args.kaggle_file:
            csvs = [c for c in csvs if os.path.basename(c) == args.kaggle_file]
            if not csvs:
                raise SystemExit(f"{args.kaggle_file} not found in {folder}")
        path = max(csvs, key=os.path.getsize)
    print(f"[data] loading {path}")
    return pd.read_csv(path)


def make_synthetic(n=60000, fraud_rate=0.002, seed=7):
    """Offline stand-in with the same columns as the Kaggle dataset."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 28))
    amount = rng.lognormal(3.2, 1.2, n)
    y = (rng.random(n) < fraud_rate).astype(int)
    idx = np.where(y == 1)[0]
    X[idx] += rng.normal(0, 1, (len(idx), 28)) * 3 + rng.choice([-3, 3], (len(idx), 28))
    amount[idx] *= rng.uniform(2, 8, len(idx))
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.sort(rng.uniform(0, 172800, n)))
    df["Amount"] = amount
    df["Class"] = y
    return df


LABELS_URL = "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_windows.json"


def _json_has_key(path, key):
    try:
        with open(path) as f:
            d = json.load(f)
        return isinstance(d, dict) and (key is None or key in d
                                        or any(k.endswith("/" + str(key)) for k in d))
    except Exception:
        return False


def find_labels(csv_path, key=None):
    """Look for combined_windows.json near the CSV (parent folders, depth<=3)
    and in the current directory. Returns a path or None."""
    roots, base = [os.getcwd()], os.path.dirname(os.path.abspath(csv_path or "."))
    for _ in range(4):
        roots.append(base)
        base = os.path.dirname(base)
    found = []
    for root in roots:
        for r, dirs, files in os.walk(root):
            if r[len(root):].count(os.sep) >= 3:
                dirs[:] = []
            for name in ("combined_windows.json", "combined_labels.json"):
                if name in files:
                    p = os.path.join(r, name)
                    if p not in [f[1] for f in found] and _json_has_key(p, key):
                        found.append((name != "combined_windows.json", p))
    return min(found)[1] if found else None   # prefer windows over point labels


def download_labels(dest="combined_windows.json"):
    import urllib.request
    print(f"[data] downloading NAB labels from GitHub -> {dest}")
    try:
        urllib.request.urlretrieve(LABELS_URL, dest)
    except Exception as e:
        raise SystemExit(f"Download failed ({e}).\nGet it manually from "
                         "https://github.com/numenta/NAB/tree/master/labels "
                         "(file: combined_windows.json) and pass --labels-json <path>.")
    return dest


def resolve_labels(args, key, has_label_col=False):
    """Decide which labels JSON to use (explicit > auto-discovered > download)."""
    lj = args.labels_json
    if lj and os.path.exists(lj):
        return lj
    if lj:
        print(f"[data] labels file not found at '{lj}' - searching nearby...")
    if args.download_labels:
        return download_labels()
    if args.csv and not has_label_col:
        found = find_labels(args.csv, key)
        if found:
            print(f"[data] using labels file: {found}")
            return found
    if lj:
        raise SystemExit(
            f"Could not find '{lj}' or any combined_windows.json / combined_labels.json near your CSV.\n"
            "Fix: re-run with --download-labels (fetches it from GitHub), or download "
            "labels/combined_windows.json from https://github.com/numenta/NAB and pass its path.\n"
            "To search your disk:  PowerShell: Get-ChildItem -Recurse -Filter combined_windows.json")
    return None


def load_window_labels(path, key, timestamps):
    """NAB-style labels JSON: {"dir/file.csv": [[start, end], ...]} (windows)
    or {"dir/file.csv": ["ts1", "ts2", ...]} (point labels)."""
    with open(path) as f:
        d = json.load(f)
    k = key if key in d else next((x for x in d if x.endswith("/" + str(key))), None)
    if k is None:
        raise SystemExit(f"Key '{key}' not in {path}. Example keys: {list(d)[:4]}")
    ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))
    y = np.zeros(len(ts), dtype=bool)
    for item in d[k]:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            y |= ((ts >= pd.to_datetime(item[0])) & (ts <= pd.to_datetime(item[1]))).to_numpy()
        else:
            y |= (ts == pd.to_datetime(item)).to_numpy()
    print(f"[data] labels from {os.path.basename(path)}['{k}']: "
          f"{len(d[k])} anomaly entries -> {int(y.sum()):,} labelled rows")
    return y.astype(int)


def ts_features(F, windows=(12, 48)):
    """Causal rolling features (only use past data, so valid for streaming):
    value, first difference, deviation from rolling mean, rolling std, z-score."""
    out = {}
    for c in F.columns:
        v = F[c]
        eps = 1e-6 * (v.std() + 1e-12)
        out[f"{c}"] = v
        out[f"{c}_diff"] = v.diff()
        for w in windows:
            r = v.rolling(w, min_periods=2)
            dev = v - r.mean()
            sd = r.std()
            out[f"{c}_dev{w}"] = dev
            out[f"{c}_std{w}"] = sd
            out[f"{c}_z{w}"] = dev / (sd + eps)
    G = pd.DataFrame(out).replace([np.inf, -np.inf], np.nan)
    return G.bfill().fillna(0.0)


def prepare(df, label=None, time_col=None, features=None,
            anomaly_value=None, log_cols=None, timeseries=None,
            ts_windows=(12, 48), labels_json=None, nab_key=None):
    """Turn any tabular DataFrame into (X, y, meta).

    * label    : ground-truth column (auto-detected; optional - without one the
                 pipeline runs unsupervised and quality metrics are skipped)
    * time_col : used only to order the stream (auto-detected, optional)
    * features : numeric columns to model (default: all numeric, non-constant)
    * log_cols : columns to log1p-transform (default: auto for skewed >=0 cols)
    """
    label = label or next((c for c in LABEL_NAMES if c in df.columns), None)
    if label and label not in df.columns:
        raise SystemExit(f"Label column '{label}' not found. Columns: {list(df.columns)}")
    time_col = time_col or next((c for c in TIME_NAMES if c in df.columns), None)

    if time_col:
        t = df[time_col]
        if not pd.api.types.is_numeric_dtype(t):
            t = pd.to_datetime(t, errors="coerce")
        df = df.assign(_t=t).sort_values("_t", kind="stable").drop(columns="_t")
    df = df.reset_index(drop=True)

    # --- labels -----------------------------------------------------------
    if labels_json:
        if not time_col:
            raise SystemExit("--labels-json needs a timestamp column (use --time-col).")
        y = load_window_labels(labels_json, nab_key, df[time_col])
        label = f"{os.path.basename(labels_json)}:{nab_key}"
    elif label:
        low = df[label].astype(str).str.strip().str.lower()
        if anomaly_value is not None:
            y = (low == str(anomaly_value).strip().lower()).to_numpy()
        else:
            y = low.isin(POSITIVE).to_numpy()
            if y.sum() == 0:
                raise SystemExit(
                    f"Couldn't tell which value of '{label}' means anomaly "
                    f"(values: {sorted(low.unique())[:8]}). Use --anomaly-value.")
        y = y.astype(int)
    else:
        y = np.zeros(len(df), dtype=int)

    # --- features ---------------------------------------------------------
    drop = {c for c in (label, time_col) if c}
    if features:
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise SystemExit(f"Feature columns not found: {missing}")
        cols = list(features)
    else:
        cols = [c for c in df.columns
                if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    F = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    F = F.fillna(F.median()).fillna(0.0)
    cols = [c for c in cols if F[c].std() > 0]            # drop constant columns
    F = F[cols]
    if not cols:
        raise SystemExit("No usable numeric feature columns found.")

    if timeseries is None:                       # auto: one signal + a time axis
        timeseries = len(cols) == 1 and bool(time_col)
        if timeseries:
            print("[data] single numeric signal detected -> time-series mode "
                  "(rolling features). Use --no-timeseries to disable.")
    if timeseries:
        F = ts_features(F, ts_windows)
        cols = list(F.columns)
        log_cols = log_cols or []
    if log_cols is None:
        log_cols = [c for c in cols if F[c].min() >= 0 and abs(F[c].skew()) > 5]
    for c in log_cols:
        F[c] = np.log1p(F[c].clip(lower=0))

    meta = dict(label=label, time_col=time_col, features=cols, log_cols=list(log_cols),
                has_labels=bool(label), timeseries=bool(timeseries))
    print(f"[data] {len(F):,} rows | {len(cols)} features | label={label} | "
          f"time={time_col} | log1p on {list(log_cols) or 'none'}")
    if label:
        print(f"[data] anomaly rate {y.mean():.4%} ({int(y.sum()):,} anomalies)")
    else:
        print("[data] no label column found - running unsupervised "
              "(quality metrics disabled)")
    return F.to_numpy(dtype=np.float64), y, meta


def setup_data(args):
    df = load_data(args)
    feats = [f.strip() for f in args.features.split(",")] if args.features else None
    logc = None
    if args.log_cols is not None:
        logc = [] if args.log_cols == "" else [c.strip() for c in args.log_cols.split(",")]
    ts = False if args.no_timeseries else (True if args.timeseries else None)
    wins = tuple(int(w) for w in args.ts_windows.split(","))
    key = args.nab_key or (os.path.basename(args.csv) if args.csv else None)
    has_col = bool(args.label) or any(c in df.columns for c in LABEL_NAMES)
    labels_path = resolve_labels(args, key, has_col)
    X, y, meta = prepare(df, args.label, args.time_col, feats, args.anomaly_value, logc,
                         ts, wins, labels_path, key)
    init = args.init_size
    if len(X) < init * 3:                     # leave room for a real stream
        init = max(200, len(X) // 3)
        print(f"[data] small dataset: warm-start size reduced to {init:,} rows")
    # make sure periodic retraining actually happens on short streams
    args.retrain_every = min(args.retrain_every, max(500, (len(X) - init) // 3))
    return X, y, meta, init


# --------------------------------------------------------------------------- #
# 2. Streaming model: sliding window + adaptive threshold + retraining
# --------------------------------------------------------------------------- #
class StreamingDetector:
    def __init__(self, X_init, window=20000, retrain_every=10000,
                 contamination=0.002, n_estimators=150, seed=42):
        self.window = deque(maxlen=window)      # training data: accepted-as-normal only
        self.calib = deque(maxlen=window)       # ALL recent events: used only to set the threshold
        self.retrain_every = retrain_every
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.seed = seed
        self.since_retrain = 0
        self.n_retrains = 0
        self._thread, self._result = None, []
        self.window.extend(X_init)
        self.calib.extend(X_init)
        self._fit()

    def _build(self, data, calib):
        scaler = StandardScaler().fit(data)
        Z = scaler.transform(data)
        model = IsolationForest(n_estimators=self.n_estimators,
                                contamination=self.contamination,
                                random_state=self.seed, n_jobs=-1).fit(Z)
        # Threshold = (1 - contamination) quantile of scores over ALL recent events
        # (flagged ones included). Calibrating on the filtered training window made
        # the threshold ratchet down at every retrain and precision collapse.
        thr = np.quantile(-model.score_samples(scaler.transform(calib)),
                          1 - self.contamination)
        return scaler, model, thr

    def _fit(self):
        self.scaler, self.model, self.threshold = self._build(
            np.asarray(self.window), np.asarray(self.calib))
        self.since_retrain = 0
        self.n_retrains += 1

    def score(self, X):
        """Anomaly score: higher = more anomalous."""
        return -self.model.score_samples(self.scaler.transform(X))

    def update(self, X, is_anomaly):
        """Add *accepted-as-normal* events to the window (prevents the model
        from learning anomalies as normal). Retraining runs in a background
        thread so scoring never stalls; the new model is swapped in by this
        (the scoring) thread once ready. Returns True when a swap happened."""
        swapped = False
        t = self._thread
        if t is not None and not t.is_alive():
            self._thread = None
            if self._result:                       # retrain succeeded
                self.scaler, self.model, self.threshold = self._result[0]
                swapped = True
                self.n_retrains += 1
            else:
                print("[model] background retrain failed; keeping current model")

        self.window.extend(X[~is_anomaly])
        self.calib.extend(X)
        self.since_retrain += len(X)
        if self.since_retrain >= self.retrain_every and self._thread is None:
            data, calib = np.asarray(self.window), np.asarray(self.calib)
            self.since_retrain = 0
            self._result = []
            self._thread = threading.Thread(
                target=lambda: self._result.append(self._build(data, calib)), daemon=True)
            self._thread.start()
        return swapped


# --------------------------------------------------------------------------- #
# 3. Producer / consumer
# --------------------------------------------------------------------------- #
def producer(q, X, y, start, rate, max_events):
    """Replays rows as a live stream. rate = events/sec (0 = unthrottled)."""
    end = len(X) if max_events is None else min(len(X), start + max_events)
    t0 = time.perf_counter()
    for i in range(start, end):
        if rate > 0:
            sleep = (i - start) / rate - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)
        q.put((i, time.perf_counter(), X[i], y[i]))  # blocks if queue is full
    q.put(SENTINEL)


def consumer(q, det, batch_size, max_wait, log_every, alerts, results):
    seen = 0
    t_start = time.perf_counter()
    done = False
    while not done:
        batch = []
        deadline = time.perf_counter() + max_wait
        while len(batch) < batch_size:
            timeout = deadline - time.perf_counter()
            if timeout <= 0:
                break
            try:
                item = q.get(timeout=timeout)
            except queue.Empty:
                break
            if item is SENTINEL:
                done = True
                break
            batch.append(item)
        if not batch:
            continue

        ids = np.array([b[0] for b in batch])
        t_in = np.array([b[1] for b in batch])
        X = np.vstack([b[2] for b in batch])
        y = np.array([b[3] for b in batch])

        scores = det.score(X)
        flags = scores > det.threshold
        t_out = time.perf_counter()

        results["id"].extend(ids)
        results["score"].extend(scores)
        results["flag"].extend(flags)
        results["label"].extend(y)
        results["latency_ms"].extend((t_out - t_in) * 1000)

        for i in np.where(flags)[0]:
            alerts.append((int(ids[i]), float(scores[i]), int(y[i])))

        if det.update(X, flags):
            print(f"[model] retrained (#{det.n_retrains}) "
                  f"threshold={det.threshold:.4f}")

        seen += len(batch)
        if seen // log_every != (seen - len(batch)) // log_every:
            el = time.perf_counter() - t_start
            print(f"[stream] {seen:>8,} events | {seen/el:>9,.0f} ev/s | "
                  f"alerts={len(alerts):,} | queue={q.qsize()}")


# --------------------------------------------------------------------------- #
# 4a. Embedded UI (served at /)
# --------------------------------------------------------------------------- #
INDEX_HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Streaming Anomaly Detection</title>
<style>
:root{
  --bg:#f5f6f8; --card:#fff; --fg:#1b1f27; --muted:#6b7280; --line:#e5e7eb;
  --blue:#3b82f6; --red:#ef4444; --amber:#f59e0b; --green:#10b981; --grid:#eceef2;
}
@media (prefers-color-scheme:dark){
  :root{--bg:#0f1218; --card:#171b24; --fg:#e6e9ef; --muted:#8b93a3; --line:#262c38; --grid:#222836;}
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;padding:14px 20px;border-bottom:1px solid var(--line);background:var(--card)}
h1{font-size:17px;margin:0;font-weight:650}
.sub{color:var(--muted);font-size:12px}
.pill{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:99px;font-size:12px;font-weight:600;border:1px solid var(--line)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--muted)}
.running .dot{background:var(--green);animation:pulse 1.2s infinite}
.error .dot{background:var(--red)} .paused .dot{background:var(--amber)} .finished .dot{background:var(--blue)}
@keyframes pulse{50%{opacity:.3}}
.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center}
button{font:inherit;padding:6px 12px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button:hover{border-color:var(--blue)}
button.primary{background:var(--blue);border-color:var(--blue);color:#fff}
button.danger{border-color:var(--red);color:var(--red)}
.rate{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--muted)}
input[type=range]{width:150px}
main{padding:16px 20px;display:grid;gap:14px;max-width:1400px;margin:auto}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.kpi .l{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.kpi .v{font-size:22px;font-weight:650;margin-top:2px;font-variant-numeric:tabular-nums}
.grid2{display:grid;grid-template-columns:2fr 1fr;gap:14px}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}
@media(max-width:900px){.grid2,.grid3{grid-template-columns:1fr}}
.card h2{font-size:13px;margin:0 0 8px;font-weight:600}
.card h2 span{color:var(--muted);font-weight:400}
canvas{width:100%;height:210px;display:block}
canvas.sm{height:150px}
table{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
th{text-align:left;color:var(--muted);font-weight:500;padding:4px 6px;border-bottom:1px solid var(--line)}
td{padding:4px 6px;border-bottom:1px solid var(--line)}
.tbl{max-height:260px;overflow:auto}
.tag{padding:1px 8px;border-radius:99px;font-size:11px;font-weight:600}
.tag.f{background:rgba(239,68,68,.15);color:var(--red)}
.tag.n{background:rgba(245,158,11,.18);color:var(--amber)}
.cm{display:grid;grid-template-columns:auto 1fr 1fr;gap:4px;font-size:12px;text-align:center;margin-top:4px}
.cm div{padding:10px 4px;border-radius:8px;background:var(--bg)}
.cm .h{background:none;color:var(--muted);padding:2px}
.cm .good{outline:1px solid var(--green)} .cm .bad{outline:1px solid var(--red)}
.cm b{display:block;font-size:18px}
.log{font-size:12.5px;max-height:190px;overflow:auto}
.log div{padding:3px 0;border-bottom:1px solid var(--line)} .log time{color:var(--muted);margin-right:8px}
.legend{display:flex;gap:14px;font-size:11px;color:var(--muted);margin-top:6px;flex-wrap:wrap}
.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}
</style>
</head>
<body>
<header>
  <div>
    <h1>Real-Time Streaming Anomaly Detection</h1>
    <div class="sub">Isolation Forest &middot; sliding window &middot; adaptive threshold &middot; periodic retraining</div>
  </div>
  <div class="controls">
    <span id="status" class="pill"><span class="dot"></span><span id="statusText">connecting</span></span>
    <button id="btnPause" class="primary">Pause</button>
    <div class="rate">Rate <input id="rate" type="range" min="0" max="1000" value="400"> <b id="rateVal" style="color:var(--fg);min-width:80px">–</b></div>
    <label class="rate" style="cursor:pointer"><input id="loop" type="checkbox"> Loop</label>
    <button id="btnInject" class="danger">Inject anomaly</button>
    <button id="btnRestart">Restart</button>
  </div>
</header>

<main>
  <section class="kpis">
    <div class="card kpi"><div class="l">Events</div><div class="v" id="kN">0</div></div>
    <div class="card kpi"><div class="l">Throughput</div><div class="v" id="kTp">0</div></div>
    <div class="card kpi"><div class="l">Alerts</div><div class="v" id="kAl">0</div></div>
    <div class="card kpi"><div class="l">Precision</div><div class="v" id="kP">–</div></div>
    <div class="card kpi"><div class="l">Recall</div><div class="v" id="kR">–</div></div>
    <div class="card kpi"><div class="l">F1</div><div class="v" id="kF">–</div></div>
    <div class="card kpi"><div class="l">Latency p95</div><div class="v" id="kL">0</div></div>
    <div class="card kpi"><div class="l">Queue</div><div class="v" id="kQ">0</div></div>
    <div class="card kpi"><div class="l">Threshold</div><div class="v" id="kT">–</div></div>
    <div class="card kpi"><div class="l">Retrains</div><div class="v" id="kRe">0</div></div>
  </section>

  <section class="grid2">
    <div class="card">
      <h2>Anomaly score <span>(max per 0.5 s, last 60 s)</span></h2>
      <canvas id="cScore"></canvas>
      <div class="legend">
        <span><i style="background:var(--red)"></i>true anomaly caught</span>
        <span><i style="background:var(--amber)"></i>false alarm</span>
        <span><i style="background:var(--muted)"></i>dashed = threshold</span>
      </div>
    </div>
    <div class="card">
      <h2>Score distribution <span>(log count)</span></h2>
      <canvas id="cHist"></canvas>
      <div class="legend">
        <span><i style="background:var(--blue)"></i>normal</span>
        <span><i style="background:var(--red)"></i>anomaly (label)</span>
      </div>
    </div>
  </section>

  <section class="grid3">
    <div class="card"><h2>Throughput <span>(events/s)</span></h2><canvas id="cTp" class="sm"></canvas></div>
    <div class="card"><h2>End-to-end latency p95 <span>(ms)</span></h2><canvas id="cLat" class="sm"></canvas></div>
    <div class="card">
      <h2>Confusion matrix</h2>
      <div class="cm">
        <div class="h"></div><div class="h">Flagged</div><div class="h">Not flagged</div>
        <div class="h">Fraud</div><div class="good"><b id="cTP">0</b>TP</div><div class="bad"><b id="cFN">0</b>FN</div>
        <div class="h">Normal</div><div class="bad"><b id="cFP">0</b>FP</div><div class="good"><b id="cTN">0</b>TN</div>
      </div>
    </div>
  </section>

  <section class="grid2">
    <div class="card">
      <h2>Live alerts <span>(latest 40)</span></h2>
      <div class="tbl"><table>
        <thead><tr><th>Time</th><th>Event ID</th><th>Score</th><th>Ground truth</th></tr></thead>
        <tbody id="alerts"></tbody>
      </table></div>
    </div>
    <div class="card">
      <h2>System log</h2>
      <div class="log" id="log"></div>
    </div>
  </section>
</main>

<script>
const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = (n, d=0) => Number(n).toLocaleString(undefined, {maximumFractionDigits:d});
const pct = x => (x*100).toFixed(1) + '%';
let state = null;

/* ---------- controls ---------- */
const post = (action, value) => fetch('/control', {method:'POST',
  headers:{'Content-Type':'application/json'}, body:JSON.stringify({action, value})});
const sliderToRate = v => Math.round(Math.pow(10, 1 + v/1000*3.7));  // 10 .. ~50k
$('rate').addEventListener('input', e => {
  const r = sliderToRate(+e.target.value); $('rateVal').textContent = fmt(r) + ' ev/s'; post('rate', r);
});
$('loop').addEventListener('change', e => post('loop', e.target.checked));
const rateToSlider = r => Math.round((Math.log10(Math.max(10, r)) - 1) / 3.7 * 1000);
$('btnPause').onclick = () => post(state && state.status === 'paused' ? 'resume' : 'pause');
$('btnInject').onclick = () => post('inject');
$('btnRestart').onclick = () => post('restart');

/* ---------- chart helpers ---------- */
function setup(c){
  const r = c.getBoundingClientRect(), d = window.devicePixelRatio || 1;
  c.width = r.width*d; c.height = r.height*d;
  const g = c.getContext('2d'); g.setTransform(d,0,0,d,0,0);
  g.font = '11px sans-serif'; return [g, r.width, r.height];
}
function axes(g, W, H, ymax, pad, ticks=4, f=v=>v.toFixed(2)){
  g.strokeStyle = css('--grid'); g.fillStyle = css('--muted'); g.lineWidth = 1;
  for(let i=0;i<=ticks;i++){
    const y = pad.t + (H-pad.t-pad.b)*(1-i/ticks);
    g.beginPath(); g.moveTo(pad.l,y); g.lineTo(W-pad.r,y); g.stroke();
    g.textAlign='right'; g.fillText(f(ymax*i/ticks), pad.l-6, y+3);
  }
}
function lineChart(c, ys, o={}){
  const [g,W,H] = setup(c), pad={l:42,r:10,t:8,b:8};
  let ymax = Math.max(1e-9, ...ys, o.thr ? Math.max(...o.thr) : 0) * 1.1;
  axes(g,W,H,ymax,pad,4,o.f);
  if(!ys.length){ g.fillStyle=css('--muted'); g.textAlign='center'; g.fillText('Waiting for first events...', W/2, H/2); return; }
  const N = 120, X = i => pad.l + (W-pad.l-pad.r)*i/(N-1);
  const Y = v => pad.t + (H-pad.t-pad.b)*(1 - v/ymax);
  if(o.thr){
    g.setLineDash([5,4]); g.strokeStyle = css('--muted'); g.beginPath();
    o.thr.forEach((v,i)=> i?g.lineTo(X(i),Y(v)):g.moveTo(X(i),Y(v))); g.stroke(); g.setLineDash([]);
  }
  g.strokeStyle = o.color || css('--blue'); g.lineWidth = 1.8; g.beginPath();
  ys.forEach((v,i)=> i?g.lineTo(X(i),Y(v)):g.moveTo(X(i),Y(v))); g.stroke();
  if(o.fill){ g.lineTo(X(ys.length-1),Y(0)); g.lineTo(X(0),Y(0)); g.closePath();
    g.globalAlpha=.12; g.fillStyle=o.color||css('--blue'); g.fill(); g.globalAlpha=1; }
  if(o.marks) o.marks.forEach((m,i)=>{ if(!m) return;
    g.fillStyle = m==='tp'?css('--red'):css('--amber');
    g.beginPath(); g.arc(X(i),Y(ys[i]),4,0,7); g.fill(); });
}
function histChart(c, hist, range, thr){
  const [g,W,H] = setup(c), pad={l:34,r:8,t:8,b:22};
  const L = v=>Math.log1p(v), n = hist[0].length;
  const ymax = Math.max(1, ...hist[0].map(L), ...hist[1].map(L)) * 1.1;
  axes(g,W,H,ymax,pad,3,v=>fmt(Math.expm1(v)));
  const bw = (W-pad.l-pad.r)/n, base = H-pad.b;
  [[0,css('--blue'),.85],[1,css('--red'),.95]].forEach(([k,col,a])=>{
    g.fillStyle=col; g.globalAlpha=a;
    hist[k].forEach((v,i)=>{ const h=(base-pad.t)*L(v)/ymax; g.fillRect(pad.l+i*bw+1, base-h, bw-2, h); });
  }); g.globalAlpha=1;
  const tx = pad.l + (W-pad.l-pad.r)*(thr-range[0])/(range[1]-range[0]);
  g.setLineDash([5,4]); g.strokeStyle=css('--muted'); g.beginPath(); g.moveTo(tx,pad.t); g.lineTo(tx,base); g.stroke(); g.setLineDash([]);
  g.fillStyle=css('--muted'); g.textAlign='center';
  [0,.5,1].forEach(t=>g.fillText((range[0]+(range[1]-range[0])*t).toFixed(2), pad.l+(W-pad.l-pad.r)*t, H-6));
}

/* ---------- render ---------- */
function render(s){
  state = s;
  $('status').className = 'pill ' + s.status;
  $('statusText').textContent = s.status === 'finished' ? 'finished - press Restart or tick Loop' : s.status === 'error' ? 'error - see log / terminal' : s.status;
  if (document.activeElement !== $('rate')) {
    $('rate').value = rateToSlider(s.rate); $('rateVal').textContent = fmt(s.rate) + ' ev/s';
  }
  $('loop').checked = s.loop;
  $('btnPause').textContent = s.status==='paused' ? 'Resume' : 'Pause';
  $('kN').textContent = fmt(s.n); $('kTp').textContent = fmt(s.throughput)+'/s';
  $('kAl').textContent = fmt(s.alerts_total);
  const L = s.has_labels;
  $('kP').textContent = L && s.tp+s.fp ? pct(s.precision) : '–';
  $('kR').textContent = L && s.tp+s.fn ? pct(s.recall) : '–';
  $('kF').textContent = L && s.tp ? s.f1.toFixed(3) : '–';
  $('kL').textContent = fmt(s.p95,0)+' ms'; $('kQ').textContent = fmt(s.queue);
  $('kT').textContent = s.threshold.toFixed(3); $('kRe').textContent = s.retrains;
  const cm = v => L ? fmt(v) : 'n/a';
  $('cTP').textContent=cm(s.tp); $('cFN').textContent=cm(s.fn);
  $('cFP').textContent=cm(s.fp); $('cTN').textContent=cm(s.tn);

  const b = s.buckets;
  lineChart($('cScore'), b.map(x=>x.max), {thr:b.map(x=>x.thr),
    marks:b.map(x=>x.tp>0?'tp':(x.fp>0?'fp':null))});
  lineChart($('cTp'), b.map(x=>x.rate), {color:css('--green'), fill:true, f:v=>fmt(v)});
  lineChart($('cLat'), b.map(x=>x.p95), {color:css('--amber'), fill:true, f:v=>fmt(v)});
  histChart($('cHist'), s.hist, s.hist_range, s.threshold);

  $('alerts').innerHTML = s.alerts.map(a =>
    `<tr><td>${a.t}</td><td>${a.id<0?'injected':a.id}</td><td>${a.score.toFixed(3)}</td>
     <td><span class="tag ${!s.has_labels?'n':a.fraud?'f':'n'}">${!s.has_labels?'flagged':a.fraud?'anomaly':'false alarm'}</span></td></tr>`).join('')
    || '<tr><td colspan="4" style="color:var(--muted)">No alerts yet</td></tr>';
  $('log').innerHTML = s.log.map(l=>`<div><time>${l.t}</time>${l.msg}</div>`).join('');
}

const es = new EventSource('/stream');
es.onmessage = e => render(JSON.parse(e.data));
let lastMsg = Date.now();
es.addEventListener('message', () => { lastMsg = Date.now(); });
es.onerror = () => { $('statusText').textContent = 'reconnecting...'; };
setInterval(() => {                      // fallback: poll if SSE is silent
  if (Date.now() - lastMsg < 3000) return;
  fetch('/api/state').then(r => r.json()).then(s => { lastMsg = Date.now(); render(s); })
    .catch(() => { $('statusText').textContent = 'server not reachable'; });
}, 1000);
window.addEventListener('resize', () => state && render(state));
</script>
</body>
</html>
'''

# --------------------------------------------------------------------------- #
# 4b. Dashboard runtime + HTTP server
# --------------------------------------------------------------------------- #
HIST_LO, HIST_HI, HIST_BINS = 0.3, 0.9, 30
BUCKET_SEC = 0.5


class Runtime:
    """One running instance of: producer -> queue -> consumer (+ metrics)."""

    def __init__(self, X, y, init_size, contamination, retrain_every,
                 has_labels=True, rate=500.0, loop=False):
        self.has_labels = has_labels
        self.X, self.y, self.init_size = X, y, init_size
        self.det = StreamingDetector(X[:init_size], window=init_size,
                                       retrain_every=retrain_every,
                                       contamination=contamination)
        self.q = queue.Queue(maxsize=10000)
        self.stop = threading.Event()
        self.paused = threading.Event()
        self.rate = float(rate)
        self.loop = bool(loop)
        self.replays = 0
        self.lock = threading.Lock()
        self.producer_done = False
        self.finished = False
        self.rng = np.random.default_rng(0)

        self.t0 = time.time()
        self.n = self.tp = self.fp = self.fn = self.tn = 0
        self.alerts = deque(maxlen=40)
        self.log = deque(maxlen=12)
        self.lat = deque(maxlen=5000)
        self.hist = np.zeros((2, HIST_BINS), dtype=int)
        self.buckets = deque(maxlen=120)
        self._new_acc()
        self._log(f"Model warm-started on {init_size:,} rows "
                  f"(threshold {self.det.threshold:.3f})")

        self.error = None
        self.threads = [threading.Thread(target=self._guard(f), daemon=True)
                        for f in (self._producer, self._consumer, self._ticker)]
        for t in self.threads:
            t.start()

    # ---- helpers --------------------------------------------------------- #
    def _guard(self, fn):
        def run():
            try:
                fn()
            except Exception as e:          # surface instead of dying silently
                import traceback
                traceback.print_exc()
                self.error = f"{fn.__name__[1:]} thread crashed: {e!r}"
                with self.lock:
                    self._log("ERROR: " + self.error)
        return run

    def _new_acc(self):
        self.acc = dict(max=0.0, n=0, alerts=0, tp=0, fp=0, lat=[])

    def _log(self, msg):
        self.log.appendleft({"t": time.strftime("%H:%M:%S"), "msg": msg})

    def shutdown(self):
        self.stop.set()
        for t in self.threads:
            t.join(timeout=2)

    def inject(self):
        """Push a synthetic extreme event into the live stream."""
        d = self.X.shape[1]
        s = self.det.scaler
        vec = s.mean_ + s.scale_ * (self.rng.normal(0, 2, d)
                                    + self.rng.choice([-4, 4], d))
        try:
            self.q.put_nowait((-1, time.perf_counter(), vec, 1))
            self._log("Injected synthetic anomaly")
        except queue.Full:
            pass

    # ---- threads --------------------------------------------------------- #
    def _producer(self):
        i, carry, tick = self.init_size, 0.0, 0.02
        last, n = time.perf_counter(), len(self.X)
        while not self.stop.is_set():
            time.sleep(tick)
            now = time.perf_counter()
            dt, last = now - last, now
            if self.paused.is_set():
                continue
            carry += self.rate * dt
            k = int(carry)
            carry -= k
            for _ in range(k):
                if i >= n:
                    if not self.loop:
                        break
                    i = self.init_size             # replay from the start
                    self.replays += 1
                    with self.lock:
                        self._log(f"Replaying stream (loop #{self.replays})")
                if self.q.full():                  # back-pressure
                    carry = 0
                    break
                self.q.put((i, time.perf_counter(), self.X[i], self.y[i]))
                i += 1
            if i >= n and not self.loop:
                break
        self.producer_done = True

    def _consumer(self):
        while not self.stop.is_set():
            batch, deadline = [], time.perf_counter() + 0.05
            while len(batch) < 256:
                timeout = deadline - time.perf_counter()
                if timeout <= 0:
                    break
                try:
                    batch.append(self.q.get(timeout=timeout))
                except queue.Empty:
                    break
            if not batch:
                if self.producer_done and self.q.empty():
                    self.finished = True
                    self._log("Stream finished")
                    return
                continue

            ids = np.array([b[0] for b in batch])
            t_in = np.array([b[1] for b in batch])
            X = np.nan_to_num(np.vstack([b[2] for b in batch]).astype(float))
            y = np.array([b[3] for b in batch]).astype(int)

            scores = self.det.score(X)
            flags = scores > self.det.threshold
            lat_ms = (time.perf_counter() - t_in) * 1000

            with self.lock:
                self.n += len(batch)
                self.tp += int((flags & (y == 1)).sum())
                self.fp += int((flags & (y == 0)).sum())
                self.fn += int((~flags & (y == 1)).sum())
                self.tn += int((~flags & (y == 0)).sum())
                bins = np.clip(((scores - HIST_LO) / (HIST_HI - HIST_LO)
                                * HIST_BINS).astype(int), 0, HIST_BINS - 1)
                np.add.at(self.hist, (y, bins), 1)
                a = self.acc
                a["max"] = max(a["max"], float(scores.max()))
                a["n"] += len(batch)
                a["alerts"] += int(flags.sum())
                a["tp"] += int((flags & (y == 1)).sum())
                a["fp"] += int((flags & (y == 0)).sum())
                a["lat"].extend(lat_ms.tolist())
                self.lat.extend(lat_ms.tolist())
                for j in np.where(flags)[0]:
                    self.alerts.appendleft({
                        "id": int(ids[j]), "score": round(float(scores[j]), 4),
                        "fraud": bool(y[j]), "t": time.strftime("%H:%M:%S")})

            if self.det.update(X, flags):
                with self.lock:
                    self._log(f"Model retrained (#{self.det.n_retrains - 1}), "
                              f"new threshold {self.det.threshold:.3f}")

    def _ticker(self):
        while not self.stop.wait(BUCKET_SEC):
            if self.paused.is_set() or self.finished:
                continue                       # freeze charts instead of plotting zeros
            with self.lock:
                a = self.acc
                self._new_acc()
                lat = a["lat"]
                self.buckets.append({
                    "max": a["max"], "rate": a["n"] / BUCKET_SEC,
                    "alerts": a["alerts"], "tp": a["tp"], "fp": a["fp"],
                    "p95": float(np.percentile(lat, 95)) if lat else 0.0,
                    "thr": float(self.det.threshold)})

    # ---- snapshot for the UI -------------------------------------------- #
    def snapshot(self):
        with self.lock:
            tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            lat = np.array(self.lat) if self.lat else np.array([0.0])
            recent = list(self.buckets)[-6:]
            status = ("error" if self.error else "finished" if self.finished else
                      "paused" if self.paused.is_set() else "running")
            return {
                "status": status, "rate": self.rate, "loop": self.loop, "n": self.n,
                "throughput": float(np.mean([b["rate"] for b in recent])) if recent else 0,
                "alerts_total": tp + fp, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": prec, "recall": rec, "f1": f1,
                "p50": float(np.percentile(lat, 50)),
                "p95": float(np.percentile(lat, 95)),
                "p99": float(np.percentile(lat, 99)),
                "queue": self.q.qsize(), "threshold": float(self.det.threshold),
                "retrains": self.det.n_retrains - 1,
                "buckets": list(self.buckets),
                "hist": self.hist.tolist(),
                "hist_range": [HIST_LO, HIST_HI], "has_labels": self.has_labels,
                "alerts": list(self.alerts), "log": list(self.log),
                "uptime": time.time() - self.t0,
            }


# --------------------------------------------------------------------------- #
class App:
    runtime = None
    data = None
    cfg = None

    @classmethod
    def start(cls):
        if cls.runtime:
            cls.runtime.shutdown()
        X, y, init, cont, retrain, has_labels, rate, loop = cls.data
        cls.runtime = Runtime(X, y, init, cont, retrain, has_labels, rate, loop)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX_HTML, "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._send(200, json.dumps(App.runtime.snapshot()))
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    self.wfile.write(
                        f"data: {json.dumps(App.runtime.snapshot())}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.5)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self._send(404, "{}")

    def do_POST(self):
        if self.path != "/control":
            return self._send(404, "{}")
        length = int(self.headers.get("Content-Length", 0))
        msg = json.loads(self.rfile.read(length) or b"{}")
        rt, action = App.runtime, msg.get("action")
        if action == "pause":
            rt.paused.set()
        elif action == "resume":
            rt.paused.clear()
        elif action == "loop":
            rt.loop = bool(msg.get("value"))
        elif action == "rate":
            rt.rate = max(10.0, min(50000.0, float(msg["value"])))
        elif action == "inject":
            rt.inject()
        elif action == "restart":
            App.start()
        self._send(200, json.dumps({"ok": True}))


# --------------------------------------------------------------------------- #
# 5. CLI
# --------------------------------------------------------------------------- #
def run_headless(args):
    X, y, meta, init = setup_data(args)
    args.init_size = init

    det = StreamingDetector(X[:args.init_size], window=args.init_size,
                            retrain_every=args.retrain_every,
                            contamination=args.contamination)
    print(f"[model] warm-started on {args.init_size:,} rows, "
          f"threshold={det.threshold:.4f}")

    q = queue.Queue(maxsize=10000)
    alerts = []
    res = {k: [] for k in ("id", "score", "flag", "label", "latency_ms")}

    tp = threading.Thread(target=producer, daemon=True,
                          args=(q, X, y, args.init_size, args.rate, args.max_events))
    tc = threading.Thread(target=consumer,
                          args=(q, det, args.batch_size, args.max_wait,
                                20000, alerts, res))
    t0 = time.perf_counter()
    tp.start(); tc.start()
    tp.join(); tc.join()
    wall = time.perf_counter() - t0

    # ---- Report ---------------------------------------------------------- #
    label = np.array(res["label"]); flag = np.array(res["flag"])
    score = np.array(res["score"]); lat = np.array(res["latency_ms"])
    p, r, f, _ = precision_recall_fscore_support(label, flag, average="binary",
                                                 zero_division=0)
    print("\n================ RESULTS ================")
    print(f"events processed : {len(label):,} in {wall:.1f}s "
          f"({len(label)/wall:,.0f} ev/s)")
    if meta["has_labels"]:
        print(f"alerts raised    : {flag.sum():,}  (true anomalies in stream: {label.sum():,})")
        print(f"precision/recall : {p:.3f} / {r:.3f}   F1 = {f:.3f}")
    else:
        print(f"alerts raised    : {flag.sum():,}  (no labels - quality metrics n/a)")
    if meta["has_labels"] and label.sum() > 0:
        print(f"ROC-AUC          : {roc_auc_score(label, score):.4f}")
        print(f"PR-AUC (AP)      : {average_precision_score(label, score):.4f}")
    print(f"latency ms       : p50={np.percentile(lat,50):.1f}  "
          f"p95={np.percentile(lat,95):.1f}  p99={np.percentile(lat,99):.1f}")
    print(f"model retrains   : {det.n_retrains - 1}")

    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame(alerts, columns=["event_id", "anomaly_score", "true_label"]) \
        .to_csv(os.path.join(args.out, "alerts.csv"), index=False)
    pd.DataFrame(res).to_csv(os.path.join(args.out, "scored_stream.csv"), index=False)
    print(f"[out] wrote {args.out}/alerts.csv and scored_stream.csv")


def run_dashboard(args):
    X, y, meta, init = setup_data(args)
    remaining = len(X) - init
    rate = args.start_rate or max(20.0, min(500.0, remaining / 120))   # ~2 min of stream
    print(f"[ui] start rate {rate:,.0f} events/s "
          f"({remaining:,} events to stream, loop={'on' if args.loop else 'off'})")
    App.data = (X, y, init, args.contamination, args.retrain_every,
                meta["has_labels"], rate, args.loop)
    App.start()
    srv, port = None, args.port
    for port in range(args.port, args.port + 10):
        try:
            srv = ThreadingHTTPServer((args.host, port), Handler)
            break
        except OSError:
            print(f"[ui] port {port} is busy (an old run still open?) - trying {port + 1}")
    if srv is None:
        raise SystemExit("No free port found. Close old Python windows and retry.")
    srv.daemon_threads = True
    print(f"[ui] dashboard at http://localhost:{port}  (Ctrl+C to quit)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="path to any CSV file")
    ap.add_argument("--kaggle", metavar="OWNER/DATASET",
                    help="download a Kaggle dataset via kagglehub")
    ap.add_argument("--kaggle-file", help="CSV filename inside the Kaggle dataset")
    ap.add_argument("--label", help="ground-truth column (auto-detected if omitted)")
    ap.add_argument("--anomaly-value", help="label value that means anomaly")
    ap.add_argument("--time-col", help="column used to order the stream")
    ap.add_argument("--timeseries", action="store_true",
                    help="build rolling-window features (auto-on for a single signal)")
    ap.add_argument("--no-timeseries", action="store_true", help="disable the above")
    ap.add_argument("--ts-windows", default="12,48",
                    help="rolling window sizes in samples (default 12,48)")
    ap.add_argument("--labels-json", help="NAB-style labels file (combined_windows.json)")
    ap.add_argument("--download-labels", action="store_true",
                    help="fetch NAB combined_windows.json from GitHub")
    ap.add_argument("--nab-key", help="key in labels JSON, e.g. realKnownCause/ambient_temperature_system_failure.csv")
    ap.add_argument("--features", help="comma-separated feature columns (default: all numeric)")
    ap.add_argument("--log-cols", help="comma-separated columns to log1p ('' = none; default auto)")
    ap.add_argument("--synthetic", action="store_true", help="offline synthetic data")
    ap.add_argument("--headless", action="store_true",
                    help="run pipeline and print a report (no UI)")
    ap.add_argument("--init-size", type=int, default=20000)
    ap.add_argument("--contamination", type=float, default=0.002)
    ap.add_argument("--retrain-every", type=int, default=10000)
    # dashboard
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--loop", action="store_true",
                    help="replay the stream forever instead of stopping at the end")
    ap.add_argument("--start-rate", type=float, default=None,
                    help="initial events/sec (default: auto, ~2 min of stream)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="use 0.0.0.0 to expose on your network")
    # headless
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--rate", type=float, default=0, help="events/sec (0 = max)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-wait", type=float, default=0.05)
    ap.add_argument("--out", default="output")
    args = ap.parse_args()
    (run_headless if args.headless else run_dashboard)(args)


if __name__ == "__main__":
    main()
