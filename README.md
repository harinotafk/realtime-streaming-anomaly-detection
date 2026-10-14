# Real-Time Streaming Anomaly Detection Pipeline

An end-to-end, unsupervised anomaly detection system that scores events **as they arrive**, adapts to changing data, and visualizes everything on a live web dashboard. It runs on any tabular CSV and ships with first-class support for two benchmark datasets: the **Kaggle Credit Card Fraud** dataset and the **Numenta Anomaly Benchmark (NAB)** time series.

The whole project is a single Python file with no web framework and no JavaScript dependencies.

---

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [How It Works](#how-it-works)
4. [Quick Start](#quick-start)
5. [Usage](#usage)
6. [Command-Line Reference](#command-line-reference)
7. [Dashboard Guide](#dashboard-guide)
8. [Evaluation and Metrics](#evaluation-and-metrics)
9. [Tuning Guide](#tuning-guide)
10. [Limitations](#limitations)
11. [Troubleshooting](#troubleshooting)
12. [Future Work](#future-work)
13. [References and Credits](#references-and-credits)

---

## Features

- **Streaming simulation** with a producer thread, a bounded queue (back-pressure), and a micro-batch consumer.
- **Unsupervised detection** using an Isolation Forest. Labels are never used for training, only for evaluation.
- **Concept-drift handling** through a sliding training window, an adaptive threshold, and periodic retraining.
- **Non-blocking retraining.** New models are built in a background thread and swapped in, so scoring latency stays flat.
- **Works with any CSV.** Feature columns, label column, and time column are auto-detected, and every choice can be overridden from the command line.
- **Time-series mode.** A single signal with a timestamp automatically gets causal rolling-window features, which makes it valid for real-time use.
- **Live dashboard** (Server-Sent Events) with score chart, score histogram, throughput, latency, confusion matrix, alert table, and system log.
- **Interactive controls:** pause, change event rate, loop the stream, inject a synthetic anomaly, restart.
- **Headless mode** that prints a full evaluation report and writes results to CSV.

---

## Architecture

```
 CSV / Kaggle dataset
        |
        v
 +--------------+     +-----------------+     +-----------------------+
 |  Data layer  | --> |    Producer     | --> |    Bounded queue      |
 | load/prepare |     | (paced events)  |     |   (back-pressure)     |
 +--------------+     +-----------------+     +-----------+-----------+
                                                          |
                                                          v
                          +-----------------------------------------------+
                          |  Consumer: micro-batch scoring (<=256 events  |
                          |  or 50 ms)                                    |
                          |   - Isolation Forest anomaly score            |
                          |   - flag if score > adaptive threshold        |
                          |   - update sliding windows, metrics, latency  |
                          +--------+---------------------------+----------+
                                   |                           |
                       normal events                    alerts + metrics
                                   v                           v
                     +---------------------------+   +---------------------+
                     | Background retraining     |   |  HTTP server + SSE  |
                     | (new model swapped in)    |   |  live web dashboard |
                     +---------------------------+   +---------------------+
```

**Threads:** producer, consumer, a metrics ticker (builds 0.5 s buckets), and one handler thread per HTTP request. Shared metrics are protected by a lock. Only the consumer thread reads or swaps the model.

---

## How It Works

### 1. Data preparation
- Rows are sorted by the time column to preserve stream order.
- Numeric, non-constant columns become features. Infinities are removed and missing values are filled with the column median.
- Non-negative, heavily skewed columns (absolute skew above 5) receive a `log1p` transform, which is why `Amount` is handled automatically.
- Labels come from a label column or from a NAB-style labels JSON (windows or point timestamps).

### 2. Time-series features
When a dataset has one numeric signal and a time column, features are built from **past data only**:

| Feature | Meaning |
|---|---|
| `value` | raw reading |
| `value_diff` | first difference |
| `value_dev{w}` | deviation from the rolling mean over `w` samples |
| `value_std{w}` | rolling standard deviation |
| `value_z{w}` | z-score, deviation divided by rolling standard deviation |

The default windows are 12 and 48 samples, giving 8 features. Using only past values keeps the features valid in a live stream.

### 3. The model
- **Algorithm:** Isolation Forest (150 trees, `random_state=42`) on standardized features. Anomalies are few and different, so they are isolated in fewer random splits. A shorter average path means a higher anomaly score.
- **Score:** the score returned by scikit-learn is negated so that **higher means more anomalous**. Typical values fall between 0.3 and 0.9.
- **Threshold:** the `(1 - contamination)` quantile of scores over all recent events.
- **Learning type:** unsupervised. There are no weights and no gradient descent; training means building random trees.

### 4. Drift handling
- **Training window:** a sliding window of recent events that were **not** flagged, so the model does not learn anomalies as normal.
- **Calibration window:** all recent events, flagged ones included. The threshold is computed from this window. Calibrating on the filtered training window would make the threshold drift downward at every retrain and collapse precision, so the two windows are kept separate.
- **Retraining:** triggered every `--retrain-every` events in a background thread. If a retrain fails, the current model is kept.

### 5. Streaming mechanics
- The producer emits events at a configurable rate (`rate x elapsed time` per 20 ms tick, carrying the fractional remainder).
- The queue holds up to 10,000 events. When it is full, the producer pauses, which is back-pressure.
- The consumer scores vectorized micro-batches. Latency is measured from the moment an event enters the queue until it is scored, so it includes queueing delay.

---

## Quick Start

**Requirements:** Python 3.9 or newer.

```bash
pip install -r requirements.txt
python stream_anomaly.py --synthetic
```

Open **http://localhost:8765** in your browser. `--synthetic` generates data with the same schema as the Kaggle credit card dataset, so you can try the system without downloading anything.

`requirements.txt`:

```
numpy
pandas
scikit-learn
kagglehub   # optional, only for --kaggle auto-download
```

---

## Usage

### Credit Card Fraud (Kaggle)

1. Download `creditcard.csv` from <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud> and place it in `data/`.
2. Run:

```bash
python stream_anomaly.py --csv data/creditcard.csv
```

Columns `Class` (label) and `Time` (ordering) are detected automatically.

### Numenta Anomaly Benchmark (NAB)

```bash
python stream_anomaly.py --csv "NAB/realKnownCause/realKnownCause/nyc_taxi.csv" --contamination 0.01 --loop
```

- A single `value` column plus a timestamp automatically enables time-series mode.
- The script searches near the CSV for `combined_windows.json` or `combined_labels.json`. If a labels file is found, precision, recall, and F1 are computed.
- To point at a labels file explicitly:

```bash
python stream_anomaly.py --csv path/to/nyc_taxi.csv --labels-json path/to/combined_windows.json --contamination 0.01
```

- To fetch the labels file from GitHub: add `--download-labels`.
- Without labels, the system still runs fully, but accuracy metrics show n/a.

### Any other CSV

```bash
python stream_anomaly.py --csv my_data.csv
python stream_anomaly.py --csv my.csv --label status --anomaly-value fault --time-col timestamp --features temp,vibration,pressure
```

If the label values are ambiguous (for example `benign` and `dos`), the script stops and asks for `--anomaly-value`.

### Headless report (no UI)

```bash
python stream_anomaly.py --headless --csv data/creditcard.csv
```

Prints precision, recall, F1, ROC-AUC, PR-AUC, latency percentiles, and throughput, and writes:

| File | Content |
|---|---|
| `output/alerts.csv` | every flagged event with its score and true label |
| `output/scored_stream.csv` | the score, flag, label, and latency of every event |

---

## Command-Line Reference

### Data

| Flag | Default | Description |
|---|---|---|
| `--csv PATH` | auto | Path to any CSV file |
| `--synthetic` | off | Use generated credit-card-style data |
| `--kaggle OWNER/DATASET` | none | Download a dataset via `kagglehub` (needs Kaggle credentials) |
| `--kaggle-file NAME` | none | CSV filename inside the Kaggle dataset |
| `--label COL` | auto | Ground-truth column |
| `--anomaly-value VAL` | auto | Label value that means anomaly |
| `--time-col COL` | auto | Column used to order the stream |
| `--features A,B,C` | all numeric | Feature columns to model |
| `--log-cols A,B` | auto | Columns to `log1p` (`""` for none) |

### Time series

| Flag | Default | Description |
|---|---|---|
| `--timeseries` / `--no-timeseries` | auto | Force or disable rolling features |
| `--ts-windows 12,48` | `12,48` | Rolling window sizes in samples |
| `--labels-json PATH` | auto-search | NAB-style labels file |
| `--nab-key KEY` | CSV name | Entry to use inside the labels JSON |
| `--download-labels` | off | Fetch NAB `combined_windows.json` from GitHub |

### Model

| Flag | Default | Description |
|---|---|---|
| `--init-size N` | 20000 | Rows used to warm-start the model (reduced automatically for small datasets) |
| `--contamination X` | 0.002 | Expected anomaly fraction, which sets the threshold quantile |
| `--retrain-every N` | 10000 | Events between retrains (reduced automatically for short streams) |

### Dashboard

| Flag | Default | Description |
|---|---|---|
| `--port N` | 8765 | HTTP port (the next free port is used if busy) |
| `--host ADDR` | `127.0.0.1` | Use `0.0.0.0` to expose on your network |
| `--loop` | off | Replay the stream instead of stopping at the end |
| `--start-rate N` | auto | Initial events per second (auto aims for roughly two minutes of stream) |

### Headless

| Flag | Default | Description |
|---|---|---|
| `--headless` | off | Print a report instead of starting the dashboard |
| `--rate N` | 0 | Events per second (0 means as fast as possible) |
| `--max-events N` | all | Stop after N events |
| `--batch-size N` | 256 | Maximum micro-batch size |
| `--max-wait S` | 0.05 | Maximum seconds to fill a micro-batch |
| `--out DIR` | `output` | Output directory |

---

## Dashboard Guide

**Header controls**

| Control | Action |
|---|---|
| Pause / Resume | Freeze or continue the stream (charts freeze while paused) |
| Rate slider | Change events per second, from about 10 to 50,000 |
| Loop | Replay the stream when it ends |
| Inject anomaly | Push an extreme synthetic event into the live stream |
| Restart | Replay from the beginning with a freshly trained model |

**Panels**

- **KPI cards:** events, throughput, alerts, precision, recall, F1, p95 latency, queue depth, current threshold, retrain count.
- **Anomaly score:** the maximum score in each 0.5 s window against the threshold. Red dots mark true anomalies caught, and amber dots mark false alarms.
- **Score distribution:** log-scale histogram of normal and anomalous events.
- **Throughput and latency:** per-bucket events per second and p95 latency.
- **Confusion matrix:** TP, FP, FN, TN (shown as n/a when no labels exist).
- **Live alerts:** the latest 40 alerts with score and ground truth.
- **System log:** warm-up, retrains, replays, injected events, and errors.

**HTTP endpoints:** `GET /` (UI), `GET /stream` (Server-Sent Events), `GET /api/state` (JSON snapshot), `POST /control` (actions: `pause`, `resume`, `rate`, `loop`, `inject`, `restart`).

---

## Evaluation and Metrics

Accuracy is meaningless on rare-event data: a model that never flags anything scores above 99.8% on credit card fraud. This project reports:

| Metric | Meaning |
|---|---|
| Precision | Of the alerts raised, the share that were real anomalies |
| Recall | Of the real anomalies, the share that were caught |
| F1 | Harmonic mean of precision and recall |
| ROC-AUC / PR-AUC | Threshold-independent ranking quality (PR-AUC is the more informative one under heavy imbalance) |
| Latency p50 / p95 / p99 | End-to-end time from enqueue to scored |
| Throughput | Events scored per second |

ROC-AUC and PR-AUC are printed in `--headless` mode.

### Your results

Run the project on your data and record the numbers here.

| Dataset | Contamination | Precision | Recall | F1 | ROC-AUC | PR-AUC | p95 latency |
|---|---|---|---|---|---|---|---|
| Credit card fraud | | | | | | | |
| NAB nyc_taxi | | | | | | | |

### Reference point from the literature

Published runs of plain scikit-learn Isolation Forest on the Kaggle credit card dataset report a ROC-AUC of about 0.95 but a PR-AUC of only about 0.14, with precision around 0.26 when the number of flagged events matches the number of frauds. Precision of 25 to 35% is therefore expected for this algorithm. Given a base rate of roughly 0.17%, that is still well over a hundred times better than random flagging.

### Notes on evaluation

- NAB labels entire time windows as anomalous, so point-by-point precision and recall look worse than NAB's own window-aware scoring. These numbers are not comparable to NAB leaderboard scores.
- In loop mode the stream replays data the model has already seen, so use loop for demonstrations, not for judging accuracy.
- Precision and recall are cumulative since the start of the run.

---

## Tuning Guide

**`--contamination`** is the most important setting. It sets the share of events flagged as anomalous by design.

| Goal | Action |
|---|---|
| Fewer false alarms, higher precision | Lower it (for example 0.002, then 0.001, then 0.0005) |
| Catch more anomalies, higher recall | Raise it |
| Credit card fraud | Start at 0.001 to 0.002 |
| NAB time series | Start at 0.005 to 0.01 |

Going extremely low (below roughly 0.0002) makes the threshold depend on only a handful of events and makes it noisy. Halve the value step by step and check precision and recall after a few thousand events.

**Other levers:** `--retrain-every` (how quickly the model adapts), `--init-size` (warm-up size), `--ts-windows` (rolling feature horizons for time series), and `--features` (restrict the model to informative columns).

---

## Limitations

- **Simulated stream.** Everything runs in one process. A production system would use Kafka or Kinesis with multiple consumers.
- **No memory in the model.** Isolation Forest scores each row from its own features. Rolling features add temporal context, but slow drifts and seasonality shifts can still be missed.
- **Modest ceiling on fraud data.** Unsupervised Isolation Forest is a baseline. Supervised models trained on labels typically outperform it on credit card fraud, at the cost of needing labels.
- **Numeric features only.** Categorical columns are dropped and must be encoded manually if needed.
- **Contamination must be tuned** per dataset.
- **Single-user dashboard.** It has no authentication and binds to `127.0.0.1` by default.
- **Validation scope.** The system was developed and tested on synthetic data matching each dataset's schema. Quote only numbers produced by your own runs on the real files.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Port already in use | The script tries the next 9 ports; read the printed `dashboard at http://localhost:XXXX` line |
| Page shows old data or looks stuck | Close old terminals and browser tabs, then press Ctrl+F5 |
| Chart goes flat after a while | The stream finished. Press Restart, tick Loop, or start with `--loop` |
| Precision, recall, and the confusion matrix show n/a | No labels were found. Pass `--label` or `--labels-json`, or use `--download-labels` for NAB |
| "Couldn't tell which value means anomaly" | Add `--anomaly-value <value>` |
| `Key ... not in labels file` | Pass `--nab-key realKnownCause/<file>.csv` |
| Status pill shows `error` | See the traceback in the terminal and the system log panel |
| Kaggle download fails | Download the CSV manually and use `--csv` |
| Latency is high when unthrottled | The queue is backed up because the producer outruns the scorer; lower the rate |

---

## Future Work

- Mahalanobis-distance or robust-covariance detector as an alternative unsupervised model.
- Supervised or semi-supervised mode that retrains on delayed labels (for example chargebacks).
- Sequence-aware models for time series (autoencoder, LSTM, or forecasting-residual detection).
- Real message broker (Kafka or Kinesis) with parallel consumers.
- Threshold smoothing across retrains and per-series thresholds.
- NAB's own window-aware scoring.
- Automated tests and a Docker image.

---

## References and Credits

- **Isolation Forest:** F. T. Liu, K. M. Ting, Z.-H. Zhou, "Isolation Forest," IEEE ICDM, 2008.
- **Credit Card Fraud dataset:** ULB Machine Learning Group, <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>. Cite as A. Dal Pozzolo et al., "Calibrating Probability with Undersampling for Unbalanced Classification," IEEE CIDM, 2015.
- **Numenta Anomaly Benchmark:** A. Lavin, S. Ahmad, "Evaluating Real-time Anomaly Detection Algorithms - the Numenta Anomaly Benchmark," IEEE ICMLA, 2015 (<https://arxiv.org/abs/1510.03336>). Repository: <https://github.com/numenta/NAB>. Kaggle mirror: <https://www.kaggle.com/datasets/boltzmannbrain/nab>.
- **Libraries:** NumPy, pandas, scikit-learn.

Each dataset is governed by its own license; review the terms on the source page before redistributing it.

---

**Author:** _your name_  
**License:** _add a license of your choice_
