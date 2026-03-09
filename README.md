# SSE-Bench

## Super-Spreading Event Forecasting Benchmark

Predict whether an emerging social event will become a **Super-Spreading Event (SSE)** using only early observable signals.

SSE-Bench studies **phase transitions in social diffusion systems**: when normal information propagation turns into explosive large-scale cascades.

This benchmark focuses on **event-level forecasting**, rather than post-level popularity prediction.

---

# Motivation

A small fraction of social events trigger **disproportionately large diffusion cascades**, similar to **super-spreading events (SSE)** in epidemiology.

Examples include:

* viral misinformation outbreaks
* major political controversies
* celebrity scandals
* sudden brand crises
* unexpected meme explosions

Current benchmarks primarily focus on:

* post popularity prediction
* cascade size prediction
* virality classification

However, these benchmarks rarely address the core scientific question:

**Can we detect early signals that an event will become a super-spreading event?**

SSE-Bench introduces a unified framework to study this problem.

---

# Benchmark Tasks

SSE-Bench defines four core tasks.

## Task 1: SSE Detection

Predict whether an event will become a **Super-Spreading Event**.

Input:

Early observations within window (w)

Output:

```
P(event becomes SSE)
```

Evaluation:

* AUROC
* AUPRC
* F1

---

## Task 2: Time-to-SSE

Predict when the event will enter the super-spreading phase.

Output:

```
t_sse
```

Evaluation:

* MAE
* Concordance index

---

## Task 3: Final Cascade Size

Predict the final magnitude of the diffusion cascade.

Output:

```
log(final_cascade_size)
```

Evaluation:

* RMSE
* Spearman correlation

---

## Task 4: Historical Analogue Retrieval

Given an emerging event, retrieve the most similar historical events.

Output:

```
Top-k similar events
```

Evaluation:

* Recall@k
* NDCG

---

# Super-Spreading Event Definition

An event is classified as SSE if it satisfies diffusion thresholds.

Let:

```
S = final cascade size
G = early growth rate
```

An event is SSE if:

```
S > percentile_q
AND
G > μ + 3σ
```

Typical configuration:

```
q = 99%
```

Alternative definitions may include:

* cross-community cascade spread
* structural cascade depth
* media amplification

The benchmark allows configurable SSE definitions.

---

# Event Object Schema

Each event is represented as an **Event Object**.

```python
class Event:
    event_id: str
    platform: str
    start_time: datetime

    # early observation windows
    engagement_series: List[int]
    sentiment_series: List[float]

    # cascade structure
    cascade_graph: Optional[Graph]

    # labels
    is_sse: bool
    time_to_sse: Optional[int]
    final_cascade_size: int
```

---

# Benchmark Protocol

SSE-Bench uses **strict chronological evaluation**.

Rules:

1. Training events occur strictly before test events
2. Models may only use information available before prediction time
3. No future leakage allowed

Prediction windows:

```
10 minutes
1 hour
6 hours
24 hours
```

Models must predict outcomes beyond the observation window.

---

# Dataset Sources

SSE-Bench supports multiple datasets through adapters.

Examples:

* SMPD (Social Media Prediction Dataset)
* SMTPD (Temporal popularity dataset)
* Reddit-V virality dataset
* custom event datasets

Adapters convert raw datasets into the unified **Event Object** format.

---

# Baseline Models

SSE-Bench includes baseline methods.

## Baseline 1: Early Growth Heuristic

Simple heuristic using early engagement growth.

---

## Baseline 2: Gradient Boosting

Model:

```
XGBoost
```

Features:

* early engagement growth
* burstiness
* sentiment statistics
* cascade statistics

---

## Baseline 3: Temporal Models

Models:

* LSTM
* Temporal Convolution Network (TCN)

Input:

early engagement time series.

---

## Baseline 4: Trajectory Retrieval

Similarity search using early diffusion trajectories.

---

# Evaluation Metrics

## Classification

```
AUROC
AUPRC
F1
```

## Regression

```
RMSE
Spearman correlation
```

## Time-to-event

```
MAE
Concordance index
```

## Retrieval

```
Recall@k
NDCG
```

## Calibration

```
Brier score
Expected Calibration Error
```

---

# Repository Structure

```
sse-bench
│
├── README.md
├── benchmark_spec.md
│
├── datasets
│   └── adapters
│
├── schema
│   └── event.py
│
├── features
│
├── baselines
│
├── evaluation
│
├── retrieval
│
└── experiments
```

---

# Research Goals

SSE-Bench aims to study:

* early warning signals of diffusion explosions
* structural conditions of super-spreading
* historical pattern retrieval
* forecasting uncertainty

Long-term goals include:

* intervention evaluation
* social diffusion digital twins
* policy simulation

---

# Roadmap

Version 0.1

* benchmark specification
* event schema
* baseline models

Version 0.2

* intervention evaluation
* counterfactual analysis

Version 0.3

* agent-based simulation integration

---

# License

Apache-2.0

---

# Citation (draft)

```
@misc{ssebench2026,
  title={SSE-Bench: Super-Spreading Event Forecasting Benchmark},
  year={2026}
}
```

---

# 下一步最关键的两件事

你现在可以做：

**1️⃣ 创建 GitHub repo**

名字：

```
sse-bench
```

**2️⃣ 把 README 放进去**

然后我们下一步马上要做的是：

**Event Schema v1 代码 + Dataset Adapter**

这一步我可以直接给你完整 Python pipeline。
