from __future__ import annotations

import math

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    ndcg_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    pred = (prob >= threshold).astype(int)

    metrics: dict[str, float] = {
        "auprc": float(average_precision_score(truth, prob)),
        "f1": float(f1_score(truth, pred, zero_division=0)),
        "precision": float(precision_score(truth, pred, zero_division=0)),
        "recall": float(recall_score(truth, pred, zero_division=0)),
    }
    metrics["auroc"] = float(roc_auc_score(truth, prob)) if len(np.unique(truth)) > 1 else math.nan
    return metrics


def regression_metrics(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((truth - pred) ** 2)))
    spearman = float(spearmanr(truth, pred).statistic)
    return {"rmse": rmse, "spearman": spearman}


def _concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    concordant = 0.0
    comparable = 0.0
    for left in range(len(y_true)):
        for right in range(left + 1, len(y_true)):
            if y_true[left] == y_true[right]:
                continue
            comparable += 1.0
            same_order = (y_true[left] < y_true[right] and y_pred[left] < y_pred[right]) or (
                y_true[left] > y_true[right] and y_pred[left] > y_pred[right]
            )
            if same_order:
                concordant += 1.0
            elif y_pred[left] == y_pred[right]:
                concordant += 0.5
    return concordant / comparable if comparable else math.nan


def time_to_event_metrics(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(truth - pred)))
    return {"mae": mae, "concordance_index": _concordance_index(truth, pred)}


def calibration_metrics(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    bins: int = 10,
) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (prob >= left) & (prob < right if right < 1.0 else prob <= right)
        if not mask.any():
            continue
        bin_conf = float(prob[mask].mean())
        bin_acc = float(truth[mask].mean())
        ece += mask.mean() * abs(bin_acc - bin_conf)
    return {"brier_score": float(brier_score_loss(truth, prob)), "ece": float(ece)}


def retrieval_metrics(
    y_true_relevance: np.ndarray,
    y_score: np.ndarray,
    k: int = 10,
) -> dict[str, float]:
    truth = np.asarray(y_true_relevance, dtype=float)
    score = np.asarray(y_score, dtype=float)
    topk = np.argsort(score, axis=1)[:, ::-1][:, :k]
    recalls = []
    for row_idx, indices in enumerate(topk):
        relevant = truth[row_idx] > 0
        denominator = relevant.sum()
        hit_count = relevant[indices].sum()
        recalls.append(float(hit_count / denominator) if denominator else 0.0)
    return {
        "recall_at_k": float(np.mean(recalls)),
        "ndcg_at_k": float(ndcg_score(truth, score, k=k)),
    }
