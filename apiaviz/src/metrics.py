"""Readout helpers: clean summary tables and simple linear-probe accuracy.

Kept dependency-light (no prettytable): ``render_table`` draws a bordered ASCII table so the
``pixi run`` evaluation tasks print a clean, aligned summary the way the reference codebase does.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def render_table(headers: Sequence[str], rows: Sequence[Sequence], title: str | None = None) -> str:
    """A bordered, aligned ASCII table (prettytable-style, no dependency)."""
    str_rows = [[str(c) for c in r] for r in rows]
    headers = [str(h) for h in headers]
    widths = [max(len(headers[i]), *(len(r[i]) for r in str_rows)) if str_rows else len(headers[i])
              for i in range(len(headers))]

    def rule(fill: str = "-", junc: str = "+") -> str:
        return junc + junc.join(fill * (w + 2) for w in widths) + junc

    def row(cells: Sequence[str]) -> str:
        return "| " + " | ".join(f"{c:<{w}}" for c, w in zip(cells, widths)) + " |"

    lines = []
    if title:
        lines.append(title)
    lines += [rule(), row(headers), rule("="), *[row(r) for r in str_rows], rule()]
    return "\n".join(lines)


def linear_probe_accuracy(feats: np.ndarray, labels: np.ndarray,
                          train_idx: np.ndarray, test_idx: np.ndarray) -> float:
    """Standardised logistic-regression probe accuracy (a simple, fixed read of a frozen code)."""
    scaler = StandardScaler().fit(feats[train_idx])
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(scaler.transform(feats[train_idx]), labels[train_idx])
    return float(accuracy_score(labels[test_idx], clf.predict(scaler.transform(feats[test_idx]))))
