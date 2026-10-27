"""Regression tests for complete, non-stale visualization coverage."""

from pathlib import Path

import pandas as pd

from module4 import visualization


def test_generate_all_plots_records_every_evaluated_architecture(tmp_path, monkeypatch):
    monkeypatch.setattr(visualization, "PLOT_DIR", str(tmp_path))
    stale = tmp_path / "architecture_comparison.png"
    stale.write_bytes(b"old-demo-chart")
    results = pd.DataFrame(
        {
            "architecture": ["Naive RAG", "Rerank RAG", "Adaptive RAG", "RAG-Fusion"],
            "iou": [0.2, 0.5, 0.4, 0.3],
            "noise_ratio": [0.5, 0.2, 0.3, 0.4],
        }
    )

    paths = visualization.generate_all_plots(results)
    manifest = visualization.load_plot_manifest(str(tmp_path))

    assert manifest["architectures"] == ["Naive RAG", "Adaptive RAG", "RAG-Fusion"]
    assert "Rerank RAG" not in manifest["architectures"]
    assert manifest["plots"] == [Path(path).name for path in paths]
    assert "iou_comparison.png" in manifest["plots"]
    assert "noise_ratio.png" in manifest["plots"]
    assert stale.read_bytes() != b"old-demo-chart"
