from __future__ import annotations

from pathlib import Path


def test_discover_prepare_scripts_includes_expected_keys() -> None:
    from scripts.prepare_all_datasets import discover_prepare_scripts

    discovered = discover_prepare_scripts()
    # The repo can grow new adapters; we only assert the current minimum set.
    assert "uci_news" in discovered
    assert "infopath" in discovered


def test_discover_suite_scripts_includes_expected_keys() -> None:
    from scripts.run_all_benchmark_suites import discover_suite_scripts

    discovered = discover_suite_scripts()
    assert "uci" in discovered
    assert "infopath" in discovered


def test_detect_processed_inputs_can_find_candidates(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_all_benchmark_suites as mod

    processed_root = tmp_path / "processed"
    (processed_root / "uci_news_sse").mkdir(parents=True)
    index_path = processed_root / "uci_news_sse" / "event_index.parquet"
    index_path.write_bytes(b"parquet-placeholder")

    monkeypatch.setattr(mod, "PROCESSED_ROOT", processed_root)
    hits = mod.detect_processed_inputs("uci")
    assert index_path in hits

