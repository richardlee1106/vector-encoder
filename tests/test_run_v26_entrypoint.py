# -*- coding: utf-8 -*-

from pathlib import Path


def test_run_py_mentions_v26_command():
    text = Path("spatial_encoder/run.py").read_text(encoding="utf-8")

    assert 'elif cmd == "serve"' in text
    assert 'elif cmd == "train_v26"' in text
    assert 'elif cmd == "export_v26"' in text
    assert 'elif cmd == "validate_v26"' in text
    assert 'elif cmd == "quick_validate_v26"' in text
    assert 'elif cmd == "preprocess_v26"' in text
    assert "run_quick_validate_v26" in text
    assert "run_preprocess_v26" in text
