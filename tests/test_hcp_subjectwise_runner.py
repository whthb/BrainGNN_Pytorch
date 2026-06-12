import argparse
import importlib.util
import json
from pathlib import Path


def load_runner():
    path = Path(__file__).parents[1] / "15-run_hcp_subjectwise_new_baseline.py"
    spec = importlib.util.spec_from_file_location("hcp_subjectwise_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_experiments_use_unique_canonical_configs():
    runner = load_runner()
    selected = runner.selected_configs("all")
    signatures = {
        (item["unit"], item["tpk"], item["glc"], item["conv"], item["fc_dim"])
        for item in selected
    }
    assert len(selected) == 15
    assert len(signatures) == len(selected)


def test_exported_split_manifest_has_no_subject_leakage(tmp_path):
    runner = load_runner()
    dataroot = tmp_path / "data"
    subject_root = dataroot / "subjects"
    subject_root.mkdir(parents=True)
    subjects = {f"{index:03d}" for index in range(20)}
    for subject in subjects:
        (subject_root / f"{subject}.h5").touch()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    runner.write_split_manifest(
        argparse.Namespace(dataroot=dataroot, output_dir=output_dir, seed=123)
    )
    manifest = json.loads((output_dir / "split_manifest.json").read_text())
    all_test_subjects = set()
    for fold in manifest["folds"]:
        train = set(fold["train_subjects"])
        val = set(fold["val_subjects"])
        test = set(fold["test_subjects"])
        assert not train & val
        assert not train & test
        assert not val & test
        assert train | val | test == subjects
        all_test_subjects.update(test)
    assert all_test_subjects == subjects
