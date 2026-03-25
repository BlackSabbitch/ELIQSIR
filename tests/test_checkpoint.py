"""Unit tests for :class:`~src.utils.checkpoint.CheckpointManager`.

Tests are fully self-contained: each fixture creates a fresh temporary
directory so tests never interfere with one another or with real pipeline
data.

Run with::

    pytest tests/test_checkpoint.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.utils.checkpoint import CheckpointManager, _MANIFEST_FILENAME


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_ckpt(tmp_path: Path) -> CheckpointManager:
    """A fresh CheckpointManager backed by a temp directory."""
    return CheckpointManager(directory=tmp_path, fmt="parquet")


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "id":   [1, 2, 3],
        "name": ["alpha", "beta", "gamma"],
        "value": [0.1, 0.2, 0.3],
    })


@pytest.fixture()
def sample_df2() -> pd.DataFrame:
    return pd.DataFrame({
        "id":   [10, 20],
        "name": ["delta", "epsilon"],
        "value": [1.0, 2.0],
    })


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestInit:
    def test_directory_is_created(self, tmp_path: Path) -> None:
        subdir = tmp_path / "nested" / "ckpts"
        assert not subdir.exists()
        CheckpointManager(directory=subdir)
        assert subdir.is_dir()

    def test_invalid_format_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="fmt must be one of"):
            CheckpointManager(directory=tmp_path, fmt="xlsx")

    def test_fresh_manager_has_empty_manifest(self, tmp_ckpt: CheckpointManager) -> None:
        assert tmp_ckpt.list_checkpoints() == []

    def test_repr_contains_directory(self, tmp_ckpt: CheckpointManager) -> None:
        assert "CheckpointManager" in repr(tmp_ckpt)


# ---------------------------------------------------------------------------
# save / exists / load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_creates_file(self, tmp_ckpt: CheckpointManager, sample_df: pd.DataFrame) -> None:
        path = tmp_ckpt.save("test_stage", sample_df)
        assert path.exists()

    def test_save_updates_manifest(self, tmp_ckpt: CheckpointManager, sample_df: pd.DataFrame) -> None:
        tmp_ckpt.save("my_stage", sample_df, stage="extraction")
        entries = tmp_ckpt.list_checkpoints()
        assert len(entries) == 1
        assert entries[0]["name"] == "my_stage"
        assert entries[0]["rows"] == len(sample_df)
        assert entries[0]["stage"] == "extraction"

    def test_save_records_columns(self, tmp_ckpt: CheckpointManager, sample_df: pd.DataFrame) -> None:
        tmp_ckpt.save("cols_test", sample_df)
        entry = tmp_ckpt.list_checkpoints()[0]
        assert entry["columns"] == list(sample_df.columns)

    def test_exists_true_after_save(self, tmp_ckpt: CheckpointManager, sample_df: pd.DataFrame) -> None:
        assert not tmp_ckpt.exists("raw")
        tmp_ckpt.save("raw", sample_df)
        assert tmp_ckpt.exists("raw")

    def test_exists_false_for_unknown(self, tmp_ckpt: CheckpointManager) -> None:
        assert not tmp_ckpt.exists("nonexistent")

    def test_load_returns_equal_dataframe(self, tmp_ckpt: CheckpointManager, sample_df: pd.DataFrame) -> None:
        tmp_ckpt.save("df_check", sample_df)
        loaded = tmp_ckpt.load("df_check")
        pd.testing.assert_frame_equal(loaded, sample_df)

    def test_load_unknown_raises_key_error(self, tmp_ckpt: CheckpointManager) -> None:
        with pytest.raises(KeyError, match="No checkpoint named"):
            tmp_ckpt.load("ghost")

    def test_load_missing_file_raises_file_not_found(
        self, tmp_ckpt: CheckpointManager, sample_df: pd.DataFrame
    ) -> None:
        path = tmp_ckpt.save("will_vanish", sample_df)
        path.unlink()   # delete the data file but keep the manifest entry
        with pytest.raises(FileNotFoundError, match="missing"):
            tmp_ckpt.load("will_vanish")

    def test_manifest_is_persisted_to_disk(
        self, tmp_ckpt: CheckpointManager, sample_df: pd.DataFrame
    ) -> None:
        tmp_ckpt.save("persisted", sample_df)
        manifest_path = tmp_ckpt.directory / _MANIFEST_FILENAME
        assert manifest_path.exists()
        with manifest_path.open() as fh:
            data = json.load(fh)
        assert "persisted" in data


# ---------------------------------------------------------------------------
# load_or_run
# ---------------------------------------------------------------------------


class TestLoadOrRun:
    def test_calls_fn_when_no_checkpoint(
        self, tmp_ckpt: CheckpointManager, sample_df: pd.DataFrame
    ) -> None:
        call_count = {"n": 0}

        def fn() -> pd.DataFrame:
            call_count["n"] += 1
            return sample_df

        result = tmp_ckpt.load_or_run("first_run", fn)
        assert call_count["n"] == 1
        pd.testing.assert_frame_equal(result, sample_df)

    def test_skips_fn_when_checkpoint_exists(
        self, tmp_ckpt: CheckpointManager, sample_df: pd.DataFrame
    ) -> None:
        tmp_ckpt.save("cached", sample_df)

        call_count = {"n": 0}

        def fn() -> pd.DataFrame:
            call_count["n"] += 1
            return pd.DataFrame()   # different data – should NOT be returned

        result = tmp_ckpt.load_or_run("cached", fn)
        assert call_count["n"] == 0                  # fn was NOT called
        pd.testing.assert_frame_equal(result, sample_df)  # cached data returned

    def test_force_rerun_ignores_checkpoint(
        self, tmp_path: Path, sample_df: pd.DataFrame, sample_df2: pd.DataFrame
    ) -> None:
        ckpt = CheckpointManager(directory=tmp_path, fmt="parquet", force_rerun=True)
        ckpt.save("stale", sample_df)                # simulate existing checkpoint

        result = ckpt.load_or_run("stale", lambda: sample_df2)
        pd.testing.assert_frame_equal(result, sample_df2)  # fresh data returned

    def test_saves_checkpoint_after_fn(
        self, tmp_ckpt: CheckpointManager, sample_df: pd.DataFrame
    ) -> None:
        assert not tmp_ckpt.exists("auto_saved")
        tmp_ckpt.load_or_run("auto_saved", lambda: sample_df)
        assert tmp_ckpt.exists("auto_saved")

    def test_stage_label_stored(
        self, tmp_ckpt: CheckpointManager, sample_df: pd.DataFrame
    ) -> None:
        tmp_ckpt.load_or_run("labelled", lambda: sample_df, stage="cleaning")
        entry = tmp_ckpt.list_checkpoints()[0]
        assert entry["stage"] == "cleaning"


# ---------------------------------------------------------------------------
# invalidate / invalidate_stage / clear
# ---------------------------------------------------------------------------


class TestInvalidation:
    def test_invalidate_removes_file_and_entry(
        self, tmp_ckpt: CheckpointManager, sample_df: pd.DataFrame
    ) -> None:
        path = tmp_ckpt.save("to_remove", sample_df)
        assert path.exists()
        removed = tmp_ckpt.invalidate("to_remove")
        assert removed is True
        assert not path.exists()
        assert not tmp_ckpt.exists("to_remove")

    def test_invalidate_nonexistent_returns_false(self, tmp_ckpt: CheckpointManager) -> None:
        assert tmp_ckpt.invalidate("ghost") is False

    def test_invalidate_stage_removes_all_in_stage(
        self, tmp_ckpt: CheckpointManager, sample_df: pd.DataFrame
    ) -> None:
        tmp_ckpt.save("ext_a", sample_df, stage="extraction")
        tmp_ckpt.save("ext_b", sample_df, stage="extraction")
        tmp_ckpt.save("clean_a", sample_df, stage="cleaning")

        removed = tmp_ckpt.invalidate_stage("extraction")
        assert set(removed) == {"ext_a", "ext_b"}
        assert not tmp_ckpt.exists("ext_a")
        assert not tmp_ckpt.exists("ext_b")
        assert tmp_ckpt.exists("clean_a")   # other stage untouched

    def test_clear_removes_everything(
        self, tmp_ckpt: CheckpointManager, sample_df: pd.DataFrame
    ) -> None:
        tmp_ckpt.save("a", sample_df)
        tmp_ckpt.save("b", sample_df)
        tmp_ckpt.clear()
        assert tmp_ckpt.list_checkpoints() == []

    def test_clear_deletes_data_files(
        self, tmp_ckpt: CheckpointManager, sample_df: pd.DataFrame
    ) -> None:
        path = tmp_ckpt.save("to_wipe", sample_df)
        tmp_ckpt.clear()
        assert not path.exists()


# ---------------------------------------------------------------------------
# Crash-resume simulation
# ---------------------------------------------------------------------------


class TestCrashResume:
    """Simulate a mid-pipeline crash: some checkpoints exist, some do not.

    On resume, only missing stages should be re-executed.
    """

    def test_partial_resume_only_reruns_missing_stages(
        self, tmp_path: Path, sample_df: pd.DataFrame, sample_df2: pd.DataFrame
    ) -> None:
        ckpt = CheckpointManager(directory=tmp_path, fmt="parquet")

        # Simulate: stage A completed before crash
        ckpt.save("stage_a", sample_df, stage="extraction")

        # Simulate: stage B was in progress when crash happened (no checkpoint)
        calls = {"stage_b": 0}

        def run_stage_b() -> pd.DataFrame:
            calls["stage_b"] += 1
            return sample_df2

        # Resume run
        result_a = ckpt.load_or_run("stage_a", lambda: pd.DataFrame())  # fn skipped
        result_b = ckpt.load_or_run("stage_b", run_stage_b)

        pd.testing.assert_frame_equal(result_a, sample_df)
        pd.testing.assert_frame_equal(result_b, sample_df2)
        assert calls["stage_b"] == 1         # only B was re-executed

    def test_manifest_survives_reload(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        """Checkpoints saved by one manager instance are visible to a new one."""
        ckpt1 = CheckpointManager(directory=tmp_path, fmt="parquet")
        ckpt1.save("persistent", sample_df, stage="modelling")

        # Simulate process restart – create a new manager pointing at the same dir
        ckpt2 = CheckpointManager(directory=tmp_path, fmt="parquet")
        assert ckpt2.exists("persistent")
        loaded = ckpt2.load("persistent")
        pd.testing.assert_frame_equal(loaded, sample_df)


# ---------------------------------------------------------------------------
# Manifest robustness
# ---------------------------------------------------------------------------


class TestManifestRobustness:
    def test_corrupted_manifest_starts_fresh(self, tmp_path: Path) -> None:
        """A corrupt manifest JSON does not crash the manager."""
        manifest = tmp_path / _MANIFEST_FILENAME
        manifest.write_text("NOT VALID JSON }{", encoding="utf-8")
        # Should not raise – falls back to empty manifest
        ckpt = CheckpointManager(directory=tmp_path)
        assert ckpt.list_checkpoints() == []

    def test_exists_returns_false_when_file_missing_despite_manifest(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        ckpt = CheckpointManager(directory=tmp_path, fmt="parquet")
        data_path = ckpt.save("vanished", sample_df)
        data_path.unlink()   # delete file but leave manifest entry
        assert not ckpt.exists("vanished")


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------


class TestSummary:
    def test_empty_summary_message(self, tmp_ckpt: CheckpointManager) -> None:
        assert "No checkpoints" in tmp_ckpt.summary()

    def test_summary_contains_checkpoint_names(
        self, tmp_ckpt: CheckpointManager, sample_df: pd.DataFrame
    ) -> None:
        tmp_ckpt.save("alpha", sample_df, stage="extraction")
        tmp_ckpt.save("beta",  sample_df, stage="cleaning")
        summary = tmp_ckpt.summary()
        assert "alpha" in summary
        assert "beta"  in summary
        assert "extraction" in summary
        assert "cleaning"   in summary
