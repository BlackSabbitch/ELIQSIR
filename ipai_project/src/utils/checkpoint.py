"""Checkpoint manager for the ELIQSIR ETL pipeline.

Provides :class:`CheckpointManager`, a lightweight helper that persists
pipeline DataFrames to disk after each stage and automatically reloads them
on the next run – skipping any work whose output already exists on disk.

Design goals
------------
- **Idempotent** – running the same pipeline twice never re-computes stages
  that already have a saved checkpoint.
- **Crash-safe** – each stage is saved as soon as it completes, so a crash
  mid-pipeline only loses the stage currently in progress.
- **Transparent** – a human-readable JSON manifest (``checkpoints.json``) is
  kept alongside the data files.  You can inspect, delete individual entries,
  or clear the whole manifest to force a full re-run.
- **Format-agnostic** – supports ``parquet``, ``pickle``, and ``csv``
  (controlled at construction time, defaults to ``parquet``).
- **Non-invasive** – the pipeline code interacts through exactly two methods:
  :meth:`save` and :meth:`load_or_run`.  Everything else is an opt-in helper.

Typical pipeline usage::

    from src.utils.checkpoint import CheckpointManager

    ckpt = CheckpointManager(directory="data/checkpoints")

    # Load from disk if the checkpoint exists, otherwise run the callable
    # and save the result automatically.
    raw_uniprot = ckpt.load_or_run("raw_uniprot", extractor.extract)

    # Later stages that depend on earlier ones:
    clean_uniprot = ckpt.load_or_run(
        "clean_uniprot",
        lambda: cleaner.clean_uniprot(raw_uniprot),
    )

Forcing a re-run of individual stages::

    ckpt.invalidate("raw_uniprot")           # delete one checkpoint
    ckpt.invalidate_stage("extraction")      # delete all extraction checkpoints
    ckpt.clear()                             # wipe everything

Inspecting the manifest::

    for entry in ckpt.list_checkpoints():
        print(entry["name"], entry["saved_at"], entry["rows"])
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.config import settings

logger = logging.getLogger(__name__)

# Shorthand for safe display of paths (project-relative, never absolute)
_dp = settings.display_path

# ---------------------------------------------------------------------------
# Supported serialisation formats
# ---------------------------------------------------------------------------

#: Map format name → (file extension, save callable, load callable)
_FORMAT_REGISTRY: dict[str, tuple[str, Callable, Callable]] = {
    "parquet": (
        ".parquet",
        lambda df, p: df.to_parquet(p, index=False),
        pd.read_parquet,
    ),
    "pickle": (
        ".pkl",
        lambda df, p: df.to_pickle(p),
        pd.read_pickle,
    ),
    "csv": (
        ".csv",
        lambda df, p: df.to_csv(p, index=False),
        pd.read_csv,
    ),
}

_MANIFEST_FILENAME = "checkpoints.json"


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Persist pipeline DataFrames to disk and reload them on the next run.

    Parameters
    ----------
    directory:
        Directory where checkpoint files and the manifest are stored.
        Created automatically if it does not exist.
    fmt:
        Serialisation format: ``"parquet"`` *(default)*, ``"pickle"``, or
        ``"csv"``.
    force_rerun:
        When ``True``, :meth:`load_or_run` **always** re-executes its
        callable and overwrites the existing checkpoint.  Useful for a
        one-off full refresh while keeping the manifest structure intact.
    """

    def __init__(
        self,
        directory: Path | str = "data/checkpoints",
        fmt: str = "parquet",
        force_rerun: bool = False,
    ) -> None:
        if fmt not in _FORMAT_REGISTRY:
            raise ValueError(
                f"fmt must be one of {list(_FORMAT_REGISTRY)}; got {fmt!r}"
            )

        self.directory = Path(directory)
        self.fmt = fmt
        self.force_rerun = force_rerun

        self._ext, self._saver, self._loader = _FORMAT_REGISTRY[fmt]
        self._manifest_path = self.directory / _MANIFEST_FILENAME

        self.directory.mkdir(parents=True, exist_ok=True)
        self._manifest: dict[str, dict[str, Any]] = self._load_manifest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def exists(self, name: str) -> bool:
        """Return ``True`` if a checkpoint named *name* is recorded in the
        manifest **and** its data file is present on disk."""
        if name not in self._manifest:
            return False
        data_path = self.directory / self._manifest[name]["filename"]
        return data_path.exists()

    def load_or_run(
        self,
        name: str,
        fn: Callable[[], pd.DataFrame],
        *,
        stage: str = "",
        description: str = "",
    ) -> pd.DataFrame:
        """Return a cached DataFrame or compute it (and cache the result).

        If a valid checkpoint named *name* already exists on disk **and**
        ``force_rerun`` is ``False``, the saved file is loaded and returned
        without calling *fn*.

        Otherwise *fn* is called, the result is saved, and the manifest is
        updated.

        Parameters
        ----------
        name:
            Unique identifier for this checkpoint (e.g. ``"raw_uniprot"``).
        fn:
            Zero-argument callable that produces the DataFrame.
        stage:
            Human-readable pipeline stage label (e.g. ``"extraction"``).
            Stored in the manifest for filtering/introspection.
        description:
            Optional free-text description stored in the manifest entry.

        Returns
        -------
        pd.DataFrame
        """
        if not self.force_rerun and self.exists(name):
            logger.info(
                "[checkpoint] ✓ '%s' – loading from disk (skip recompute).", name
            )
            return self.load(name)

        logger.info("[checkpoint] ↻ '%s' – running stage …", name)
        df = fn()
        self.save(name, df, stage=stage, description=description)
        return df

    def save(
        self,
        name: str,
        df: pd.DataFrame,
        *,
        stage: str = "",
        description: str = "",
    ) -> Path:
        """Persist *df* to disk and update the manifest.

        Parameters
        ----------
        name:
            Unique checkpoint name.
        df:
            DataFrame to persist.
        stage:
            Pipeline stage label (stored in manifest only).
        description:
            Free-text description (stored in manifest only).

        Returns
        -------
        pathlib.Path
            Absolute path of the written file.
        """
        filename = f"{name}{self._ext}"
        data_path = self.directory / filename

        self._saver(df, data_path)

        entry: dict[str, Any] = {
            "name": name,
            "filename": filename,
            "format": self.fmt,
            "stage": stage,
            "description": description,
            "rows": len(df),
            "columns": list(df.columns),
            "saved_at": datetime.now(tz=timezone.utc).isoformat(),
            "path": _dp(data_path),
        }
        self._manifest[name] = entry
        self._flush_manifest()

        logger.info(
            "[checkpoint] ✔ '%s' saved – %d rows → '%s'.", name, len(df), _dp(data_path)
        )
        return data_path

    def load(self, name: str) -> pd.DataFrame:
        """Load and return the DataFrame for checkpoint *name*.

        Raises
        ------
        KeyError
            If *name* is not in the manifest.
        FileNotFoundError
            If the data file recorded in the manifest no longer exists.
        """
        if name not in self._manifest:
            raise KeyError(
                f"No checkpoint named '{name}'. "
                f"Available: {list(self._manifest)}"
            )
        entry = self._manifest[name]
        data_path = self.directory / entry["filename"]

        if not data_path.exists():
            raise FileNotFoundError(
                f"Checkpoint '{name}' is in the manifest but its data file "
                f"'{_dp(data_path)}' is missing.  Run invalidate('{name}') and "
                "re-execute the pipeline."
            )

        # Use the format recorded in the manifest (not self.fmt) so that
        # a manager can load checkpoints saved by a previous run that used
        # a different format.
        _, _, loader = _FORMAT_REGISTRY[entry["format"]]
        df = loader(data_path)
        logger.info(
            "[checkpoint] ← '%s' loaded – %d rows from '%s'.",
            name, len(df), _dp(data_path),
        )
        return df

    def invalidate(self, name: str) -> bool:
        """Remove checkpoint *name* from the manifest and delete its file.

        Returns ``True`` if the checkpoint existed, ``False`` otherwise.
        Silently ignores a missing data file (the manifest entry is still
        removed).
        """
        if name not in self._manifest:
            logger.debug("[checkpoint] invalidate('%s') – not found, nothing to do.", name)
            return False

        entry = self._manifest.pop(name)
        data_path = self.directory / entry["filename"]
        if data_path.exists():
            data_path.unlink()
            logger.info("[checkpoint] ✗ '%s' invalidated (file deleted).", name)
        else:
            logger.info("[checkpoint] ✗ '%s' removed from manifest (file was missing).", name)

        self._flush_manifest()
        return True

    def invalidate_stage(self, stage: str) -> list[str]:
        """Invalidate all checkpoints belonging to *stage*.

        Returns the list of checkpoint names that were removed.
        """
        targets = [n for n, e in self._manifest.items() if e.get("stage") == stage]
        for name in targets:
            self.invalidate(name)
        if targets:
            logger.info(
                "[checkpoint] Invalidated %d checkpoint(s) for stage '%s': %s",
                len(targets), stage, targets,
            )
        return targets

    def clear(self) -> None:
        """Delete every checkpoint file and reset the manifest to empty."""
        for entry in list(self._manifest.values()):
            data_path = self.directory / entry["filename"]
            if data_path.exists():
                data_path.unlink()
        self._manifest.clear()
        self._flush_manifest()
        logger.info("[checkpoint] All checkpoints cleared.")

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """Return all manifest entries as a list, sorted by save time.

        Each entry is a dict with keys: ``name``, ``stage``, ``rows``,
        ``columns``, ``format``, ``saved_at``, ``path``, ``description``.
        The ``on_disk`` key is added to indicate whether the file exists.
        """
        entries = []
        for entry in self._manifest.values():
            data_path = self.directory / entry["filename"]
            enriched = {**entry, "on_disk": data_path.exists()}
            entries.append(enriched)
        return sorted(entries, key=lambda e: e.get("saved_at", ""))

    def summary(self) -> str:
        """Return a human-readable summary table of all checkpoints."""
        entries = self.list_checkpoints()
        if not entries:
            return "No checkpoints recorded."

        lines = [
            f"{'Name':<25} {'Stage':<15} {'Rows':>8}  {'Saved at (UTC)':<28}  {'On disk'}",
            "-" * 90,
        ]
        for e in entries:
            on_disk = "✓" if e["on_disk"] else "✗ MISSING"
            saved = e.get("saved_at", "")[:19].replace("T", " ")
            lines.append(
                f"{e['name']:<25} {e.get('stage',''):<15} {e['rows']:>8}  "
                f"{saved:<28}  {on_disk}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Manifest helpers (private)
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict[str, dict[str, Any]]:
        """Read the JSON manifest from disk, or return an empty dict."""
        if not self._manifest_path.exists():
            return {}
        try:
            with self._manifest_path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                logger.warning(
                    "[checkpoint] Manifest at '%s' has unexpected format – "
                    "starting fresh.", self._manifest_path
                )
                return {}
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "[checkpoint] Could not read manifest '%s': %s – starting fresh.",
                self._manifest_path, exc,
            )
            return {}

    def _flush_manifest(self) -> None:
        """Write the in-memory manifest dict to disk atomically."""
        tmp = self._manifest_path.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(self._manifest, fh, indent=2, ensure_ascii=False)
            tmp.replace(self._manifest_path)
        except OSError as exc:
            logger.error(
                "[checkpoint] Failed to write manifest '%s': %s",
                self._manifest_path, exc,
            )
            raise

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CheckpointManager(directory={str(self.directory)!r}, "
            f"fmt={self.fmt!r}, "
            f"checkpoints={list(self._manifest)!r})"
        )
