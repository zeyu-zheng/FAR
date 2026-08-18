"""Corpus access.

The corpus is the only place a paper's text lives: stage records carry a row
index instead of a copy, and every stage that needs the text comes back here.
Rows are numbered by position, so the same corpus always yields the same
indices -- reordering it between runs invalidates both `--resume` and every
record already written.

The corpus is one uncompressed Arrow IPC file with a `text` column. Arrow is
memory-mapped, so a lookup reads one row off disk rather than loading the file,
and the row offsets are part of the format rather than something this module
has to rebuild.
"""

import threading
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa

from src.utils import PAPER_FIELD

SUFFIX = ".arrow"
SCAN_CHUNK = 256


def open_table(path: str | Path) -> pa.Table:
    """Memory-map the corpus. Rows stay on disk until they are read."""
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix != SUFFIX:
        raise ValueError(f"unsupported corpus format: {resolved} (expected {SUFFIX})")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return pa.ipc.open_file(pa.memory_map(str(resolved), "r")).read_all()


def iter_rows(
    path: str | Path,
    stop_event: threading.Event | None = None,
    limit: int | None = None,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield (row_index, row) in corpus order. Indices start at 0."""
    table = open_table(path)
    row_index = 0
    for start in range(0, table.num_rows, SCAN_CHUNK):
        for row in table.slice(start, SCAN_CHUNK).to_pylist():
            if stop_event is not None and stop_event.is_set():
                return
            yield row_index, row
            row_index += 1
            if limit is not None and row_index >= limit:
                return


class Corpus:
    """The paper text, addressed by row index."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self._table: pa.Table | None = None
        self._lock = threading.Lock()

    def open(self) -> pa.Table:
        with self._lock:
            if self._table is None:
                self._table = open_table(self.path)
            return self._table

    def text(self, row_index: int) -> str:
        table = self.open()
        if not 0 <= row_index < table.num_rows:
            raise IndexError(
                f"row {row_index} is outside the corpus at {self.path} "
                f"({table.num_rows} rows); the record was written against a different corpus"
            )
        return str(table.column(PAPER_FIELD)[row_index].as_py() or "")
