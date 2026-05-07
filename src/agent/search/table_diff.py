from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


Row = Mapping[str, Any]


@dataclass(frozen=True)
class ChangedRow:
    key: str
    before: dict[str, Any]
    after: dict[str, Any]
    changed_cells: tuple[str, ...]


@dataclass(frozen=True)
class TableDiff:
    key_column: str
    added: tuple[dict[str, Any], ...]
    removed: tuple[dict[str, Any], ...]
    changed: tuple[ChangedRow, ...]
    unchanged: tuple[dict[str, Any], ...]


DEFAULT_KEY_COLUMNS = (
    "id",
    "row id",
    "document id",
    "doc id",
    "doc_id",
    "requirement id",
    "req id",
    "test id",
    "record id",
    "item",
    "name",
)

_MISSING = object()


def diff_table_rows(
    before_rows: Sequence[Row],
    after_rows: Sequence[Row],
    *,
    preferred_key_column: str | None = None,
) -> TableDiff:
    key_column = _resolve_key_column(before_rows, after_rows, preferred_key_column)
    before_by_key, before_order = _index_rows(before_rows, key_column)
    after_by_key, after_order = _index_rows(after_rows, key_column)

    added = tuple(after_by_key[key] for key in after_order if key not in before_by_key)
    removed = tuple(before_by_key[key] for key in before_order if key not in after_by_key)
    changed: list[ChangedRow] = []
    unchanged: list[dict[str, Any]] = []

    for key in after_order:
        if key not in before_by_key:
            continue
        before = before_by_key[key]
        after = after_by_key[key]
        changed_cells = _changed_cells(before, after, key_column)
        if changed_cells:
            changed.append(
                ChangedRow(
                    key=key,
                    before=before,
                    after=after,
                    changed_cells=changed_cells,
                )
            )
        else:
            unchanged.append(after)

    return TableDiff(
        key_column=key_column,
        added=added,
        removed=removed,
        changed=tuple(changed),
        unchanged=tuple(unchanged),
    )


def _resolve_key_column(
    before_rows: Sequence[Row],
    after_rows: Sequence[Row],
    preferred_key_column: str | None,
) -> str:
    rows = [*before_rows, *after_rows]
    if preferred_key_column:
        match = _find_column(rows, preferred_key_column)
        if match is None:
            raise ValueError(f"preferred key column {preferred_key_column!r} was not found")
        return match

    for candidate in DEFAULT_KEY_COLUMNS:
        match = _find_column(rows, candidate)
        if match and _has_unique_keys(before_rows, match) and _has_unique_keys(after_rows, match):
            return match
    for column in _ordered_columns(rows):
        if _has_unique_keys(before_rows, column) and _has_unique_keys(after_rows, column):
            return column
    raise ValueError("could not infer a unique key column")


def _index_rows(rows: Sequence[Row], key_column: str) -> tuple[dict[str, dict[str, Any]], list[str]]:
    indexed: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        key = _row_key(row, key_column)
        if key in indexed:
            raise ValueError(f"duplicate key {key!r} for column {key_column!r}")
        indexed[key] = dict(row)
        order.append(key)
    return indexed, order


def _row_key(row: Row, key_column: str) -> str:
    value = _cell(row, key_column)
    if value is _MISSING or value is None or str(value).strip() == "":
        raise ValueError(f"row is missing a non-empty key value for column {key_column!r}")
    return str(value).strip()


def _changed_cells(before: Row, after: Row, key_column: str) -> tuple[str, ...]:
    changed: list[str] = []
    for column in _ordered_columns([before, after]):
        if _same_column(column, key_column):
            continue
        if _cell(before, column) != _cell(after, column):
            changed.append(column)
    return tuple(changed)


def _cell(row: Row, column: str) -> Any:
    actual_column = _find_column([row], column)
    if actual_column is None:
        return _MISSING
    return row[actual_column]


def _find_column(rows: Sequence[Row], column: str) -> str | None:
    wanted = _normalize_column(column)
    for row in rows:
        for existing in row:
            if _normalize_column(existing) == wanted:
                return existing
    return None


def _ordered_columns(rows: Sequence[Row]) -> tuple[str, ...]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for column in row:
            key = _normalize_column(column)
            if key not in seen:
                seen.add(key)
                columns.append(column)
    return tuple(columns)


def _has_unique_keys(rows: Sequence[Row], key_column: str) -> bool:
    seen: set[str] = set()
    for row in rows:
        value = _cell(row, key_column)
        if value is _MISSING or value is None or str(value).strip() == "":
            return False
        key = str(value).strip()
        if key in seen:
            return False
        seen.add(key)
    return True


def _normalize_column(column: str) -> str:
    return " ".join(str(column).replace("_", " ").casefold().split())


def _same_column(left: str, right: str) -> bool:
    return _normalize_column(left) == _normalize_column(right)


__all__ = [
    "ChangedRow",
    "DEFAULT_KEY_COLUMNS",
    "TableDiff",
    "diff_table_rows",
]
