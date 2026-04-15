from __future__ import annotations

from pathlib import Path

import pandas as pd


STEP_MAPPING_FILES = {
    "memory": "mem_prc_with_met.csv",
    "foundry": "fdry_prc_with_met.csv",
}


def normalize_step_tokens(raw_value: str) -> list[str]:
    if not raw_value:
        return []
    normalized = str(raw_value).replace("\n", ",").replace(";", ",")
    return [token.strip().lower() for token in normalized.split(",") if token.strip()]


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized_columns = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in normalized_columns:
            return normalized_columns[candidate]
    return None


def load_step_mapping_df_for_fab(fab_type: str) -> pd.DataFrame:
    repository_dir = Path(__file__).resolve().parents[1] / "repositories"
    filename = STEP_MAPPING_FILES.get(fab_type)
    if not filename:
        return pd.DataFrame()

    csv_path = repository_dir / filename
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()


def get_prc_step_candidates(
    step_mapping_df: pd.DataFrame,
    metro_step_text: str,
    line_ids: list[str] | None = None,
    process_ids: list[str] | None = None,
) -> list[str]:
    if step_mapping_df is None or step_mapping_df.empty:
        return []

    prc_col = find_column(
        step_mapping_df,
        ["prc_step", "prc step", "prc_step_seq", "prc step seq", "step_seq", "step"],
    )
    metro_col = find_column(
        step_mapping_df,
        ["metro_step", "metro step", "metro_step_seq", "met_step_seq", "metro"],
    )
    line_col = find_column(step_mapping_df, ["line_id", "line id"])
    process_col = find_column(step_mapping_df, ["process_id", "process id"])
    if prc_col is None or metro_col is None:
        return []

    metro_tokens = normalize_step_tokens(metro_step_text)
    if not metro_tokens:
        return []

    selected_columns = [prc_col, metro_col]
    if line_col is not None:
        selected_columns.append(line_col)
    if process_col is not None:
        selected_columns.append(process_col)

    working_df = step_mapping_df[selected_columns].dropna(subset=[prc_col, metro_col]).copy()
    working_df[metro_col] = working_df[metro_col].astype(str).str.strip()
    working_df[prc_col] = working_df[prc_col].astype(str).str.strip()

    if line_col is not None:
        working_df[line_col] = working_df[line_col].astype(str).str.strip()
        normalized_line_ids = [line_id.strip().lower() for line_id in (line_ids or []) if line_id.strip()]
        if normalized_line_ids:
            working_df = working_df[working_df[line_col].str.lower().isin(normalized_line_ids)]

    if process_col is not None:
        working_df[process_col] = working_df[process_col].astype(str).str.strip()
        normalized_process_ids = [
            process_id.strip().lower() for process_id in (process_ids or []) if process_id.strip()
        ]
        if normalized_process_ids:
            working_df = working_df[working_df[process_col].str.lower().isin(normalized_process_ids)]

    matched_rows = working_df[working_df[metro_col].str.lower().isin(metro_tokens)]
    return sorted(matched_rows[prc_col].drop_duplicates().tolist())


def get_side_metro_candidates(
    step_mapping_df: pd.DataFrame,
    target_prc_step: str | None,
    line_ids: list[str] | None = None,
    process_ids: list[str] | None = None,
) -> list[str]:
    if step_mapping_df is None or step_mapping_df.empty or not target_prc_step:
        return []

    prc_col = find_column(
        step_mapping_df,
        ["prc_step", "prc step", "prc_step_seq", "prc step seq", "step_seq", "step"],
    )
    metro_col = find_column(
        step_mapping_df,
        ["metro_step", "metro step", "metro_step_seq", "met_step_seq", "metro"],
    )
    line_col = find_column(step_mapping_df, ["line_id", "line id"])
    process_col = find_column(step_mapping_df, ["process_id", "process id"])
    if prc_col is None or metro_col is None:
        return []

    selected_columns = [prc_col, metro_col]
    if line_col is not None:
        selected_columns.append(line_col)
    if process_col is not None:
        selected_columns.append(process_col)

    working_df = step_mapping_df[selected_columns].dropna(subset=[prc_col, metro_col]).copy()
    working_df[prc_col] = working_df[prc_col].astype(str).str.strip()
    working_df[metro_col] = working_df[metro_col].astype(str).str.strip()
    working_df = working_df[working_df[prc_col].str.lower() == str(target_prc_step).strip().lower()]

    if line_col is not None:
        working_df[line_col] = working_df[line_col].astype(str).str.strip()
        normalized_line_ids = [line_id.strip().lower() for line_id in (line_ids or []) if line_id.strip()]
        if normalized_line_ids:
            working_df = working_df[working_df[line_col].str.lower().isin(normalized_line_ids)]

    if process_col is not None:
        working_df[process_col] = working_df[process_col].astype(str).str.strip()
        normalized_process_ids = [
            process_id.strip().lower() for process_id in (process_ids or []) if process_id.strip()
        ]
        if normalized_process_ids:
            working_df = working_df[working_df[process_col].str.lower().isin(normalized_process_ids)]

    return sorted(working_df[metro_col].drop_duplicates().tolist())
