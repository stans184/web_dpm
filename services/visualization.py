"""DPM 결과를 Plotly Figure로 변환하는 시각화 모듈."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _safe_color_column(df: pd.DataFrame, preferred_column: str) -> str | None:
    """요청한 범례 컬럼이 없을 때 사용할 대체 컬럼을 찾는다."""
    candidate_columns = [preferred_column, "ppid", "lot_type", "item_id", "root_lot_id"]
    for col in candidate_columns:
        if col in df.columns:
            return col
    return None


def _build_empty_figure(title: str, message: str = "데이터가 없습니다.") -> go.Figure:
    """데이터가 없을 때 보여줄 빈 Figure를 만든다."""
    fig = go.Figure()
    fig.update_layout(title=title)
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False)
    return fig


def build_trend_scatter(df: pd.DataFrame, title: str, legend_column: str = "ppid") -> go.Figure:
    """원본 incoming/target 추세 산점도를 만든다."""
    working_df = df.copy() if df is not None else pd.DataFrame()
    if working_df.empty:
        return _build_empty_figure(title)

    color_col = _safe_color_column(working_df, legend_column)
    fig = px.scatter(
        working_df.sort_values("tkout_time"),
        x="tkout_time",
        y="fab_value",
        color=color_col,
        title=title,
    )
    fig.update_layout(
        xaxis_title="tkout_time",
        yaxis_title="fab_value",
        legend_title=color_col or "legend",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def join_incoming_target_for_comparison(
    incoming_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """incoming과 target을 wafer 단위로 연결한다."""
    if incoming_df is None or target_df is None or incoming_df.empty or target_df.empty:
        return pd.DataFrame()

    incoming_cols = ["root_lot_id", "wafer_id", "fab_value", "ppid", "lot_type", "item_id"]
    target_cols = ["root_lot_id", "wafer_id", "fab_value", "ppid", "lot_type", "item_id"]

    incoming_base = incoming_df[[col for col in incoming_cols if col in incoming_df.columns]].copy()
    target_base = target_df[[col for col in target_cols if col in target_df.columns]].copy()

    joined_df = incoming_base.merge(
        target_base,
        on=["root_lot_id", "wafer_id"],
        how="inner",
        suffixes=("_incoming", "_target"),
    )
    return joined_df.reset_index(drop=True)


def build_comparison_scatter_with_regression(
    joined_df: pd.DataFrame,
    legend_column: str = "ppid_incoming",
) -> tuple[go.Figure, float | None]:
    """incoming/target 비교 산점도와 회귀선을 만든다."""
    if joined_df is None or joined_df.empty:
        return _build_empty_figure("Incoming vs Target", "조인된 데이터가 없습니다."), None

    color_col = _safe_color_column(joined_df, legend_column)
    fig = px.scatter(
        joined_df,
        x="fab_value_incoming",
        y="fab_value_target",
        color=color_col,
        title="Incoming vs Target",
    )

    x = joined_df["fab_value_incoming"].astype(float)
    y = joined_df["fab_value_target"].astype(float)
    annotation_text = "<b>회귀선을 계산할 수 없습니다.</b>"

    if len(joined_df) >= 2 and x.nunique() > 1:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="regression",
                line=dict(color="black", dash="dash"),
            )
        )
        r_value = x.corr(y)
        r_squared = None if pd.isna(r_value) else float(r_value ** 2)
        annotation_text = (
            f"<b>y = {slope:.4f}x + {intercept:.4f}</b><br>"
            f"<b>Pearson R^2 = {'-' if r_squared is None else f'{r_squared:.4f}'}</b>"
        )
    else:
        r_squared = None

    fig.add_annotation(
        x=0.98,
        y=0.98,
        xref="paper",
        yref="paper",
        xanchor="right",
        yanchor="top",
        align="left",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=18, color="black"),
        text=annotation_text,
    )
    fig.update_layout(
        xaxis_title="incoming fab_value",
        yaxis_title="target fab_value",
        legend_title=color_col or "legend",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig, r_squared


def build_outlier_scatter(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
) -> go.Figure:
    """outlier 상태를 색으로 구분한 추세 산점도를 만든다."""
    working_df = df.copy() if df is not None else pd.DataFrame()
    if working_df.empty or x_col not in working_df.columns or y_col not in working_df.columns:
        return _build_empty_figure(title)

    if "outlier_status" not in working_df.columns:
        working_df["outlier_status"] = "normal"

    working_df["outlier_status"] = pd.Categorical(
        working_df["outlier_status"],
        categories=["normal", "lower_out", "upper_out"],
        ordered=True,
    )

    fig = px.scatter(
        working_df.sort_values(x_col),
        x=x_col,
        y=y_col,
        color="outlier_status",
        color_discrete_map={
            "normal": "lightgrey",
            "lower_out": "red",
            "upper_out": "red",
        },
        category_orders={"outlier_status": ["normal", "lower_out", "upper_out"]},
        title=title,
    )
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        legend_title="Outlier",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def build_judged_trend_scatter(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    judge_col: str,
    title: str,
    color_map: dict[str, str],
    category_order: list[str],
) -> go.Figure:
    """판정 결과를 색으로 구분한 추세 산점도를 만든다."""
    working_df = df.copy() if df is not None else pd.DataFrame()
    if working_df.empty or x_col not in working_df.columns or y_col not in working_df.columns:
        return _build_empty_figure(title)

    if judge_col not in working_df.columns:
        working_df[judge_col] = category_order[0]

    working_df[judge_col] = pd.Categorical(
        working_df[judge_col],
        categories=category_order,
        ordered=True,
    )

    fig = px.scatter(
        working_df.sort_values(x_col),
        x=x_col,
        y=y_col,
        color=judge_col,
        color_discrete_map=color_map,
        category_orders={judge_col: category_order},
        title=title,
    )
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        legend_title=judge_col,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def build_target_control_boxplot(
    df: pd.DataFrame,
    *,
    title: str,
    target_col: str = "fab_value",
    controlled_col: str = "dpm_controlled_value",
) -> go.Figure:
    """target과 보정값 분포를 비교하는 box plot을 만든다."""
    working_df = df.copy() if df is not None else pd.DataFrame()
    if working_df.empty or target_col not in working_df.columns or controlled_col not in working_df.columns:
        return _build_empty_figure(title)

    box_df = pd.DataFrame(
        {
            "Target": pd.to_numeric(working_df[target_col], errors="coerce"),
            "DPM Controlled": pd.to_numeric(working_df[controlled_col], errors="coerce"),
        }
    ).melt(var_name="series", value_name="value").dropna(subset=["value"])

    if box_df.empty:
        return _build_empty_figure(title)

    fig = px.box(
        box_df,
        x="series",
        y="value",
        color="series",
        color_discrete_map={"Target": "green", "DPM Controlled": "red"},
        title=title,
        points=False,
    )
    fig.update_layout(
        xaxis_title="series",
        yaxis_title="value",
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
