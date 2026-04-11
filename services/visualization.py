import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# visualization 계층은 dataframe을 plotly figure로 변환하는 역할만 담당한다.
# 데이터 가공과 계산 로직은 여기서 하지 않고, 시각화에 필요한 최소 join 정도만 수행한다.


def _safe_color_column(df: pd.DataFrame, preferred_column: str) -> str | None:
    # 사용자가 선택한 legend 컬럼이 없을 수도 있으므로
    # 대체 가능한 컬럼을 순서대로 찾아서 반환한다.
    candidate_columns = [preferred_column, "ppid", "lot_type", "item_id", "root_lot_id"]
    for col in candidate_columns:
        if col in df.columns:
            return col
    return None


def build_trend_scatter(df: pd.DataFrame, title: str, legend_column: str = "ppid") -> go.Figure:
    # incoming / target metro trend scatterplot을 생성한다.
    # x축은 tkout_time, y축은 fab_value, color는 사용자가 선택한 legend 기준이다.
    working_df = df.copy() if df is not None else pd.DataFrame()
    if working_df.empty:
        fig = go.Figure()
        fig.update_layout(title=title)
        fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)
        return fig

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
    # incoming / target raw dataframe을
    # root_lot_id + wafer_id 기준으로 inner join해서 비교용 데이터셋을 만든다.
    if incoming_df is None or target_df is None or incoming_df.empty or target_df.empty:
        return pd.DataFrame()

    incoming_cols = [
        "root_lot_id",
        "wafer_id",
        "fab_value",
        "ppid",
        "lot_type",
        "item_id",
    ]
    target_cols = [
        "root_lot_id",
        "wafer_id",
        "fab_value",
        "ppid",
        "lot_type",
        "item_id",
    ]
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
    # join된 incoming / target 데이터에서 비교 scatterplot을 만들고
    # 회귀선을 추가한 뒤 Pearson R^2 값을 계산한다.
    # 또한 회귀식과 R^2를 그래프 내부 annotation으로 표시한다.
    if joined_df is None or joined_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Incoming vs Target")
        fig.add_annotation(text="No joined data", x=0.5, y=0.5, showarrow=False)
        return fig, None

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

    annotation_text = "<b>Regression unavailable</b>"
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
    # 제어용 scatterplot은 outlier_status 기준으로 색상을 고정한다.
    # upper_out / lower_out은 red, normal은 lightgrey로 표시해 시인성을 높인다.
    working_df = df.copy() if df is not None else pd.DataFrame()
    if working_df.empty or x_col not in working_df.columns or y_col not in working_df.columns:
        fig = go.Figure()
        fig.update_layout(title=title)
        fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)
        return fig

    if "outlier_status" not in working_df.columns:
        working_df["outlier_status"] = "normal"

    status_order = ["normal", "lower_out", "upper_out"]
    working_df["outlier_status"] = pd.Categorical(
        working_df["outlier_status"],
        categories=status_order,
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
        category_orders={"outlier_status": status_order},
        title=title,
    )
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        legend_title="Outlier",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
