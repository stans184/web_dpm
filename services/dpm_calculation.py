"""DPM 제어 계산 로직을 담당하는 서비스 모듈."""

from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd

from models.dpm_models import ControlSectionResult, DpmCalculationResult, DpmInputModel


LOT_SUBITEM_TO_COLUMN = {
    "avg": "lot_avg_fab_value",
    "std": "lot_std_fab_value",
    "min": "lot_min_fab_value",
    "max": "lot_max_fab_value",
    "range": "lot_range_fab_value",
}

JUDGE_PRIORITY = {"normal": 0, "BOB": 1, "WOW": 2}
CONTROLLED_JUDGE_PRIORITY = {"normal": 0, "BOB": 1, "DPM_Controlled": 2}


def _safe_float(value) -> float | None:
    """문자열 기반 입력값을 안전하게 실수로 변환한다."""
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value) -> int | None:
    """문자열 기반 입력값을 안전하게 정수로 변환한다."""
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


class DpmCalculator:
    """전처리된 incoming/target 데이터를 바탕으로 DPM 결과를 계산한다."""

    def __init__(
        self,
        incoming_df: pd.DataFrame,
        target_df: pd.DataFrame,
        inputs: DpmInputModel,
        *,
        df_side: pd.DataFrame | None = None,
        df_l2: pd.DataFrame | None = None,
        df_l3: pd.DataFrame | None = None,
    ):
        self.incoming_df = incoming_df if incoming_df is not None else pd.DataFrame()
        self.target_df = target_df if target_df is not None else pd.DataFrame()
        self.inputs = inputs
        self.df_side = df_side if df_side is not None else pd.DataFrame()
        self.df_l2 = df_l2 if df_l2 is not None else pd.DataFrame()
        self.df_l3 = df_l3 if df_l3 is not None else pd.DataFrame()

    @property
    def lot_value_col(self) -> str:
        """incoming subitem에 맞는 lot 집계 컬럼명을 반환한다."""
        return LOT_SUBITEM_TO_COLUMN.get(self.inputs.incoming_subitem, LOT_SUBITEM_TO_COLUMN["avg"])

    @property
    def use_absolute_formula(self) -> bool:
        """망목제어일 때만 절대값 공식을 사용한다."""
        return self.inputs.control_scope_mode == "mesh"

    @property
    def use_ascending_rank(self) -> bool:
        """망대제어만 내림차순 정렬을 사용한다."""
        return self.inputs.control_scope_mode != "large"

    @property
    def window_days(self) -> int:
        """target 이력 조회 기간을 일 단위로 반환한다."""
        return max(0, _safe_int(self.inputs.window_days) or 0)

    @property
    def min_wafer_qty(self) -> int:
        """설비 그룹별 최소 wafer 수량을 반환한다."""
        return max(1, _safe_int(self.inputs.min_wafer_qty) or 1)

    def run(self) -> DpmCalculationResult:
        """세 가지 제어 결과를 계산해 최종 결과 객체를 만든다."""
        lot_eqp_result = self._build_lot_control_result(
            equipment_col="eqp_id",
            result_key="lot_eqp",
            title="Lot - EQP 제어",
        )
        lot_chamber_result = self._build_lot_control_result(
            equipment_col="eqp_chamber",
            result_key="lot_chamber",
            title="Lot - Chamber 제어",
        )
        wafer_chamber_result = self._build_wafer_chamber_result()

        return DpmCalculationResult(
            status="implemented_for_dpm_control",
            message="DPM 제어 계산 결과가 준비되었습니다.",
            incoming_rows=len(self.incoming_df),
            target_rows=len(self.target_df),
            side_rows=len(self.df_side),
            l2_rows=len(self.df_l2),
            l3_rows=len(self.df_l3),
            fab_type=self.inputs.fab_type,
            control_scope=self.inputs.control_scope,
            control_scope_mode=self.inputs.control_scope_mode,
            control_type=self.inputs.control_type,
            incoming_subitem=self.inputs.incoming_subitem,
            lot_value_col=self.lot_value_col,
            lot_eqp_result=lot_eqp_result,
            lot_chamber_result=lot_chamber_result,
            wafer_chamber_result=wafer_chamber_result,
        )

    def _resolve_wow_count(self, total_count: int) -> int:
        """WOW로 판정할 설비 수량을 계산한다."""
        if total_count <= 0:
            return 0

        wow_portion = _safe_float(self.inputs.wow_portion)
        if wow_portion is None or wow_portion <= 0:
            return 0

        if wow_portion <= 1:
            wow_count = math.ceil(total_count * wow_portion)
        elif wow_portion <= 100:
            wow_count = math.ceil(total_count * (wow_portion / 100.0))
        else:
            wow_count = math.ceil(wow_portion)

        return min(total_count, max(0, wow_count))

    def _build_lot_representative_df(self) -> pd.DataFrame:
        """wafer 단위 incoming 데이터를 lot 대표 행으로 축약한다."""
        if self.incoming_df.empty:
            return pd.DataFrame()
        if any(col not in self.incoming_df.columns for col in ["lot_id", "tkout_time"]):
            return pd.DataFrame()

        working_df = self.incoming_df.sort_values("tkout_time").copy()
        return working_df.groupby("lot_id", as_index=False).first().reset_index(drop=True)

    def _build_joined_wafer_df(self) -> pd.DataFrame:
        """incoming과 target을 wafer 단위로 연결한다."""
        if self.incoming_df.empty or self.target_df.empty:
            return pd.DataFrame()

        incoming_cols = [col for col in self.incoming_df.columns if col not in {"eqp_id", "eqp_chamber"}]
        return self.incoming_df[incoming_cols].copy().merge(
            self.target_df.copy(),
            on=["root_lot_id", "wafer_id"],
            how="inner",
            suffixes=("_incoming", "_target"),
        ).reset_index(drop=True)

    def _is_positive_correlation(self, joined_df: pd.DataFrame) -> bool:
        """incoming/target 상관관계가 양인지 판단한다."""
        if joined_df.empty:
            return True
        if "fab_value_incoming" not in joined_df.columns or "fab_value_target" not in joined_df.columns:
            return True

        corr = pd.to_numeric(joined_df["fab_value_incoming"], errors="coerce").corr(
            pd.to_numeric(joined_df["fab_value_target"], errors="coerce")
        )
        return False if pd.notna(corr) and corr < 0 else True

    def _resolve_thresholds(self, df: pd.DataFrame, value_col: str) -> tuple[float | None, float | None]:
        """현재 제어 방식에 맞는 하한/상한 임계값을 계산한다."""
        if df.empty or value_col not in df.columns:
            return None, None

        values = pd.to_numeric(df[value_col], errors="coerce").dropna()
        if values.empty:
            return None, None

        if self.inputs.control_type == "sigma":
            sigma = _safe_float(self.inputs.sigma)
            if sigma is None:
                return None, None

            avg = values.mean()
            std = values.std()
            if pd.isna(std):
                return None, None
            return float(avg - sigma * std), float(avg + sigma * std)

        if self.inputs.control_type == "spec":
            return _safe_float(self.inputs.lsl), _safe_float(self.inputs.usl)

        percentile = _safe_float(self.inputs.percentile)
        if percentile is None:
            return None, None
        percentile = max(0.0, min(100.0, percentile))
        return float(values.quantile(percentile / 100.0)), None

    def _classify_outliers(self, df: pd.DataFrame, value_col: str) -> tuple[pd.DataFrame, dict]:
        """값 컬럼 기준으로 normal/lower_out/upper_out 판정을 수행한다."""
        if df.empty or value_col not in df.columns:
            return pd.DataFrame(columns=df.columns if df is not None else None), {
                "value_col": value_col,
                "lower_spec": None,
                "upper_spec": None,
                "control_type": self.inputs.control_type,
            }

        working_df = df.copy()
        numeric_values = pd.to_numeric(working_df[value_col], errors="coerce")
        lower_spec, upper_spec = self._resolve_thresholds(working_df, value_col)

        status = pd.Series("normal", index=working_df.index)
        if lower_spec is not None:
            status = status.mask(numeric_values < lower_spec, "lower_out")
        if upper_spec is not None:
            status = status.mask(numeric_values > upper_spec, "upper_out")

        working_df["outlier_status"] = status
        return working_df.reset_index(drop=True), {
            "value_col": value_col,
            "lower_spec": lower_spec,
            "upper_spec": upper_spec,
            "control_type": self.inputs.control_type,
        }

    def _build_window_summary(self, event_time, equipment_col: str) -> pd.DataFrame:
        """특정 시점 기준으로 target 이력 window를 설비 단위로 요약한다."""
        if self.target_df.empty or equipment_col not in self.target_df.columns:
            return pd.DataFrame()
        if pd.isna(event_time):
            return pd.DataFrame()

        end_time = pd.Timestamp(event_time)
        start_time = end_time - pd.Timedelta(days=self.window_days)
        window_df = self.target_df[
            (self.target_df["tkout_time"] >= start_time) & (self.target_df["tkout_time"] <= end_time)
        ].copy()
        if window_df.empty:
            return pd.DataFrame()

        summary_df = (
            window_df.groupby(equipment_col, dropna=False)
            .agg(
                fab_value_count=("fab_value", "count"),
                fab_value_med=("fab_value", "median"),
            )
            .reset_index()
        )
        summary_df = summary_df[summary_df["fab_value_count"] >= self.min_wafer_qty].reset_index(drop=True)
        summary_df["window_start_time"] = start_time
        summary_df["window_end_time"] = end_time
        return summary_df

    def _score_window_summary(
        self,
        summary_df: pd.DataFrame,
        *,
        incoming_value: float,
        incoming_avg: float,
        target_avg: float,
        positive_correlation: bool,
    ) -> pd.DataFrame:
        """window 설비들을 점수화하고 BOB/WOW를 판정한다."""
        if summary_df.empty:
            return pd.DataFrame(
                columns=[
                    "fab_value_count",
                    "fab_value_med",
                    "window_start_time",
                    "window_end_time",
                    "dpm_score",
                    "dpm_rank",
                    "dpm_judge",
                ]
            )

        working_df = summary_df.copy()
        b_value = 1.0 if target_avg == 0 else incoming_avg / target_avg
        c_value = -2 * incoming_avg if positive_correlation else 0.0
        score = (1.0 * incoming_value) + (b_value * working_df["fab_value_med"].astype(float)) + c_value
        if self.use_absolute_formula:
            score = score.abs()

        working_df["dpm_score"] = score
        working_df["dpm_rank"] = working_df["dpm_score"].rank(
            method="dense",
            ascending=self.use_ascending_rank,
        ).astype(int)
        working_df["dpm_judge"] = "BOB"

        if len(working_df) < 3:
            return working_df.reset_index(drop=True)

        wow_count = self._resolve_wow_count(len(working_df))
        if wow_count <= 0:
            return working_df.reset_index(drop=True)

        worst_rows = working_df.sort_values(by=["dpm_rank", "dpm_score"], ascending=[False, False]).head(wow_count)
        working_df.loc[worst_rows.index, "dpm_judge"] = "WOW"
        return working_df.reset_index(drop=True)

    def _initialize_target_result_df(self) -> pd.DataFrame:
        """전체 target 데이터를 normal 상태 기본값으로 초기화한다."""
        if self.target_df.empty:
            return pd.DataFrame()

        working_df = self.target_df.copy().reset_index(drop=False).rename(columns={"index": "_target_row_id"})
        working_df["dpm_judge"] = "normal"
        working_df["dpm_rank"] = pd.NA
        working_df["dpm_score"] = pd.NA
        working_df["dpm_controlled_value"] = working_df["fab_value"]
        working_df["dpm_controlled_judge"] = "normal"
        return working_df

    def _apply_equipment_judge_to_target_rows(
        self,
        target_rows: pd.DataFrame,
        equipment_summary_df: pd.DataFrame,
        *,
        equipment_col: str,
        correction_value: float,
    ) -> pd.DataFrame:
        """설비 단위 판정 결과를 target 행에 반영한다."""
        working_df = target_rows.copy()
        if working_df.empty:
            return working_df

        if equipment_summary_df.empty or equipment_col not in working_df.columns:
            return working_df

        judge_map = equipment_summary_df.set_index(equipment_col)["dpm_judge"].to_dict()
        rank_map = equipment_summary_df.set_index(equipment_col)["dpm_rank"].to_dict()
        score_map = equipment_summary_df.set_index(equipment_col)["dpm_score"].to_dict()

        working_df["dpm_judge"] = working_df[equipment_col].map(judge_map).fillna("normal")
        working_df["dpm_rank"] = working_df[equipment_col].map(rank_map)
        working_df["dpm_score"] = working_df[equipment_col].map(score_map)
        working_df["dpm_controlled_judge"] = working_df["dpm_judge"].replace({"WOW": "DPM_Controlled"})

        wow_mask = working_df["dpm_judge"] == "WOW"
        if correction_value and wow_mask.any():
            working_df.loc[wow_mask, "dpm_controlled_value"] = working_df.loc[wow_mask, "fab_value"] - correction_value

        return working_df

    def _select_highest_priority(self, series: pd.Series, priority_map: dict[str, int]) -> str:
        """같은 행에 여러 판정이 겹칠 때 우선순위가 가장 높은 값을 고른다."""
        labels = [label for label in series.dropna().astype(str).tolist() if label in priority_map]
        if not labels:
            return "normal"
        return max(labels, key=lambda label: priority_map[label])

    def _merge_updates_into_target(self, base_target_df: pd.DataFrame, updates_df: pd.DataFrame) -> pd.DataFrame:
        """부분 판정 결과를 전체 target 데이터프레임에 덮어쓴다."""
        if base_target_df.empty:
            return pd.DataFrame()
        if updates_df.empty:
            return base_target_df

        agg_map = {
            "dpm_judge": ("dpm_judge", lambda s: self._select_highest_priority(s, JUDGE_PRIORITY)),
            "dpm_controlled_judge": (
                "dpm_controlled_judge",
                lambda s: self._select_highest_priority(s, CONTROLLED_JUDGE_PRIORITY),
            ),
            "dpm_rank": ("dpm_rank", "min"),
            "dpm_score": ("dpm_score", "mean"),
            "dpm_controlled_value": ("dpm_controlled_value", "mean"),
            "lot_control_value": ("lot_control_value", "mean"),
            "outlier_status": ("outlier_status", "first"),
            "control_group": ("control_group", "first"),
            "control_entity": ("control_entity", "first"),
        }
        agg_map = {key: value for key, value in agg_map.items() if value[0] in updates_df.columns}
        aggregated_df = updates_df.groupby("_target_row_id", as_index=False).agg(**agg_map)

        merged_df = base_target_df.merge(
            aggregated_df,
            on="_target_row_id",
            how="left",
            suffixes=("", "_upd"),
        )

        for col in [
            "dpm_judge",
            "dpm_controlled_judge",
            "dpm_rank",
            "dpm_score",
            "dpm_controlled_value",
            "lot_control_value",
            "outlier_status",
            "control_group",
            "control_entity",
        ]:
            update_col = f"{col}_upd"
            if update_col in merged_df.columns:
                merged_df[col] = merged_df[update_col].combine_first(merged_df.get(col))
                merged_df = merged_df.drop(columns=[update_col])

        return merged_df

    def _summarize_box_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """box plot 옆에 함께 보여줄 요약 통계를 계산한다."""
        if df.empty:
            return pd.DataFrame(columns=["series", "count", "avg", "med", "std"])

        summary_rows = []
        for series_name, value_col in [("Target", "fab_value"), ("DPM Controlled", "dpm_controlled_value")]:
            if value_col not in df.columns:
                continue
            values = pd.to_numeric(df[value_col], errors="coerce").dropna()
            summary_rows.append(
                {
                    "series": series_name,
                    "count": int(values.count()),
                    "avg": float(values.mean()) if not values.empty else None,
                    "med": float(values.median()) if not values.empty else None,
                    "std": float(values.std()) if not values.empty else None,
                }
            )
        return pd.DataFrame(summary_rows)

    def _build_lot_control_result(
        self,
        *,
        equipment_col: str,
        result_key: str,
        title: str,
    ) -> ControlSectionResult:
        """Lot-EQP 또는 Lot-Chamber 제어 결과를 계산한다."""
        lot_representative_df = self._build_lot_representative_df()
        lot_control_df, lot_meta = self._classify_outliers(lot_representative_df, self.lot_value_col)
        lot_outlier_df = lot_control_df[lot_control_df["outlier_status"] != "normal"].copy()

        joined_df = self._build_joined_wafer_df()
        positive_correlation = self._is_positive_correlation(joined_df)
        incoming_avg = float(pd.to_numeric(self.incoming_df.get("fab_value"), errors="coerce").dropna().mean()) if not self.incoming_df.empty and "fab_value" in self.incoming_df.columns else 0.0
        target_avg = float(pd.to_numeric(self.target_df.get("fab_value"), errors="coerce").dropna().mean()) if not self.target_df.empty and "fab_value" in self.target_df.columns else 0.0

        base_target_df = self._initialize_target_result_df()
        update_frames: list[pd.DataFrame] = []
        window_frames: list[pd.DataFrame] = []

        for _, lot_row in lot_outlier_df.iterrows():
            lot_id = lot_row.get("lot_id")
            if not lot_id or base_target_df.empty or "lot_id" not in base_target_df.columns:
                continue

            lot_target_df = base_target_df[base_target_df["lot_id"].astype(str) == str(lot_id)].copy()
            if lot_target_df.empty:
                continue

            event_time = pd.to_datetime(lot_target_df["tkout_time"], errors="coerce").max()
            incoming_value = _safe_float(lot_row.get(self.lot_value_col))
            if pd.isna(event_time) or incoming_value is None:
                continue

            scored_window_df = self._score_window_summary(
                self._build_window_summary(event_time, equipment_col),
                incoming_value=incoming_value,
                incoming_avg=incoming_avg,
                target_avg=target_avg,
                positive_correlation=positive_correlation,
            )

            wow_med = scored_window_df.loc[scored_window_df["dpm_judge"] == "WOW", "fab_value_med"]
            bob_med = scored_window_df.loc[scored_window_df["dpm_judge"] == "BOB", "fab_value_med"]
            correction_value = float(wow_med.mean() - bob_med.mean()) if (not wow_med.empty and not bob_med.empty) else 0.0

            lot_target_df["lot_control_value"] = incoming_value
            lot_target_df["outlier_status"] = lot_row.get("outlier_status")
            lot_target_df = self._apply_equipment_judge_to_target_rows(
                lot_target_df,
                scored_window_df,
                equipment_col=equipment_col,
                correction_value=correction_value,
            )
            lot_target_df["control_group"] = result_key
            lot_target_df["control_entity"] = lot_id
            update_frames.append(lot_target_df)

            if not scored_window_df.empty:
                scored_window_df = scored_window_df.copy()
                scored_window_df["control_group"] = result_key
                scored_window_df["control_entity"] = lot_id
                window_frames.append(scored_window_df)

        updates_df = pd.concat(update_frames, ignore_index=True) if update_frames else pd.DataFrame()
        full_target_df = self._merge_updates_into_target(base_target_df, updates_df)
        window_summary_df = pd.concat(window_frames, ignore_index=True) if window_frames else pd.DataFrame()

        return ControlSectionResult(
            title=title,
            incoming_outlier_df=lot_control_df,
            target_trend_df=full_target_df,
            controlled_trend_df=full_target_df.copy() if not full_target_df.empty else pd.DataFrame(),
            boxplot_df=full_target_df.copy() if not full_target_df.empty else pd.DataFrame(),
            boxplot_stats_df=self._summarize_box_metrics(full_target_df),
            window_summary_df=window_summary_df,
            value_col=self.lot_value_col,
            equipment_col=equipment_col,
            meta=lot_meta,
        )

    def _build_wafer_chamber_result(self) -> ControlSectionResult:
        """Wafer-Chamber 제어 결과를 계산한다."""
        wafer_control_df, wafer_meta = self._classify_outliers(self.incoming_df, "fab_value")
        joined_df = self._build_joined_wafer_df()
        base_target_df = self._initialize_target_result_df()

        if joined_df.empty:
            return ControlSectionResult(
                title="Wafer - Chamber 제어",
                incoming_outlier_df=wafer_control_df,
                target_trend_df=base_target_df,
                controlled_trend_df=base_target_df.copy() if not base_target_df.empty else pd.DataFrame(),
                boxplot_df=base_target_df.copy() if not base_target_df.empty else pd.DataFrame(),
                boxplot_stats_df=self._summarize_box_metrics(base_target_df),
                window_summary_df=pd.DataFrame(),
                value_col="fab_value",
                equipment_col="eqp_chamber",
                meta=wafer_meta,
            )

        status_cols = wafer_control_df[["root_lot_id", "wafer_id", "outlier_status", "fab_value"]].copy()
        joined_df = joined_df.merge(status_cols, on=["root_lot_id", "wafer_id"], how="left", suffixes=("", "_classified"))
        joined_outlier_df = joined_df[joined_df["outlier_status"] != "normal"].copy()

        positive_correlation = self._is_positive_correlation(joined_df)
        incoming_avg = float(pd.to_numeric(self.incoming_df.get("fab_value"), errors="coerce").dropna().mean()) if not self.incoming_df.empty and "fab_value" in self.incoming_df.columns else 0.0
        target_avg = float(pd.to_numeric(self.target_df.get("fab_value"), errors="coerce").dropna().mean()) if not self.target_df.empty and "fab_value" in self.target_df.columns else 0.0

        update_frames: list[pd.DataFrame] = []
        window_frames: list[pd.DataFrame] = []
        root_corrections: defaultdict[str, list[float]] = defaultdict(list)

        for _, wafer_row in joined_outlier_df.iterrows():
            root_lot_id = wafer_row.get("root_lot_id")
            wafer_id = wafer_row.get("wafer_id")
            incoming_value = _safe_float(wafer_row.get("fab_value_incoming"))
            event_time = wafer_row.get("tkout_time_target")
            if root_lot_id is None or incoming_value is None or pd.isna(event_time):
                continue

            root_target_df = base_target_df[base_target_df["root_lot_id"].astype(str) == str(root_lot_id)].copy()
            if root_target_df.empty:
                continue

            scored_window_df = self._score_window_summary(
                self._build_window_summary(event_time, "eqp_chamber"),
                incoming_value=incoming_value,
                incoming_avg=incoming_avg,
                target_avg=target_avg,
                positive_correlation=positive_correlation,
            )

            wow_med = scored_window_df.loc[scored_window_df["dpm_judge"] == "WOW", "fab_value_med"]
            bob_med = scored_window_df.loc[scored_window_df["dpm_judge"] == "BOB", "fab_value_med"]
            correction_value = float(wow_med.mean() - bob_med.mean()) if (not wow_med.empty and not bob_med.empty) else 0.0
            root_corrections[str(root_lot_id)].append(correction_value)

            root_target_df["source_wafer_id"] = wafer_id
            root_target_df["incoming_wafer_value"] = incoming_value
            root_target_df["outlier_status"] = wafer_row.get("outlier_status")
            root_target_df = self._apply_equipment_judge_to_target_rows(
                root_target_df,
                scored_window_df,
                equipment_col="eqp_chamber",
                correction_value=0.0,
            )
            root_target_df["control_group"] = "wafer_chamber"
            root_target_df["control_entity"] = f"{root_lot_id}:{wafer_id}"
            update_frames.append(root_target_df)

            if not scored_window_df.empty:
                scored_window_df = scored_window_df.copy()
                scored_window_df["control_group"] = "wafer_chamber"
                scored_window_df["control_entity"] = f"{root_lot_id}:{wafer_id}"
                scored_window_df["source_wafer_id"] = wafer_id
                window_frames.append(scored_window_df)

        updates_df = pd.concat(update_frames, ignore_index=True) if update_frames else pd.DataFrame()
        full_target_df = self._merge_updates_into_target(base_target_df, updates_df)

        if not full_target_df.empty:
            correction_map = {
                root_lot_id: (sum(values) / len(values) if values else 0.0)
                for root_lot_id, values in root_corrections.items()
            }
            full_target_df["root_lot_correction"] = full_target_df["root_lot_id"].astype(str).map(correction_map).fillna(0.0)
            wow_mask = full_target_df["dpm_judge"] == "WOW"
            full_target_df.loc[wow_mask, "dpm_controlled_value"] = (
                full_target_df.loc[wow_mask, "fab_value"] - full_target_df.loc[wow_mask, "root_lot_correction"]
            )
            full_target_df.loc[wow_mask, "dpm_controlled_judge"] = "DPM_Controlled"

        window_summary_df = pd.concat(window_frames, ignore_index=True) if window_frames else pd.DataFrame()
        return ControlSectionResult(
            title="Wafer - Chamber 제어",
            incoming_outlier_df=wafer_control_df,
            target_trend_df=full_target_df,
            controlled_trend_df=full_target_df.copy() if not full_target_df.empty else pd.DataFrame(),
            boxplot_df=full_target_df.copy() if not full_target_df.empty else pd.DataFrame(),
            boxplot_stats_df=self._summarize_box_metrics(full_target_df),
            window_summary_df=window_summary_df,
            value_col="fab_value",
            equipment_col="eqp_chamber",
            meta=wafer_meta,
        )


def run_dpm_calculation(
    incoming_df: pd.DataFrame,
    target_df: pd.DataFrame,
    inputs: dict,
    df_side: pd.DataFrame | None = None,
    df_l2: pd.DataFrame | None = None,
    df_l3: pd.DataFrame | None = None,
) -> dict:
    """계산 객체를 만들고 최종 결과를 사전 형태로 반환한다."""
    calculator = DpmCalculator(
        incoming_df,
        target_df,
        DpmInputModel.from_dict(inputs),
        df_side=df_side,
        df_l2=df_l2,
        df_l3=df_l3,
    )
    return calculator.run().to_dict()
