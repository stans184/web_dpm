from __future__ import annotations

from collections import defaultdict

import pandas as pd

from models.dpm_models import ControlSectionResult, DpmCalculationResult, DpmInputModel
from services.dpm_calc_common import LOT_SUBITEM_TO_COLUMN, safe_float, safe_int
from services.dpm_calc_frame_ops import (
    build_joined_wafer_df,
    build_lot_representative_df,
    build_window_summary,
    initialize_target_result_df,
    is_positive_correlation,
    summarize_box_metrics,
)
from services.dpm_calc_scoring import (
    apply_equipment_judge_to_target_rows,
    classify_outliers,
    merge_updates_into_target,
    score_window_summary,
)


class DpmCalculator:
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
        return LOT_SUBITEM_TO_COLUMN.get(self.inputs.incoming_subitem, LOT_SUBITEM_TO_COLUMN["avg"])

    @property
    def use_absolute_formula(self) -> bool:
        return self.inputs.control_scope_mode == "mesh"

    @property
    def use_ascending_rank(self) -> bool:
        return self.inputs.control_scope_mode != "large"

    @property
    def window_days(self) -> int:
        return max(0, safe_int(self.inputs.window_days) or 0)

    @property
    def min_wafer_qty(self) -> int:
        return max(1, safe_int(self.inputs.min_wafer_qty) or 1)

    def run(self) -> DpmCalculationResult:
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

    def _classify_outliers(self, df: pd.DataFrame, value_col: str) -> tuple[pd.DataFrame, dict]:
        return classify_outliers(
            df,
            value_col,
            control_type=self.inputs.control_type,
            sigma=self.inputs.sigma,
            lsl=self.inputs.lsl,
            usl=self.inputs.usl,
            percentile=self.inputs.percentile,
        )

    def _score_window_summary(
        self,
        summary_df: pd.DataFrame,
        *,
        incoming_value: float,
        incoming_avg: float,
        target_avg: float,
        positive_correlation: bool,
    ) -> pd.DataFrame:
        return score_window_summary(
            summary_df,
            incoming_value=incoming_value,
            incoming_avg=incoming_avg,
            target_avg=target_avg,
            positive_correlation=positive_correlation,
            use_absolute_formula=self.use_absolute_formula,
            use_ascending_rank=self.use_ascending_rank,
            wow_portion=self.inputs.wow_portion,
        )

    def _build_window_summary(self, event_time, equipment_col: str) -> pd.DataFrame:
        return build_window_summary(
            self.target_df,
            event_time,
            equipment_col,
            window_days=self.window_days,
            min_wafer_qty=self.min_wafer_qty,
        )

    def _initialize_target_result_df(self) -> pd.DataFrame:
        return initialize_target_result_df(self.target_df)

    def _build_lot_representative_df(self) -> pd.DataFrame:
        return build_lot_representative_df(self.incoming_df)

    def _build_joined_wafer_df(self) -> pd.DataFrame:
        return build_joined_wafer_df(self.incoming_df, self.target_df)

    def _is_positive_correlation(self, joined_df: pd.DataFrame) -> bool:
        return is_positive_correlation(joined_df)

    def _summarize_box_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        return summarize_box_metrics(df)

    def _incoming_avg(self) -> float:
        if self.incoming_df.empty or "fab_value" not in self.incoming_df.columns:
            return 0.0
        return float(pd.to_numeric(self.incoming_df["fab_value"], errors="coerce").dropna().mean())

    def _target_avg(self) -> float:
        if self.target_df.empty or "fab_value" not in self.target_df.columns:
            return 0.0
        return float(pd.to_numeric(self.target_df["fab_value"], errors="coerce").dropna().mean())

    def _build_lot_control_result(
        self,
        *,
        equipment_col: str,
        result_key: str,
        title: str,
    ) -> ControlSectionResult:
        lot_representative_df = self._build_lot_representative_df()
        lot_control_df, lot_meta = self._classify_outliers(lot_representative_df, self.lot_value_col)
        lot_outlier_df = lot_control_df[lot_control_df["outlier_status"] != "normal"].copy()

        joined_df = self._build_joined_wafer_df()
        positive_correlation = self._is_positive_correlation(joined_df)
        incoming_avg = self._incoming_avg()
        target_avg = self._target_avg()

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
            incoming_value = safe_float(lot_row.get(self.lot_value_col))
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
            lot_target_df = apply_equipment_judge_to_target_rows(
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
        full_target_df = merge_updates_into_target(base_target_df, updates_df)
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
        incoming_avg = self._incoming_avg()
        target_avg = self._target_avg()

        update_frames: list[pd.DataFrame] = []
        window_frames: list[pd.DataFrame] = []
        root_corrections: defaultdict[str, list[float]] = defaultdict(list)

        for _, wafer_row in joined_outlier_df.iterrows():
            root_lot_id = wafer_row.get("root_lot_id")
            wafer_id = wafer_row.get("wafer_id")
            incoming_value = safe_float(wafer_row.get("fab_value_incoming"))
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
            root_target_df = apply_equipment_judge_to_target_rows(
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
        full_target_df = merge_updates_into_target(base_target_df, updates_df)

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
    calculator = DpmCalculator(
        incoming_df,
        target_df,
        DpmInputModel.from_dict(inputs),
        df_side=df_side,
        df_l2=df_l2,
        df_l3=df_l3,
    )
    return calculator.run().to_dict()
