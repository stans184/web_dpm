"""DPM 화면과 계산 로직에서 공통으로 사용하는 데이터 모델 모음."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd


LOT_TYPE_OPTIONS = ["ALL", "P type"]
INCOMING_SUBITEM_OPTIONS = ["avg", "std", "min", "max", "range"]
CONTROL_SCOPE_OPTIONS = [
    "망목제어",
    "망소제어 (설비 제어)",
    "망대제어 (주요 설비 제어)",
]


@dataclass(slots=True)
class DpmInputModel:
    """DPM 화면 입력값을 하나의 객체로 표현한다."""

    fab_type: str = "memory"
    line_id: list[str] = field(default_factory=list)
    process_id: list[str] = field(default_factory=list)
    incoming_prc_step: str | None = None
    incoming_metro_step: str = ""
    incoming_item: str = ""
    incoming_ppid: str = ""
    incoming_lot_type: str = "ALL"
    incoming_lot_filter: str = ""
    incoming_subitem: str = "avg"
    target_prc_step: str | None = None
    target_metro_step: str = ""
    target_item: str = ""
    target_ppid: str = ""
    target_lot_type: str = "ALL"
    side_metro_step: str | None = None
    side_metro_item: str = ""
    l2_step_seq: str = ""
    l2_item_id: str = ""
    l3_step_seq: str = ""
    l3_item_id: str = ""
    control_scope: str = CONTROL_SCOPE_OPTIONS[0]
    control_scope_mode: str = "mesh"
    from_date: object = None
    to_date: object = None
    control_type: str = "sigma"
    sigma: str = ""
    usl: str = ""
    lsl: str = ""
    percentile: str = ""
    wow_portion: str = ""
    window_days: str = ""
    min_wafer_qty: str = ""

    @classmethod
    def from_dict(cls, payload: dict | None) -> "DpmInputModel":
        """사전 형태 입력값을 모델 객체로 변환한다."""
        payload = payload or {}
        valid_keys = {field_name for field_name in cls.__dataclass_fields__}
        filtered_payload = {key: value for key, value in payload.items() if key in valid_keys}
        return cls(**filtered_payload)

    def to_dict(self) -> dict:
        """모델을 세션 저장용 사전으로 변환한다."""
        return asdict(self)


@dataclass(slots=True)
class ControlSectionResult:
    """제어 섹션 하나에 대한 계산 결과를 담는다."""

    title: str
    incoming_outlier_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    target_trend_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    controlled_trend_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    boxplot_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    boxplot_stats_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    window_summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    value_col: str = "fab_value"
    equipment_col: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """기존 뷰 코드와 호환되도록 사전 형태로 변환한다."""
        return {
            "title": self.title,
            "incoming_outlier_df": self.incoming_outlier_df,
            "target_trend_df": self.target_trend_df,
            "controlled_trend_df": self.controlled_trend_df,
            "boxplot_df": self.boxplot_df,
            "boxplot_stats_df": self.boxplot_stats_df,
            "window_summary_df": self.window_summary_df,
            "value_col": self.value_col,
            "equipment_col": self.equipment_col,
            "meta": self.meta,
        }


@dataclass(slots=True)
class DpmCalculationResult:
    """계산 서비스가 뷰에 넘겨주는 최종 결과 객체다."""

    status: str
    message: str
    incoming_rows: int
    target_rows: int
    side_rows: int
    l2_rows: int
    l3_rows: int
    fab_type: str
    control_scope: str
    control_scope_mode: str
    control_type: str
    incoming_subitem: str
    lot_value_col: str
    lot_eqp_result: ControlSectionResult
    lot_chamber_result: ControlSectionResult
    wafer_chamber_result: ControlSectionResult

    def to_dict(self) -> dict:
        """기존 코드와 호환되는 사전 형태 결과를 만든다."""
        lot_eqp_dict = self.lot_eqp_result.to_dict()
        lot_chamber_dict = self.lot_chamber_result.to_dict()
        wafer_chamber_dict = self.wafer_chamber_result.to_dict()
        return {
            "status": self.status,
            "message": self.message,
            "incoming_rows": self.incoming_rows,
            "target_rows": self.target_rows,
            "side_rows": self.side_rows,
            "l2_rows": self.l2_rows,
            "l3_rows": self.l3_rows,
            "fab_type": self.fab_type,
            "control_scope": self.control_scope,
            "control_scope_mode": self.control_scope_mode,
            "control_type": self.control_type,
            "incoming_subitem": self.incoming_subitem,
            "lot_value_col": self.lot_value_col,
            "lot_eqp_result": lot_eqp_dict,
            "lot_chamber_result": lot_chamber_dict,
            "wafer_chamber_result": wafer_chamber_dict,
            "lot_eqp_df": lot_eqp_dict["incoming_outlier_df"],
            "lot_chamber_df": lot_chamber_dict["incoming_outlier_df"],
            "wafer_chamber_df": wafer_chamber_dict["incoming_outlier_df"],
            "lot_meta": lot_eqp_dict["meta"],
            "wafer_meta": wafer_chamber_dict["meta"],
        }
