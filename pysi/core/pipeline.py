# pysi/core/pipeline.py

# pysi/core/pipeline.py
# V0R8 Hook 版 Pipeline（V0R7準拠: psi_dual排除、operations/demand_generate使用）
from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd

from pysi.core.hooks.core import HookBus
# V0R7関数インポート

#@STOP
#from pysi.plan.operations import (
#from pysi.core.node_base import (
#    set_S2psi, calcS2P, get_set_childrenP2S2psi, shiftS2P_LV
#)

from pysi.plan.demand_generate import convert_monthly_to_weekly

#@STOP
#from pysi.network.tree import traverse_tree  # ツリー走査用

from pysi.core.wom_state import WOMState  # 新クラス
from pysi.utils.config import Config

#def _as_state(obj) -> Dict[str, Any]:
#    # (前回と同じ, 略)

def _as_state(obj) -> Dict[str, Any]:
    """
    Node / dict の両対応で obj.state（dict）を返す。
    無ければ空dictを作って紐づける。
    """
    if isinstance(obj, dict):
        st = obj.get("state")
        if isinstance(st, dict):
            return st
        st = {}
        obj["state"] = st
        return st

    # Node オブジェクト
    st = getattr(obj, "state", None)
    if isinstance(st, dict):
        return st
    st = {}
    setattr(obj, "state", st)
    return st





class Pipeline:
    def __init__(self, hooks: HookBus, io, logger=None):
        self.hooks, self.io, self.logger = hooks, io, logger

    def run(self, db_path: str, scenario_id: str, calendar: Dict[str, Any], out_dir: str = "out"):
        run_id = calendar.get("run_id")

        # WOMStateインスタンス作成
        config = Config()
        wom_state = WOMState(config)

        # ---- Timebase ----
        calendar = self.hooks.apply_filters(
            "timebase:calendar:build", calendar,
            db_path=db_path, scenario_id=scenario_id, logger=self.logger, run_id=run_id, wom_state=wom_state
        )

        # ---- Data Load ----
        self.hooks.do_action(
            "before_data_load",
            db_path=db_path, scenario_id=scenario_id, logger=self.logger, run_id=run_id, wom_state=wom_state
        )
        spec = {"db_path": db_path, "scenario_id": scenario_id}
        spec = self.hooks.apply_filters(
            "scenario:preload", spec,
            db_path=db_path, scenario_id=scenario_id, logger=self.logger, run_id=run_id, wom_state=wom_state
        )

        # ---- Tree Build ----
        root = self.io.build_tree(raw=spec, logger=self.logger)
        root = self.hooks.apply_filters(
            "plan:graph:build", root,
            db_path=db_path, scenario_id=scenario_id, logger=self.logger, run_id=run_id, wom_state=wom_state
        )
        wom_state.root_node_outbound = root  # 状態更新

        # ---- Plan ----
        # V0R7関数呼出 (wom_state経由)
        wom_state.demand_planning4multi_product()  # 状態更新
        wom_state.supply_planning4multi_product()  # 状態更新

        self.hooks.do_action(
            "before_plan",
            root=root, db_path=db_path, scenario_id=scenario_id, logger=self.logger, run_id=run_id, wom_state=wom_state
        )

        # ---- 結果収集 / 可視化 / 出力 ----
        result = self.io.collect_result(root, params={})
        result["psi_df"] = self.io.to_series_df(result, horizon=weeks)

        result = self.hooks.apply_filters("viz:series", result, calendar=calendar, logger=self.logger, wom_state=wom_state)

        exporters = self.hooks.apply_filters("report:exporters", [])
        for ex in exporters or []:
            try:
                ex(result=result, out_dir=out_dir, logger=self.logger, wom_state=wom_state)
            except Exception:
                if self.logger:
                    self.logger.exception("[report] exporter failed")

        return result
