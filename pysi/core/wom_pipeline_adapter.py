# pysi/core/wom_pipeline_adapter.py
"""
Adapter that turns the original wom_main.py procedural main()
into a HOOK/PLUGIN/PIPELINE-style runnable unit.

Usage:
  from pysi.core.wom_pipeline_adapter import run_pipeline_from_dir
  run_pipeline_from_dir(data_dir, product=None)

This wrapper:
 - provides a minimal HookBus (use real one if available)
 - provides an IO adapter that builds the PlanNode tree using WOMEnv
 - runs the three planning steps: demand_planning4multi_product,
   demand_leveling4multi_prod, supply_planning4multi_product
 - collects a minimal result dict (root node plus some metadata)
"""
from typing import Any, Dict, Optional
import logging
import os

# try to reuse project's HookBus if available
try:
    from pysi.core.hooks.core import HookBus  # real implementation
except Exception:
    # Minimal fallback HookBus for extension points (no-op)
    class HookBus:
        def __init__(self):
            self._filters = {}
            self._actions = {}

        def add_filter(self, name, func):
            self._filters.setdefault(name, []).append(func)

        def apply_filters(self, name, value, **kwargs):
            for f in self._filters.get(name, []):
                value = f(value, **kwargs)
            return value

        def add_action(self, name, func):
            self._actions.setdefault(name, []).append(func)

        def do_action(self, name, **kwargs):
            for f in self._actions.get(name, []):
                f(**kwargs)


# Import your existing classes (WOMEnv, Config) from wom_main.py location
# Adjust import path if your package layout differs.
from pysi.utils.config import Config
# WOMEnv is defined in your uploaded wom_main.py
from pysi.wom_main import WOMEnv  # path may be pysi.wom_main depending on package

logger = logging.getLogger(__name__)


class WOMIOAdapter:
    """
    IO adapter providing the build_tree / collect_result methods
    expected by pipeline-style runners.
    This adapter wraps WOMEnv and returns the root PlanNode.
    """
    def __init__(self, data_dir: Optional[str] = None, product: Optional[str] = None):
        self.data_dir = data_dir
        self.product = product

    def build_tree(self, raw: Dict[str, Any], logger=None):
        """
        raw is expected to be a dict containing db_path / scenario params in
        pipeline.py. We'll map db_path -> data_dir here.
        """
        cfg = Config()
        if self.data_dir:
            # prefer explicit adapter data_dir
            cfg.DATA_DIRECTORY = self.data_dir
        else:
            # try raw input
            db_path = raw.get("db_path")
            if db_path:
                cfg.DATA_DIRECTORY = db_path

        # instantiate WOMEnv and load data files (this mirrors original main())
        env = WOMEnv(cfg)
        env.load_data_files()

        # Optionally set selected product if provided
        if self.product:
            if self.product in getattr(env, "product_name_list", []):
                env.product_selected = self.product
            else:
                logger and logger.warning("Requested product %s not found; using default", self.product)

        # Build and return root (outbound). For multi-product flows, code
        # in WOMEnv uses .prod_tree_dict_OT / product_selected; we return
        # the outbound root node so downstream code expects a PlanNode.
        # Prefer product-scoped root if available.
        root = None
        try:
            # prefer per-product root if present
            prod = env.product_selected
            if prod and prod in getattr(env, "prod_tree_dict_OT", {}):
                root = env.prod_tree_dict_OT[prod]
                # also attach the env for later use (so plugins/hook code can access)
                setattr(root, "_wom_env", env)
            else:
                # fallback: entire outbound root (if GUI nodes exist)
                root = getattr(env, "root_node_outbound", None)
                if root is not None:
                    setattr(root, "_wom_env", env)
        except Exception as e:
            logger and logger.exception("build_tree adapter failure: %s", e)
            raise

        # keep a reference so collect_result can access env if needed
        self._env = env
        return root

    def collect_result(self, root, params: Dict[str, Any] = None):
        """Collect minimal results: root reference + some evaluation numbers"""
        env = getattr(self, "_env", None)
        r = {
            "root": root,
            "env": env,
            "product_selected": getattr(env, "product_selected", None),
            "total_revenue": getattr(env, "total_revenue", None),
            "total_profit": getattr(env, "total_profit", None),
        }
        return r

    def to_series_df(self, result, horizon: int = None):
        # optional: produce time-series DataFrame for visualization
        # keep simple and safe (return empty)
        return None


class SimplePipelineRunner:
    """
    Small pipeline runner that executes the same three steps as original main().
    This mirrors the structure in pysi/core/pipeline.py but keeps it minimal and
    compatible with existing WOMEnv methods.
    """
    def __init__(self, hooks: HookBus, io_adapter: WOMIOAdapter, logger=None):
        self.hooks = hooks
        self.io = io_adapter
        self.logger = logger or logging.getLogger("wom_pipeline")

    def run(self, db_path: str, scenario_id: str = "default", calendar: Dict[str, Any] = None, out_dir: str = "out"):
        # 1) timebase hooks (if any)
        calendar = calendar or {}
        calendar = self.hooks.apply_filters("timebase:calendar:build", calendar, db_path=db_path, scenario_id=scenario_id)

        # 2) data load hooks
        self.hooks.do_action("before_data_load", db_path=db_path, scenario_id=scenario_id)

        # 3) build tree (wraps WOMEnv.load_data_files)
        spec = {"db_path": db_path, "scenario_id": scenario_id}
        spec = self.hooks.apply_filters("scenario:preload", spec, db_path=db_path, scenario_id=scenario_id)
        root = self.io.build_tree(raw=spec, logger=self.logger)
        if root is None:
            raise RuntimeError("build_tree returned None (no plan tree built)")

        # attach root to wom_state-like place if desirable
        # -- many components expect wom_state or env; we attached the WOMEnv to root._wom_env

        # 4) plan steps (call WOMEnv methods through attached env)
        env = getattr(root, "_wom_env", None)
        if env is None:
            # best-effort: maybe env stored elsewhere
            self.logger.warning("No WOMEnv attached to root; attempting to find env in IO adapter")
            env = getattr(self.io, "_env", None)

        # Run the three steps present in the original main()
        # demand_planning4multi_product -> demand_leveling4multi_prod -> supply_planning4multi_product
        try:
            if hasattr(env, "demand_planning4multi_product"):
                env.demand_planning4multi_product()
            else:
                self.logger.warning("env has no demand_planning4multi_product")

            if hasattr(env, "demand_leveling4multi_prod"):
                env.demand_leveling4multi_prod()
            else:
                self.logger.warning("env has no demand_leveling4multi_prod")

            if hasattr(env, "supply_planning4multi_product"):
                env.supply_planning4multi_product()
            else:
                self.logger.warning("env has no supply_planning4multi_product")
        except Exception:
            self.logger.exception("planning steps failed")
            raise

        # call before_plan hook
        self.hooks.do_action("before_plan", root=root, db_path=db_path, scenario_id=scenario_id)

        # 5) collect results
        result = self.io.collect_result(root, params={})
        result = self.hooks.apply_filters("viz:series", result, calendar=calendar)

        # 6) exporters (no-op by default)
        exporters = self.hooks.apply_filters("report:exporters", [])
        for ex in exporters or []:
            try:
                ex(result=result, out_dir=out_dir, logger=self.logger)
            except Exception:
                self.logger.exception("report exporter failed")

        return result


# ----------------------------
# convenience function to run
# ----------------------------
def run_pipeline_from_dir(data_dir: str, product: Optional[str] = None, logger=None):
    hooks = HookBus()
    io = WOMIOAdapter(data_dir, product=product)
    runner = SimplePipelineRunner(hooks, io, logger=logger)
    # db_path param expected by older pipeline -- pass data_dir for compatibility
    res = runner.run(db_path=data_dir, scenario_id="default", calendar={"run_id": "run_01"})
    return res


# If run as script, provide CLI-like behavior
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", "-d", required=True, help="Path to data directory (CSV etc.)")
    ap.add_argument("--product", "-p", required=False, help="Product to select (optional)")
    args = ap.parse_args()
    r = run_pipeline_from_dir(args.data_dir, product=args.product, logger=logger)
    print("Pipeline finished. total_revenue:", r.get("total_revenue"), " total_profit:", r.get("total_profit"))

