# pysi/core/wom_state.py
# WOM準拠の計画状態管理クラス (GUI分離版, V0R7 PySI継承)

from pysi.utils.config import Config
from pysi.network.node_base import Node  # Nodeクラスインポート (メソッド使用のため)
from pysi.network.tree import traverse_tree, build_tree_from_csv, make_E2E_positions
from pysi.plan.demand_generate import convert_monthly_to_weekly
from pysi.plan.operations import *  # operations.pyの関数 (shiftS2P_LVなど)
from pysi.utils.file_io import load_csv, load_monthly_demand  # CSVロード用
# 他のimport (calendar445など) を追加

class WOMState:
    def __init__(self, config: Config):
        self.config = config
        self.initialize_parameters()

        # V0R7の計画変数
        self.product_name_list = []  # 製品リスト
        self.product_selected = None

        self.tree_structure = None

        # PSI planner
        self.outbound_data = None
        self.inbound_data = None

        self.root_node_outbound = None
        self.nodes_outbound = None
        self.leaf_nodes_out = []

        self.root_node_inbound = None
        self.nodes_inbound = None
        self.leaf_nodes_in = []

        self.root_node_out_opt = None
        self.nodes_out_opt = None
        self.leaf_nodes_opt = []

        self.optimized_root = None
        self.optimized_nodes = None

        self.node_psi_dict_In4Dm = {}  # 需要側 PSI 辞書
        self.node_psi_dict_In4Sp = {}  # 供給側 PSI 辞書

        # Evaluation
        self.total_revenue = 0
        self.total_profit = 0
        self.profit_ratio = 0

        # 製品別ツリー
        self.prod_tree_dict_IN = {}
        self.prod_tree_dict_OT = {}

        # ビュー/最適化
        self.select_node = None
        self.G = None
        self.pos_E2E = None
        self.flowDict_opt = {}
        self.flowCost_opt = {}
        self.total_supply_plan = 0

        self.base_leaf_name = {}  # {product_name: leaf_node_name}

        # supply_plan / decoupling / buffer stock
        self.decouple_node_dic = {}
        self.decouple_node_selected = []

        # ファイル/ディレクトリ
        self.directory = None
        self.load_directory = None

        # WOM拡張 (記事の3層対応)
        self.physical_layer = {}  # 物理資源 (e.g., 工場/倉庫データ)
        self.planning_layer = {}  # PSI計画データ (psi_dictなど)
        self.management_kpis = {}  # KPI (CO2排出, サービス率など)

    def initialize_parameters(self):
        # V0R7のデフォルト設定 (configからロード)
        self.lot_size = self.config.DEFAULT_LOT_SIZE
        self.plan_year_st = self.config.DEFAULT_START_YEAR
        self.plan_range = self.config.DEFAULT_PLAN_RANGE
        self.pre_proc_LT = self.config.DEFAULT_PRE_PROC_LT
        self.market_potential = self.config.DEFAULT_MARKET_POTENTIAL
        self.target_share = self.config.DEFAULT_TARGET_SHARE
        self.total_supply = self.config.DEFAULT_TOTAL_SUPPLY
        # WOM追加: 記事準拠
        self.co2_emission_limit = 0  # サステナビリティKPI例

    # V0R7の計画メソッド (GUIなし版)
    def demand_planning4multi_product(self):
        # V0R7コードをコピー/適応 (messageboxをprint/loggerに置換)
        for prod in self.product_name_list:
            # ... (demand_fileロード, convert_monthly_to_weekly, set_S2psiなど)
            print(f"[INFO] Demand planning completed for {prod}")
            # WOM拡張: planning_layer更新
            self.planning_layer[prod] = "Demand set"

    def supply_planning4multi_product(self):
        # V0R7コードをコピー/適応 (calcS2P, get_set_childrenP2S2psi, shiftS2P_LVなど)
        for prod in self.product_name_list:
            # ...
            print(f"[INFO] Supply planning completed for {prod}")
            # WOM拡張: management_kpis更新 (e.g., service_rate計算)
            self.management_kpis['service_rate'] = 96.82  # 記事ログ例

    # 他のV0R7メソッド (load_data_files, update_total_supply_planなど) を追加/適応
    # e.g., def load_data_files(self, directory): ...
        self.directory = directory
        # V0R7 load_data_filesコード移植 (CSVロード, ツリー構築など)
        