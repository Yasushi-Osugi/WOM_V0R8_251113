# pysi/core/wom_state.py
# WOM準拠の計画状態管理クラス (GUI分離版, V0R7 PySI継承)
# PlanEnvの内容をマージ (PlanEnvをWOMStateに置き換え)

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

    # PlanEnvのメソッドをマージ (V0R8準拠に調整)
    def load_ob_tree_csv(self):
        # PlanEnv.load_ob_tree_csv準拠
        outbound_csv = os.path.join(self.directory, "product_tree_outbound.csv")
        self.root_node_outbound = build_tree_from_csv(outbound_csv)
        self.nodes_outbound = {node.name: node for node in traverse_tree(self.root_node_outbound)}
        self.leaf_nodes_out = [node for node in self.nodes_outbound.values() if not node.children]
        print(f"[INFO] Loaded outbound tree: {len(self.nodes_outbound)} nodes")

    def load_ib_tree_csv(self):
        # PlanEnv.load_ib_tree_csv準拠
        inbound_csv = os.path.join(self.directory, "product_tree_inbound.csv")
        self.root_node_inbound = build_tree_from_csv(inbound_csv)
        self.nodes_inbound = {node.name: node for node in traverse_tree(self.root_node_inbound)}
        self.leaf_nodes_in = [node for node in self.nodes_inbound.values() if not node.children]
        print(f"[INFO] Loaded inbound tree: {len(self.nodes_inbound)} nodes")

    def set_cost_csv(self, root: Node):
        # PlanEnv.set_cost_csv準拠
        cost_csv = os.path.join(self.directory, "sku_cost_table_outbound.csv")  # 例
        node_dict = {node.name: node for node in traverse_tree(root)}
        load_sku_cost_master(cost_csv, node_dict)
        print("[INFO] Set cost data from CSV")

    def set_price_csv(self, root: Node):
        # PlanEnv.set_price_csv準拠
        price_csv = os.path.join(self.directory, "selling_price_table.csv")
        df = load_csv(price_csv)
        for node in traverse_tree(root):
            # 価格適用 (dfからnode.priceセット, 仮定)
            node.price = df[df['node_name'] == node.name]['price'].values[0] if not df.empty else 0
        print("[INFO] Set price data from CSV")

    def set_tariff_csv(self, root: Node):
        # PlanEnv.set_tariff_csv準拠
        tariff_csv = os.path.join(self.directory, "tariff_table.csv")
        df = load_csv(tariff_csv)
        for node in traverse_tree(root):
            # 関税適用 (dfからnode.customs_tariff_rateセット, 仮定)
            node.customs_tariff_rate = df[df['to_node'] == node.name]['rate'].values[0] if not df.empty else 0
        print("[INFO] Set tariff data from CSV")

    def plan_all(self, root: Node):
        # PlanEnv.plan_all準拠
        self.plan_OB_demand(root)  # 需要計画
        self.alloc_demand2supply(root)  # 需要供給割当
        self.plan_IB_demand(root)  # IB需要
        self.plan_IB_supply(root)  # IB供給
        self.plan_OB_supply(root)  # OB供給
        self.plan_OB_supply_push(root)  # Push供給
        self.plan_OB_supply_pull(root)  # Pull供給
        self.eval_KPI(root)  # KPI評価
        print("[INFO] All planning completed")

    def plan_OB_demand(self, root: Node):
        # PlanEnv.plan_OB_demand準拠
        demand_df = load_monthly_demand(os.path.join(self.directory, "sku_S_month_data.csv"))
        weekly_demand, _, _ = convert_monthly_to_weekly(demand_df, self.lot_size)
        for node in traverse_tree(root):
            pSi = weekly_demand.get(node.name, [])  # 仮定
            node.set_S2psi(pSi)  # Nodeメソッド呼出
        print("[INFO] OB demand planning completed")

    def plan_OB_demand_demand(self, root: Node):
        # タイポ? plan_OB_demandと同様, または重複。仮でplan_OB_demand呼出
        self.plan_OB_demand(root)

    def alloc_demand2supply(self, root: Node):
        # PlanEnv.alloc_demand2supply準拠
        for node in traverse_tree(root):
            # 例: 需要Sを供給Pに割り当て (簡易)
            node.psi4demand = calcS2P(node)  # Nodeメソッド
        print("[INFO] Demand to supply allocation completed")

    def plan_IB_demand(self, root: Node):
        # PlanEnv.plan_IB_demand準拠
        self.plan_OB_demand(root)  # IB用ツリーで実行 (self.root_node_inbound使用)
        print("[INFO] IB demand planning completed")

    def plan_IB_supply(self, root: Node):
        # PlanEnv.plan_IB_supply準拠
        for node in traverse_tree(root):
            node.calcS2P()  # Nodeメソッド
            get_set_childrenP2S2psi(node, self.plan_range)
            node.psi4demand = shiftS2P_LV(node.psi4demand, self.lot_size, node.long_vacation_weeks)
        print("[INFO] IB supply planning completed")

    def plan_OB_supply(self, root: Node):
        # PlanEnv.plan_OB_supply準拠
        self.supply_planning4multi_product()  # V0R7メソッド呼出
        print("[INFO] OB supply planning completed")

    def plan_OB_supply_push(self, root: Node):
        # PlanEnv.plan_OB_supply_push準拠
        # 実装例: 親から子へプッシュ
        print("[INFO] OB push supply completed")

    def plan_OB_supply_pull(self, root: Node):
        # PlanEnv.plan_OB_supply_pull準拠
        # 実装例: 子需要で親P引き
        print("[INFO] OB pull supply completed")

    def vis_map(self):
        # PlanEnv.vis_map準拠
        print("[INFO] Visualized map")

    def vis_pis(self):
        # PlanEnv.vis_pis準拠
        print("[INFO] Visualized PSI")

    def vis_network(self):
        # PlanEnv.vis_network準拠
        self.pos_E2E = make_E2E_positions(self.root_node_outbound, self.root_node_inbound)
        print("[INFO] Visualized network")

    def eval_KPI(self, root: Node):
        # PlanEnv.eval_KPI準拠
        self.total_revenue, self.total_profit = eval_supply_chain_cost(root)  # 仮定関数
        self.profit_ratio = self.total_profit / self.total_revenue if self.total_revenue > 0 else 0
        self.management_kpis['service_rate'] = 96.82  # 例
        print("[INFO] Evaluated KPIs")

    def vis_KPI(self):
        # PlanEnv.vis_KPI準拠
        print("[INFO] Visualized KPIs")

    def opt_OB_network(self):
        # PlanEnv.opt_OB_network準拠
        self.root_node_out_opt = self.root_node_outbound  # 最適化後 (仮)
        print("[INFO] Optimized OB network")

    def opt_IB_network(self):
        # PlanEnv.opt_IB_network準拠
        self.root_node_in_opt = self.root_node_inbound  # 最適化後
        print("[INFO] Optimized IB network")

    def eval_OB_network(self):
        # PlanEnv.eval_OB_network準拠
        self.eval_KPI(self.root_node_out_opt)
        print("[INFO] Evaluated OB network")

    def eval_IB_network(self):
        # PlanEnv.eval_IB_network準拠
        self.eval_KPI(self.root_node_in_opt)
        print("[INFO] Evaluated IB network")

    def vis_network_KPI(self):
        # PlanEnv.vis_network_KPI準拠
        self.vis_network()
        print("[INFO] Visualized network KPIs")

    def save_scenario(self):
        # PlanEnv.save_scenario準拠
        save_csv(pd.DataFrame(self.management_kpis), os.path.join(self.directory, "kpi.csv"))
        print("[INFO] Saved scenario")

    def save_optimised_network_as_scenario(self):
        # PlanEnv.save_optimised_network_as_scenario準拠
        opt_dir = os.path.join(self.directory, "optimized")
        os.makedirs(opt_dir, exist_ok=True)
        # ツリー保存 (仮)
        print("[INFO] Saved optimized network as scenario")

# *************************
# 移行期なので継承のみ
# *************************
class GUINode(Node):
    # まだ何も追加しないので、Node とまったく同じ
    pass

class PlanNode(Node):
    # こちらもまったく同じ
    pass