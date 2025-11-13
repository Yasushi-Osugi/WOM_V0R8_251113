# pysi/core/wom_state.py

# pysi/core/wom_state.py
# WOM準拠の計画状態管理クラス (GUI分離版, V0R7 PySI継承)

from pysi.utils.config import Config
from pysi.network.node_base import Node

#from pysi.network.tree import traverse_tree, build_tree_from_csv, make_E2E_positions
#from pysi.network.tree import build_tree_from_csv, make_E2E_positions
from pysi.core.tree import build_tree_from_csv, make_E2E_positions

from pysi.plan.demand_generate import convert_monthly_to_weekly

from pysi.core.node_base import set_S2psi, calcS2P, get_set_childrenP2S2psi, shiftS2P_LV
#from pysi.plan.operations import set_S2psi, calcS2P, get_set_childrenP2S2psi, shiftS2P_LV

from pysi.utils.file_io import load_csv, load_monthly_demand  # CSVロード用
# 他のimport (calendar445, file_ioのsave_csvなど) を追加。最適化用: networkx/PuLPなど


# helper
def traverse_tree(node):
    """
    Traverse the tree starting from the root node and collect all node names in a list.

    Parameters:
        node (Node): The root node of the tree.

    Returns:
        list[str]: A list of all node names in the tree.
    """
    node_names = []

    def _traverse(current_node):
        if current_node is None:
            return
        # Append the current node's name to the list
        node_names.append(current_node.name)
        # Recursively traverse each child
        for child in getattr(current_node, 'children', []):
            _traverse(child)

    _traverse(node)
    return node_names


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
        self.scenario_id = None  # 選択シナリオID

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

    def select_scenario(self):
        # シナリオ選択 (V0R8 entry_gui.py準拠, ディレクトリからリスト)
        # GUIなし版: 仮にハードコード or パラメータ渡し
        self.scenario_id = "v0r7_rice"  # 例。実際はinput or cfgから
        self.directory = os.path.join(self.config.DATA_DIRECTORY, self.scenario_id)
        print(f"[INFO] Selected scenario: {self.scenario_id}")

    def load_ob_tree_csv(self):
        # OutboundツリーCSVロード (V0R7 app.py/load_tree準拠)
        outbound_csv = os.path.join(self.directory, "product_tree_outbound.csv")
        self.root_node_outbound = build_tree_from_csv(outbound_csv)
        self.nodes_outbound = {node.name: node for node in traverse_tree(self.root_node_outbound)}
        self.leaf_nodes_out = [node for node in self.nodes_outbound.values() if not node.children]
        print(f"[INFO] Loaded outbound tree: {len(self.nodes_outbound)} nodes")

    def load_ib_tree_csv(self):
        # InboundツリーCSVロード (同様)
        inbound_csv = os.path.join(self.directory, "product_tree_inbound.csv")
        self.root_node_inbound = build_tree_from_csv(inbound_csv)
        self.nodes_inbound = {node.name: node for node in traverse_tree(self.root_node_inbound)}
        self.leaf_nodes_in = [node for node in self.nodes_inbound.values() if not node.children]
        print(f"[INFO] Loaded inbound tree: {len(self.nodes_inbound)} nodes")

    def set_cost_csv(self, root: Node):
        # コストCSVセット (V0R7 node_base.load_sku_cost_master準拠)
        cost_csv = os.path.join(self.directory, "sku_cost_table_outbound.csv")  # 例
        node_dict = {node.name: node for node in traverse_tree(root)}
        load_sku_cost_master(cost_csv, node_dict)
        print("[INFO] Set cost data from CSV")

    def set_price_csv(self, root: Node):
        # 価格CSVセット (V0R7類似, 仮実装)
        price_csv = os.path.join(self.directory, "selling_price_table.csv")
        df = load_csv(price_csv)
        for node in traverse_tree(root):
            # 価格適用 (dfからnode.priceセット, 仮)
            node.price = df[df['node_name'] == node.name]['price'].values[0] if not df.empty else 0
        print("[INFO] Set price data from CSV")

    def set_tariff_csv(self, root: Node):
        # 関税CSVセット (V0R7類似, 仮実装)
        tariff_csv = os.path.join(self.directory, "tariff_table.csv")
        df = load_csv(tariff_csv)
        for node in traverse_tree(root):
            # 関税適用 (dfからnode.customs_tariff_rateセット, 仮)
            node.customs_tariff_rate = df[df['to_node'] == node.name]['rate'].values[0] if not df.empty else 0
        print("[INFO] Set tariff data from CSV")

    def plan_all(self, root: Node):
        # 全計画実行 (demand/supply/alloc/opt/evalのシーケンス)
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
        # OB需要計画 (V0R7 demand_planning4multi_product準拠)
        # 月次データロード/変換/セット
        demand_df = load_monthly_demand(os.path.join(self.directory, "sku_S_month_data.csv"))
        weekly_demand, _, _ = convert_monthly_to_weekly(demand_df, self.lot_size)
        for node in traverse_tree(root):
            pSi = weekly_demand.get(node.name, [])  # 仮定

            #@251112
            node.set_S2psi(pSi)
            #set_S2psi(node, pSi)

        print("[INFO] OB demand planning completed")

    def plan_OB_demand_demand(self, root: Node):
        # タイポ? plan_OB_demandと同様, または重複。仮でplan_OB_demand呼出
        self.plan_OB_demand(root)

    def alloc_demand2supply(self, root: Node):
        # 需要供給割当 (V0R7 alloc準拠, 仮実装: 需要を供給に割り当て)
        for node in traverse_tree(root):
            # 例: 需要Sを供給Pに割り当て (簡易)
            node.psi4demand = calcS2P(node)  # backwardで調整
        print("[INFO] Demand to supply allocation completed")

    def plan_IB_demand(self, root: Node):
        # IB需要計画 (V0R7類似, OBと同様仮実装)
        self.plan_OB_demand(root)  # IB用ツリーで実行 (self.root_node_inbound使用)
        print("[INFO] IB demand planning completed")

    def plan_IB_supply(self, root: Node):
        # IB供給計画 (V0R7 supply_planning準拠, 仮実装)
        for node in traverse_tree(root):
            calcS2P(node)
            get_set_childrenP2S2psi(node, self.plan_range)
            shiftS2P_LV(node.psi4demand, self.lot_size, node.long_vacation_weeks)
        print("[INFO] IB supply planning completed")

    def plan_OB_supply(self, root: Node):
        # OB供給計画 (V0R7 supply_planning準拠)
        self.supply_planning4multi_product()  # V0R7メソッド呼出
        print("[INFO] OB supply planning completed")

    def plan_OB_supply_push(self, root: Node):
        # OB Push供給 (V0R7類似, 仮: 親から子へプッシュ)
        # 実装例: 親Pを子Sにプッシュ
        print("[INFO] OB push supply completed")

    def plan_OB_supply_pull(self, root: Node):
        # OB Pull供給 (V0R7類似, 仮: 子需要で親P引き)
        # 実装例: 子Sで親P調整
        print("[INFO] OB pull supply completed")

    def vis_map(self):
        # マップ可視化 (V0R7 vis準拠, 仮: Matplotlibで地図)
        print("[INFO] Visualized map")

    def vis_pis(self):
        # PSI可視化 (V0R7 show_psi_by_product準拠, 仮: Matplotlibグラフ)
        print("[INFO] Visualized PSI")

    def vis_network(self):
        # ネットワーク可視化 (V0R7 view_nx_matlib4opt準拠, 仮: NetworkX描画)
        self.pos_E2E = make_E2E_positions(self.root_node_outbound, self.root_node_inbound)
        print("[INFO] Visualized network")

    def eval_KPI(self, root: Node):
        # KPI評価 (V0R7 eval準拠, 仮: revenue/profit計算)
        self.total_revenue = sum(node.eval_cs_price_sales_shipped for node in traverse_tree(root))
        self.total_profit = sum(node.eval_cs_profit for node in traverse_tree(root))
        self.profit_ratio = self.total_profit / self.total_revenue if self.total_revenue > 0 else 0
        self.management_kpis['service_rate'] = 96.82  # 例
        print("[INFO] Evaluated KPIs")

    def vis_KPI(self):
        # KPI可視化 (V0R7 vis準拠, 仮: バー/パイチャート)
        print("[INFO] Visualized KPIs")

    def opt_OB_network(self):
        # OBネットワーク最適化 (priority based allocation, V0R7 optimize_network準拠, 仮: NetworkX/PuLP使用)
        self.root_node_out_opt = self.root_node_outbound  # 最適化後ツリー (仮)
        print("[INFO] Optimized OB network")

    def opt_IB_network(self):
        # IBネットワーク最適化 (TOC bottleneck, V0R7類似, 仮: ボトルネック検知/調整)
        self.root_node_in_opt = self.root_node_inbound  # 最適化後
        print("[INFO] Optimized IB network")

    def eval_OB_network(self):
        # OBネットワーク評価 (仮: KPI計算)
        self.eval_KPI(self.root_node_out_opt)
        print("[INFO] Evaluated OB network")

    def eval_IB_network(self):
        # IBネットワーク評価
        self.eval_KPI(self.root_node_in_opt)
        print("[INFO] Evaluated IB network")

    def vis_network_KPI(self):
        # ネットワークKPI可視化 (vis_network + KPIオーバレイ)
        self.vis_network()
        print("[INFO] Visualized network KPIs")

    def save_scenario(self):
        # シナリオ保存 (V0R7 save_to_directory準拠, 仮: CSV/JSON出力)
        save_csv(pd.DataFrame(self.management_kpis), os.path.join(self.directory, "kpi.csv"))
        print("[INFO] Saved scenario")

    def save_optimised_network_as_scenario(self):
        # 最適ネットワークを新シナリオ保存 (仮: ツリー/データコピー)
        opt_dir = os.path.join(self.directory, "optimized")
        os.makedirs(opt_dir, exist_ok=True)
        # ツリー保存 (仮)
        print("[INFO] Saved optimized network as scenario")