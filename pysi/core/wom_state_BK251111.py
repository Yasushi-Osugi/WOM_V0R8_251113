# pysi/core/wom_state.py
# WOM準拠の計画状態管理クラス (GUI分離版, V0R7 PySI継承)

from pysi.utils.config import Config
from pysi.network.node_base import Node

#from pysi.network.tree import traverse_tree, build_tree_from_csv, make_E2E_positions
#from pysi.network.tree import build_tree_from_csv, make_E2E_positions

from pysi.core.tree import build_tree_from_csv, make_E2E_positions

from pysi.plan.demand_generate import convert_monthly_to_weekly

#@STOP
#from pysi.plan.operations import set_S2psi, calcS2P, get_set_childrenP2S2psi, shiftS2P_LV

# 他のimport (file_io, calendar445など) を追加

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
            # ... (demand_fileロード, convert_monthly_to_weekly, set_S2psiなど。V0R7 app.pyから移植)
            print(f"[INFO] Demand planning completed for {prod}")
            # WOM拡張: planning_layer更新
            self.planning_layer[prod] = "Demand set"

    def supply_planning4multi_product(self):
        # V0R7コードをコピー/適応 (calcS2P, get_set_childrenP2S2psi, shiftS2P_LVなど)
        for prod in self.product_name_list:
            # ... (V0R7 app.pyから移植)
            print(f"[INFO] Supply planning completed for {prod}")
            # WOM拡張: management_kpis更新 (e.g., service_rate計算)
            self.management_kpis['service_rate'] = 96.82  # 記事ログ例


    # 他のV0R7メソッド (load_data_files, update_total_supply_planなど) を追加/適応
    # e.g., def load_data_files(self, directory): ...
        self.directory = directory
        # V0R7 load_data_filesコード移植 (CSVロード, ツリー構築など)

    def load_data_files(self):
        directory = filedialog.askdirectory(title="Select Data Directory")
        if directory:
            try:
                self.lot_size = int(self.lot_size_entry.get())
                self.plan_year_st = int(self.plan_year_entry.get())
                self.plan_range = int(self.plan_range_entry.get())
            except ValueError:
                print("Invalid input for lot size, plan year start, or plan range. Using default values.")
            self.outbound_data = []
            self.inbound_data = []
            data_file_list = os.listdir(directory)
            print("data_file_list", data_file_list)
            self.directory = directory
            self.load_directory = directory
            # --- Load Tree Structures ---
            if "product_tree_outbound.csv" in data_file_list:
                file_path_OT = os.path.join(directory, "product_tree_outbound.csv")
                nodes_outbound, root_node_name_out = create_tree_set_attribute(file_path_OT)
                root_node_outbound = nodes_outbound[root_node_name_out]
                def make_leaf_nodes(node, leaf_list):
                    if not node.children:
                        leaf_list.append(node.name)
                    for child in node.children:
                        make_leaf_nodes(child, leaf_list)
                    return leaf_list
                leaf_nodes_out = make_leaf_nodes(root_node_outbound, [])
                self.nodes_outbound = nodes_outbound
                self.root_node_outbound = root_node_outbound
                self.leaf_nodes_out = leaf_nodes_out
                set_positions(root_node_outbound)
                set_parent_all(root_node_outbound)
                print_parent_all(root_node_outbound)
            else:
                print("error: product_tree_outbound.csv is missed")
            if "product_tree_inbound.csv" in data_file_list:
                file_path_IN = os.path.join(directory, "product_tree_inbound.csv")
                nodes_inbound, root_node_name_in = create_tree_set_attribute(file_path_IN)
                root_node_inbound = nodes_inbound[root_node_name_in]
                self.nodes_inbound = nodes_inbound
                self.root_node_inbound = root_node_inbound
                set_positions(root_node_inbound)
                set_parent_all(root_node_inbound)
                print_parent_all(root_node_inbound)
            else:
                print("error: product_tree_inbound.csv is missed")
            # **************************************
            # join nodes_outbound and nodes_inbound
            # **************************************
            # マージ前に重複チェック（ログ出力あり）
            overlapping_keys = set(nodes_inbound) & set(nodes_outbound)
            if overlapping_keys:
                print(f"[Warn] Overlapping node names: {overlapping_keys}")
            #@STOP python 3.9 upper
            #node_dict = nodes_inbound | nodes_outbound
            # **************************************
            # this is Nodes_all for GUI handling
            # **************************************
            node_dict = {**nodes_inbound, **nodes_outbound}
            #  注意：重複キーがあると、後に出てくる辞書の値で上書きされます。
            # "supply_point"がoutboundとinboundで重複しoutboundで上書きされる
            #@250726 ココでby productのPlanNodeを生成
            # **************************************
            # make subtree by product_name from "csv files"
            # **************************************
            def build_prod_tree_from_csv(csv_data, product_name):
                node_dict = {}
                # 対象 product のみ抽出
                rows = [row for row in csv_data if row["Product_name"] == product_name]
                for row in rows:
                    p_name = row["Parent_node"]
                    c_name = row["Child_node"]
                    # ノード生成（product依存で一意）
                    if p_name not in node_dict:
                        node_dict[p_name] = PlanNode(name=p_name) #@250726MARK
                        #node_dict[p_name] = Node(name=p_name) #@250726MARK
                    if c_name not in node_dict:
                        node_dict[c_name] = PlanNode(name=c_name)
                        #node_dict[c_name] = Node(name=c_name)
                    parent = node_dict[p_name]
                    child = node_dict[c_name]
                    child.lot_size = int(row["lot_size"])
                    child.leadtime = int(row["leadtime"])
                    # SKUインスタンスを割り当て（planning用）
                    # ← PSI計算後にpsi4demandなどを持たせる
                    child.sku = SKU(product_name, child.name)
                    #@250728 STOP see "link_planning_nodes_to_gui_sku" in end_of_loading
                    ##@250728 MEMO linkage plan_node2gui_node
                    #gui_node = nodes_all[child.name]         # picking up gui_node
                    #gui_node.sku_dict[product_name] = child　# linking plan_node 2 gui_node
                    ##@250728 MEMO "plan_node = sku"となるので、planning engineはplan_nodeで良いsku無し
                    #@250726 STOP by productのPlanNodeの世界なので、node直下にskuがあり、sku_dictはxxx
                    #@250728 sku_dict[product_name] = plan_nodeとして、GUINodeとPlanNodeをlinkする
                    #        このlinkingは、plan_nodeのbuilding processで行う
                    ##@250725 MEMO setting for Multi-Product
                    #child.sku_dict[product_name] = SKU(product_name, child.name)
                    child.parent = parent
                    parent.add_child(child)
                return node_dict  # this is all nodes
                #return node_dict["supply_point"]  # root node
            prod_tree_dict_IN = {} # inbound  {product_name:subtree, ,,,}
            prod_tree_dict_OT = {} # outbound {product_name:subtree, ,,,}
            product_name_list = list(node_dict["supply_point"].sku_dict.keys())
            print("product_name_list", product_name_list)
            # initial setting product_name
            self.product_name_list = product_name_list
            self.product_selected = product_name_list[0]
            # initial setting for "Select Product" BOX UI
            self.cb_product['values'] = self.product_name_list
            if self.product_name_list:
                self.cb_product.current(0)
            prod_nodes = {} # by product tree node"s"
            product_tree_dict = {}
            for prod_nm in product_name_list:
                print("product_nm 4 subtree", prod_nm )
                #@250717 node4psi tree上のnode辞書も見えるようにしておく
                csv_data = read_csv_as_dictlist(file_path_OT)
                node4psi_dict_OT = build_prod_tree_from_csv(csv_data, prod_nm)
                # setting outbound root node
                prod_tree_root_OT = node4psi_dict_OT["supply_point"]
                prod_tree_dict_OT[prod_nm] = prod_tree_root_OT # by product root_node
                csv_data = read_csv_as_dictlist(file_path_IN)
                node4psi_dict_IN = build_prod_tree_from_csv(csv_data, prod_nm)
                # setting inbound root node
                prod_tree_root_IN = node4psi_dict_IN["supply_point"]
                prod_tree_dict_IN[prod_nm] = prod_tree_root_IN # by Product root_node
                #@250717 STOP root_nodeのみ
                #prod_tree_dict_OT[prod_nm] = build_prod_tree_from_csv(csv_data, prod_nm)
                #prod_tree_dict_IN[prod_nm] = build_prod_tree_from_csv(csv_data, prod_nm)
                #@250726 MEMO by Productでroot_nodeとnodesを生成後、
                # PlanNodeのroot_nodeからselfたどって? self.xxxとしてセットする
                def make_leaf_nodes(node, list):
                    if node.children == []: # leaf_nodeの場合
                        list.append(node.name)
                    else:
                        pass
                    for child in node.children:
                        make_leaf_nodes(child, list)
                    return list
                leaf_nodes = []
                leaf_nodes = make_leaf_nodes(prod_tree_root_OT, leaf_nodes)
                #leaf_nodes = make_leaf_nodes(root_node_out, leaf_nodes)
                #@250726 STOP このself.はGUI
                ## PlanNodeのinstanceも、self.xxx/plan_node.xxxの属性名は共通
                #leaf_nodes_out = make_leaf_nodes(prod_tree_root_OT, [])
                #
                #self.nodes_outbound = node4psi_dict_OT
                #self.root_node_outbound = prod_tree_root_OT
                #
                #self.leaf_nodes_out = leaf_nodes_out
                #
                ## このby ProductのpositionsはPlanNodeでは未使用GUINodeを使う
                ##set_positions(prod_tree_root_OT)
                #@250726 STOP 各nodeがtree全体の情報nodesを持つのは冗長
                #for node_name in list( node4psi_dict_OT.keys() ):
                #    plan_node = node4psi_dict_OT[node_name]
                #
                #    plan_node.nodes_outbound = node4psi_dict_OT
                #    plan_node.root_node_outbound = prod_tree_root_OT
                #
                #    leaf_nodes_out = make_leaf_nodes(prod_tree_root_OT, [])
                #    plan_node.leaf_nodes_out = leaf_nodes_out
                #@250726 GO
                set_parent_all(prod_tree_root_OT)
                print_parent_all(prod_tree_root_OT)
                #leaf_nodes_out = make_leaf_nodes(root_node_outbound, [])
                #self.nodes_outbound = nodes_outbound
                #self.root_node_outbound = root_node_outbound
                #self.leaf_nodes_out = leaf_nodes_out
                #set_positions(root_node_outbound)
                #set_parent_all(root_node_outbound)
                #print_parent_all(root_node_outbound)
                #@250726 STOP このself.はGUI
                ## nodes_inbound, root_node_name_in = create_tree_set_attribute(file_path_IN)
                ## root_node_inbound = nodes_inbound[root_node_name_in]
                #
                #self.nodes_inbound = node4psi_dict_IN
                #self.root_node_inbound = prod_tree_root_IN
                #
                ## このby ProductのpositionsはPlanNodeでは未使用GUINodeを使う
                ##set_positions(prod_tree_root_IN)
                #
                #@250726 STOP 各nodeがtree全体の情報nodesを持つのは冗長
                #for node_name in list( node4psi_dict_IN.keys() ):
                #    plan_node = node4psi_dict_IN[node_name]
                #
                #    plan_node.nodes_inbound = node4psi_dict_IN
                #    plan_node.root_node_inbound = prod_tree_root_IN
                #
                #    leaf_nodes_in = make_leaf_nodes(prod_tree_root_IN, [])
                #    plan_node.leaf_nodes_in = leaf_nodes_in
                #@250726 GO
                set_parent_all(prod_tree_root_IN)
                print_parent_all(prod_tree_root_IN)
                #nodes_inbound, root_node_name_in = create_tree_set_attribute(file_path_IN)
                #root_node_inbound = nodes_inbound[root_node_name_in]
                #
                #self.nodes_inbound = nodes_inbound
                #self.root_node_inbound = root_node_inbound
                #set_positions(root_node_inbound)
                #set_parent_all(root_node_inbound)
                #print_parent_all(root_node_inbound)
            # **************************
            # GUI-計算構造のリンク
            # **************************
            # 設計項目	内容
            # plan_node.name	GUIと計算ノードの一致キーは "Node.name"
            # gui_node_dict[name]	GUI上の全ノードを辞書化しておく
            #@250728 STOP
            ## sku_dict[product_name]	GUI上のSKU単位で .psi_node_refをセット
            ## psi_node_ref	計算結果（PSI/Costなど）の直接参照ポインタ
            #@250728 GO
            # sku_dict[product_name]	gui_nodeのby product(SKU単位)で、ここにplan_nodeを直接セット
            # "plan_node=sku"という意味合い
            def link_planning_nodes_to_gui_sku(product_tree_root, gui_node_dict, product_name):
                """
                product_tree_root: 計算用Node（product別）
                gui_node_dict: GUI上の全ノード（node.name -> Nodeインスタンス）
                product_name: 対象製品名（'JPN_Koshihikari'など）
                SKUオブジェクトに計算ノード（Node）のポインタを渡す
                """
                def traverse_and_link(plan_node):
                    gui_node = gui_node_dict.get(plan_node.name)
                    if gui_node is not None:
                        #@250728 STOP
                        #sku = gui_node.sku_dict.get(product_name)
                        #if sku:
                        #    #計算ノードへのリンク
                        #    sku.psi_node_ref = plan_node
                        #@250728 GO
                        gui_node.sku_dict[product_name] = plan_node # Plan2GUI direct setting
                    for child in plan_node.children:
                        traverse_and_link(child)
                traverse_and_link(product_tree_root)
            for prod_nm in product_name_list:
                link_planning_nodes_to_gui_sku(prod_tree_dict_OT[prod_nm], nodes_outbound, prod_nm)
                link_planning_nodes_to_gui_sku(prod_tree_dict_IN[prod_nm], nodes_inbound, prod_nm)
            # save to self.xxx
            self.prod_tree_dict_OT = prod_tree_dict_OT
            self.prod_tree_dict_IN = prod_tree_dict_IN
            # 検証表示
            for prod_nm in product_name_list:
                print("検証表示product_nm 4 subtree", prod_nm )
                if prod_tree_root_IN:
                    print("Inbound prod_tree:")
                    prod_tree_dict_IN[prod_nm].print_tree()
                if prod_tree_root_OT:
                    print("Outbound prod_tree:")
                    prod_tree_dict_OT[prod_nm].print_tree()
            # **************************************
            # end of GUI_node, PSI_node and sku data building
            # **************************************
            # **************************************
            # setting cost parameters
            # **************************************
            #@250719 ADD
            def load_cost_param_csv(filepath):
                import csv
                param_dict = {}
                with open(filepath, newline='', encoding="utf-8-sig") as f:
                #with open(filepath, newline='', encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    print("CSV columns:", reader.fieldnames)  # デバッグ用
                    #reader = csv.DictReader(f, delimiter="\t") # タブ区切りに
                    #print("CSV columns:", reader.fieldnames)   # 確認用に追加
                    #reader = csv.DictReader(f)
                    for row in reader:
                        product = row["product_name"]
                        node = row["node_name"]
                        if product not in param_dict:
                            param_dict[product] = {}
                        param_dict[product][node] = {
                            "price": float(row.get("price_sales_shipped", 0)),
                            "cost_total": float(row.get("cost_total", 0)),
                            "profit_margin": float(row.get("profit", 0)),  # optional
                            "marketing": float(row.get("marketing_promotion", 0)),
                            "sales_admin_cost": float(row.get("sales_admin_cost", 0)),
                            "SGA_total": float(row.get("SGA_total", 0)),
                            "transport_cost": float(row.get("logistics_costs", 0)),
                            "storage_cost": float(row.get("warehouse_cost", 0)),
                            "purchase_price": float(row.get("direct_materials_costs", 0)),
                            "tariff_cost": float(row.get("tariff_cost", 0)),
                            "purchase_total_cost": float(row.get("purchase_total_cost", 0)),
                            "direct_labor_cost": float(row.get("direct_labor_costs", 0)),
                            "fixed_cost": float(row.get("manufacturing_overhead", 0)),
                            # optional: detail for GUI use
                            "prod_indirect_labor": float(row.get("prod_indirect_labor", 0)),
                            "prod_indirect_cost": float(row.get("prod_indirect_others", 0)),
                            "depreciation_cost": float(row.get("depreciation_others", 0)),
                            # ... 他の詳細項目も追加可
                        }
                return param_dict
            if "sku_cost_table_outbound.csv" in data_file_list:
                cost_param_OT_dict = load_cost_param_csv(os.path.join(directory, "sku_cost_table_outbound.csv"))
                print("cost_param_OT_dict", cost_param_OT_dict)
                #@STOP
                #load_sku_cost_master(os.path.join(directory, "sku_cost_table_outbound.csv"), self.nodes_outbound)
                #@STOP
                #read_set_cost(os.path.join(directory, "node_cost_table_outbound.csv"), self.nodes_outbound)
            else:
                print("error: sku_cost_table_outbound.csv is missed")
            if "sku_cost_table_inbound.csv" in data_file_list:
                cost_param_IN_dict = load_cost_param_csv(os.path.join(directory, "sku_cost_table_inbound.csv"))
                #@STOP
                #load_sku_cost_master(os.path.join(directory, "sku_cost_table_inbound.csv"), self.nodes_inbound)
                #@STOP
                #read_set_cost(os.path.join(directory, "node_cost_table_inbound.csv"), self.nodes_inbound)
            else:
                print("error: sku_cost_table_inbound.csv is missed")
            ## Cost structure demand
            #self.price_sales_shipped = 0
            #self.cost_total = 0
            #self.profit = 0
            #self.marketing_promotion = 0
            #self.sales_admin_cost = 0
            #self.SGA_total = 0
            #self.custom_tax = 0
            #self.tax_portion = 0
            #self.logistics_costs = 0
            #self.warehouse_cost = 0
            #self.direct_materials_costs = 0
            #self.purchase_total_cost = 0
            #self.prod_indirect_labor = 0
            #self.prod_indirect_others = 0
            #self.direct_labor_costs = 0
            #self.depreciation_others = 0
            #self.manufacturing_overhead = 0
            # this is "product_tree" operation / that is "PlanNode"
            def cost_param_setter(product_tree_root, param_dict, product_name):
                def traverse(node):
                    node_name = node.name
                    if product_name in param_dict and node_name in param_dict[product_name]:
                        param_set = param_dict[product_name][node_name]
                        #@250801 memo node is an instance of "PlanNode"
                        sku = node.sku
                        sku.price               = param_set.get("price", 0)
                        sku.cost_total          = param_set.get("cost_total", 0)
                        sku.profit_margin       = param_set.get("profit_margin", 0)
                        sku.marketing           = param_set.get("marketing", 0)
                        sku.sales_admin_cost    = param_set.get("sales_admin_cost", 0)
                        sku.SGA_total           = param_set.get("SGA_total", 0)
                        sku.transport_cost      = param_set.get("transport_cost", 0)
                        sku.storage_cost        = param_set.get("storage_cost", 0)
                        sku.purchase_price      = param_set.get("purchase_price", 0)
                        sku.tariff_cost         = param_set.get("tariff_cost", 0)
                        sku.purchase_total_cost = param_set.get("purchase_total_cost", 0)
                        sku.direct_labor_costs  = param_set.get("direct_labor_costs", 0)
                        sku.fixed_cost          = param_set.get("fixed_cost", 0)
                        sku.prod_indirect_labor = param_set.get("prod_indirect_labor", 0)
                        sku.prod_indirect_cost  = param_set.get("prod_indirect_cost", 0)
                        sku.depreciation_cost   = param_set.get("depreciation_cost", 0)
                        #sku.price               = param_set.get("sku.price", 0)
                        #sku.cost_total          = param_set.get("sku.cost_total", 0)
                        #sku.profit_margin       = param_set.get("sku.profit_margin", 0)
                        #sku.marketing           = param_set.get("sku.marketing", 0)
                        #sku.sales_admin_cost    = param_set.get("sku.sales_admin_cost", 0)
                        #sku.SGA_total           = param_set.get("sku.SGA_total", 0)
                        #sku.transport_cost      = param_set.get("sku.transport_cost", 0)
                        #sku.storage_cost        = param_set.get("sku.storage_cost", 0)
                        #sku.purchase_price      = param_set.get("sku.purchase_price", 0)
                        #sku.tariff_cost         = param_set.get("tariff_cost", 0)
                        #sku.purchase_total_cost = param_set.get("sku.purchase_total_cost", 0)
                        #sku.direct_labor_costs  = param_set.get("sku.direct_labor_costs", 0)
                        #sku.fixed_cost          = param_set.get("sku.fixed_cost", 0)
                        #sku.prod_indirect_labor = param_set.get("sku.prod_indirect_labor", 0)
                        #sku.prod_indirect_cost  = param_set.get("sku.prod_indirect_cost", 0)
                        #sku.depreciation_cost   = param_set.get("sku.depreciation_cost", 0)
                        #sku.price = param_set.get("price", 0)
                        #sku.transport_cost = param_set.get("transport_cost", 0)
                        #sku.storage_cost = param_set.get("storage_cost", 0)
                        #sku.purchase_price = param_set.get("purchase_price", 0)
                        #sku.fixed_cost = param_set.get("fixed_cost", 0)
                        #sku.other_cost = param_set.get("other_cost", 0)
                        #sku.total_cost = (
                        #    sku.purchase_price + sku.transport_cost + sku.storage_cost +
                        #    sku.tariff_cost + sku.fixed_cost + sku.other_cost
                        #)
                        # ✅ PlanNode 側へコピー
                        node.cs_price_sales_shipped    = sku.price
                        node.cs_cost_total             = sku.cost_total
                        node.cs_profit                 = sku.profit_margin
                        node.cs_marketing_promotion    = sku.marketing
                        node.cs_sales_admin_cost       = sku.sales_admin_cost
                        node.cs_SGA_total              = sku.SGA_total
                        node.cs_logistics_costs        = sku.transport_cost
                        node.cs_warehouse_cost         = sku.storage_cost
                        node.cs_direct_materials_costs = sku.purchase_price
                        node.cs_tax_portion            = sku.tariff_cost
                        node.cs_purchase_total_cost    = sku.purchase_total_cost
                        node.cs_direct_labor_costs     = sku.direct_labor_costs
                        node.cs_manufacturing_overhead = sku.fixed_cost
                        node.cs_prod_indirect_labor    = sku.prod_indirect_labor
                        node.cs_prod_indirect_others   = sku.prod_indirect_cost
                        node.cs_depreciation_others    = sku.depreciation_cost
                        #node.eval_cs_price_sales_shipped = sku.price
                        #node.eval_cs_profit = sku.price - sku.total_cost
                        #node.eval_cs_SGA_total = param_set.get("SGA_total", 0)
                        #node.eval_cs_tax_portion = sku.tariff_cost
                        #node.eval_cs_logistics_costs = sku.transport_cost
                        #node.eval_cs_warehouse_cost = sku.storage_cost
                        #node.eval_cs_direct_materials_costs = sku.purchase_price
                    for child in node.children:
                        traverse(child)
                traverse(product_tree_root)
            # 読み込んだ辞書を全製品ツリーに適用
            for product_name in list(prod_tree_dict_OT.keys()):
                #@250729 ADD
                print("cost_param_OT_dict", cost_param_OT_dict)
                cost_param_setter(prod_tree_dict_OT[product_name], cost_param_OT_dict, product_name)
                #cost_param_setter(subtree_OT_dict[product_name], cost_param_OT_dict, product_name)
            for product_name in list(prod_tree_dict_IN.keys()):
                cost_param_setter(prod_tree_dict_IN[product_name], cost_param_IN_dict, product_name)
                #cost_param_setter(subtree_IN_dict[product_name], cost_param_IN_dict, product_name)
                #cost_param_setter(product_tree_dict[product_name], param_dict, product_name)
            #@250719 ADD from import
            # *****************************
            # cost propagation
            # *****************************
            # 0.setting price table
            selling_price_table_csv = os.path.join(directory, "selling_price_table.csv")
            tobe_price_dict = load_tobe_prices(selling_price_table_csv)
            assign_tobe_prices_to_leaf_nodes(prod_tree_dict_OT, tobe_price_dict)
            shipping_price_table_csv = os.path.join(directory, "shipping_price_table.csv")
            asis_price_dict = load_asis_prices(shipping_price_table_csv)
            assign_asis_prices_to_root_nodes(prod_tree_dict_OT, asis_price_dict)
            print("offering_price check: self.nodes_outbound[ CS_JPN ].sku_dict[ JPN_Koshihikari ].offering_price_TOBE", self.nodes_outbound["CS_JPN"].sku_dict["JPN_Koshihikari"].offering_price_TOBE)
            print("offering_price check: self.nodes_outbound[ DADJPN ].sku_dict[ JPN_RICE_1 ].offering_price_TOBE", self.nodes_outbound["DADJPN"].sku_dict["JPN_RICE_1"].offering_price_TOBE)
            # 1.initial propagation 実行
            print("cost propagation processing")
            gui_run_initial_propagation(prod_tree_dict_OT, directory)
            #@250807 STOP
            #gui_run_initial_propagation(prod_tree_dict_IN, directory)
            print("offering_price check: self.nodes_outbound[ CS_JPN ].sku_dict[ JPN_Koshihikari ].offering_price_TOBE", self.nodes_outbound["CS_JPN"].sku_dict["JPN_Koshihikari"].offering_price_TOBE)
            print("offering_price check: self.nodes_outbound[ DADJPN ].sku_dict[ JPN_RICE_1 ].offering_price_TOBE", self.nodes_outbound["DADJPN"].sku_dict["JPN_RICE_1"].offering_price_TOBE)
            # 2.PlanNodeへの評価値のコピー
            print("propagate_cost_to_plan_nodes start...")
            #self.print_cost_sku()
            #self.print_cost_node_cs()
            #self.print_cost_node_eval_cs()
            propagate_cost_to_plan_nodes(prod_tree_dict_OT)
            propagate_cost_to_plan_nodes(prod_tree_dict_IN)
            print("propagate_cost_to_plan_nodes end...")
            #self.print_cost_sku()
            #self.print_cost_node_cs()
            #self.print_cost_node_eval_cs()
#@250720 STOP
#            #@250720 ADD この後のloading processがココで止まる
#            self.view_nx_matlib4opt()
#
#
#    #@250720 ADD loading processの続きを仮設で定義
#    def load_data_files_CONTONUE(self):
            # **************************************
            # setting S_month 2 psi4demand
            # **************************************
            if "S_month_data.csv" in data_file_list:
                in_file_path = os.path.join(directory, "S_month_data.csv")
                df_weekly, plan_range, plan_year_st = process_monthly_demand(in_file_path, self.lot_size)
                self.plan_year_st = plan_year_st
                self.plan_range = plan_range
                self.plan_year_entry.delete(0, tk.END)
                self.plan_year_entry.insert(0, str(self.plan_year_st))
                self.plan_range_entry.delete(0, tk.END)
                self.plan_range_entry.insert(0, str(self.plan_range))
                df_weekly.to_csv(os.path.join(directory, "S_iso_week_data.csv"), index=False)
            else:
                print("error: S_month_data.csv is missed")
            # ****************************************
            # Original Node base demand setting
            # ****************************************
            root_node_outbound.set_plan_range_lot_counts(plan_range, plan_year_st)
            root_node_inbound.set_plan_range_lot_counts(plan_range, plan_year_st)
            node_psi_dict_Ot4Dm = make_psi_space_dict(root_node_outbound, {}, plan_range)
            node_psi_dict_Ot4Sp = make_psi_space_dict(root_node_outbound, {}, plan_range)
            self.node_psi_dict_In4Dm = make_psi_space_dict(root_node_inbound, {}, plan_range)
            self.node_psi_dict_In4Sp = make_psi_space_dict(root_node_inbound, {}, plan_range)
            set_dict2tree_psi(root_node_outbound, "psi4demand", node_psi_dict_Ot4Dm)
            set_dict2tree_psi(root_node_outbound, "psi4supply", node_psi_dict_Ot4Sp)
            set_dict2tree_psi(root_node_inbound, "psi4demand", self.node_psi_dict_In4Dm)
            set_dict2tree_psi(root_node_inbound, "psi4supply", self.node_psi_dict_In4Sp)
            # **********************************
            # make&set weekly demand "Slots" on leaf_node, propagate2root
            # initial setting psi4"demand"[w][0] to psi4"supply"[w][0]
            # **********************************
            #set_df_Slots2psi4demand(self.root_node_outbound, df_weekly)
            set_df_Slots2psi4demand(root_node_outbound, df_weekly)
            # convert_monthly_to_weekly() → set_df_Slots2psi4demand() の後
            for node in self.nodes_outbound.values():
                print(f"[{node.name}] demand lots per week:",
                      [len(node.psi4demand[w][0]) for w in range(1, self.plan_range + 1)])
                      #[len(node.psi4demand[w][0]) for w in range(1, min(self.plan_range + 1, 10))])
            # ****************************************
            # by Product tree with PlanNode  demand setting
            # ****************************************
            for prod_nm in product_name_list:
                prod_tree_root_OT = prod_tree_dict_OT[prod_nm]
                prod_tree_root_IN = prod_tree_dict_IN[prod_nm]
                prod_tree_root_OT.set_plan_range_lot_counts(plan_range, plan_year_st)
                prod_tree_root_IN.set_plan_range_lot_counts(plan_range, plan_year_st)
                #root_node_outbound.set_plan_range_lot_counts(plan_range, plan_year_st)
                #root_node_inbound.set_plan_range_lot_counts(plan_range, plan_year_st)
                node_psi_dict_Ot4Dm = make_psi_space_dict(prod_tree_root_OT, {}, plan_range)
                node_psi_dict_Ot4Sp = make_psi_space_dict(prod_tree_root_OT, {}, plan_range)
                self.node_psi_dict_In4Dm = make_psi_space_dict(prod_tree_root_IN, {}, plan_range)
                self.node_psi_dict_In4Sp = make_psi_space_dict(prod_tree_root_IN, {}, plan_range)
                #node_psi_dict_Ot4Dm = make_psi_space_dict(root_node_outbound, {}, plan_range)
                #node_psi_dict_Ot4Sp = make_psi_space_dict(root_node_outbound, {}, plan_range)
                #self.node_psi_dict_In4Dm = make_psi_space_dict(root_node_inbound, {}, plan_range)
                #self.node_psi_dict_In4Sp = make_psi_space_dict(root_node_inbound, {}, plan_range)
                set_dict2tree_psi(prod_tree_root_OT, "psi4demand", node_psi_dict_Ot4Dm)
                set_dict2tree_psi(prod_tree_root_OT, "psi4supply", node_psi_dict_Ot4Sp)
                set_dict2tree_psi(prod_tree_root_IN, "psi4demand", self.node_psi_dict_In4Dm)
                set_dict2tree_psi(prod_tree_root_IN, "psi4supply", self.node_psi_dict_In4Sp)
                # **********************************
                # make&set weekly demand "Slots" on leaf_node, propagate2root
                # initial setting psi4"demand"[w][0] to psi4"supply"[w][0]
                # **********************************
                #set_df_Slots2psi4demand(self.root_node_outbound, df_weekly)
                set_df_Slots2psi4demand(prod_tree_root_OT, df_weekly)
                # convert_monthly_to_weekly() → set_df_Slots2psi4demand() の後
                #for node in self.prod_tree_dict_OT[prod_nm].values():
                for node in self.nodes_outbound.values():
                    print(f"[{node.name}] by Product demand lots per week:",
                          [len(node.psi4demand[w][0]) for w in range(1, self.plan_range + 1)])
                          #[len(node.psi4demand[w][0]) for w in range(1, min(self.plan_range + 1, 10))])
                #prod_tree_root_OT = prod_tree_dict_OT[prod_nm]
        # *****************************
        # export offring_price ASIS/TOBE to csv
        # *****************************
        filename = "offering_price_ASIS_TOBE.csv"
        output_csv_path = os.path.join(self.directory, filename)
        self.export_offering_prices(output_csv_path)
        #@STOP can NOT eval before "psi" loading
        ## eval area
        #self.update_evaluation_results()
        # network area
        self.decouple_node_selected = []
        #self.decouple_node_selected = decouple_node_names
        self.view_nx_matlib4opt()
        #self.view_nx_matlib4opt_WO_capa()
        # _init_ self.product_selected = self.product_name_list[0]
        #product_name = self.product_selected
        product_name = self.product_name_list[1]
        # PSI area
        self.root.after(1000, self.show_psi_by_product("outbound", "demand", product_name))
        #@STOP
        ## PSI area
        ##self.root.after(1000, self.show_psi("outbound", "supply"))
        #
        #self.root.after(1000, lambda: self.show_psi("outbound", "supply"))
        ## lambda: にすることで、1000ms 後に初めて show_psi() を実行する
        # ****************************
        # market potential Graph viewing
        # ****************************
        self.initialize_parameters()
        # Enable buttons after loading is complete
        self.supply_planning_button.config(state="normal")
        self.eval_buffer_stock_button.config(state="normal")
        print("Data files loaded and buttons enabled.")
        # Return focus to the main window
        self.root.focus_force()
        # ****************************
        # passing following process
        # ****************************
        pass
# **** A PART of ORIGINAL load_data_files END *****
