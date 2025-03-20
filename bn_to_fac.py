import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pomegranate import DiscreteDistribution, ConditionalProbabilityTable, BayesianNetwork, Node

def bn_to_fac(bn_model, DG):
    FG = nx.Graph()
    variable_nodes = [f'X{node}' for node in DG.nodes()]
    factor_nodes = []
    factor_to_cpd = {}

    # 加入變數節點
    for var in variable_nodes:
        FG.add_node(var, type='variable')

    # 加入因子節點
    for node in DG.nodes():
        parents = list(DG.predecessors(node))
        involved_vars = [node] + parents  # 這個因子涉及的變數
        factor_name = f'F{node}'  # 用 F 來代表因子節點
        factor_nodes.append(factor_name)
        FG.add_node(factor_name, type='factor')

        # 連接該因子與所有涉及的變數
        for var in involved_vars:
            FG.add_edge(factor_name, f'X{var}')

        # 儲存因子的 CPT
        factor_to_cpd[factor_name] = bn_model.states[node].distribution

    draw(FG)

    return FG

#transform_data(FG,variable_nodes,factor_nodes, factor_to_cpd)

def draw(FG):
    pos = nx.spring_layout(FG)  # 佈局算法
    node_colors = [
        "lightblue" if FG.nodes[n]["type"] == "variable" else "lightcoral"
        for n in FG.nodes()
    ]

    plt.figure(figsize=(8, 6))
    nx.draw(
        FG, pos, with_labels=True, node_color=node_colors, edge_color="gray",
        node_size=150, font_size=10
    )
    plt.title("Factor Graph")
    plt.show()