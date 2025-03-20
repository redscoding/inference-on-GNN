import numpy as np
import networkx as nx
from pomegranate import *
from data_viz import data_viz
import random
from bn_to_fac import bn_to_fac
from myConstants import ASYMMETRIC
#from data_structs import BinaryMRF
import matplotlib.pyplot as plt
import matplotlib
struct_names = ["star", "random_tree", "powerlaw_tree", "path",
                "cycle", "ladder", "grid",
                "circ_ladder", "barbell", "loll", "wheel",
                "bipart", "tripart", "fc"]

def generate_struct_mask(struct, n_nodes, shuffle_nodes):
    # a horrible collection of ifs due to args in nx constructors
    if struct == "star":
        g = nx.star_graph(n_nodes)
    elif struct == "random_tree":
        g = nx.random_tree(n_nodes)
    elif struct == "powerlaw_tree":
        g = nx.powerlaw_tree(n_nodes, gamma=3, seed=None)
    elif struct == "binary_tree":
        raise NotImplementedError("Implement a binary tree.")
    elif struct == "path":
        g = nx.path_graph(n_nodes)
    elif struct == "cycle":
        g = nx.cycle_graph(n_nodes)
    elif struct == "ladder":
        g = nx.ladder_graph(n_nodes)
    elif struct == "grid":
        n = m = int(np.sqrt(n_nodes))
        assert n*m == n_nodes
        # m = np.random.choice(range(1, n_nodes+1))
        # n = n_nodes // m
        g = nx.generators.lattice.grid_2d_graph(m, n)
    elif struct == "circ_ladder":
        g = nx.circular_ladder_graph(n_nodes)
    elif struct == "barbell":
        assert n_nodes >= 4
        m = np.random.choice(range(2, n_nodes-1))
        blocks = (m, n_nodes-m)
        g = nx.barbell_graph(*blocks)
    elif struct == "loll":
        assert n_nodes >= 2
        m = np.random.choice(range(2, n_nodes+1))
        g = nx.lollipop_graph(m, n_nodes-m)
    elif struct == "wheel":
        g = nx.wheel_graph(n_nodes)
    elif struct == "bipart":
        m = np.random.choice(range(n_nodes))
        blocks = (m, n_nodes-m)
        g = nx.complete_multipartite_graph(*blocks)
    elif struct == "tripart":
        # allowed to be zero
        m, M = np.random.choice(range(n_nodes), size=2)
        if m > M:
            m, M = M, m
        blocks = (m, M-m, n_nodes-M)
        g = nx.complete_multipartite_graph(*blocks)
    elif struct == "fc":
        g = nx.complete_graph(n_nodes)
    else:
        raise NotImplementedError("Structure {} not implemented yet.".format(struct))

    # fix bugs, relabel nodes to make sure nodes are indexed by integers!
    mapping = {n:idx for idx,n in enumerate(g.nodes())}
    g = nx.relabel_nodes(g, mapping)

    node_order = list(range(n_nodes))
    if shuffle_nodes:
        np.random.shuffle(node_order)

    # a weird subclass by default; raises a deprecation warning
    # with a new update of networkx, this should be updated to
    # nx.convert_matrix.to_numpy_array
    np_arr_g = nx.to_numpy_array(g, nodelist=node_order)
    return g,np_arr_g.astype(int)  #return graph obj and array

def construct_bayesian_network(struct, n_nodes, shuffle_nodes=True):
    """

    Arguments:
        struct {string} -- structure of the graph
        (on of "path", "ladder", ...)
        n_nodes {int} -- number of nodes in the graph
        shuffle_nodes {bool} -- whether to permute node labelings
                                uniformly at random
    Returns:
        BinaryMRF object
    """
    G, G_array =generate_struct_mask(struct, n_nodes, shuffle_nodes)#generate graph from networkx

    # W = np.random.normal(0., 1., (n_nodes, n_nodes))
    # W = (W+W.T)/2 #make adj asymmetric
    # b = np.random.normal(0,0.25,n_nodes)
    # W *=G_array
    # make graph direct
    DG = nx.DiGraph()
    for u, v in G.edges():
        if random.choice([True, False]):  # 50% 機率保留 (u, v)
            DG.add_edge(u, v)
        else:  # 50% 機率反轉 (v, u)
            DG.add_edge(v, u)

    print("show data-viz")
    data_viz(DG,G_array)
    #0319修改 確保成為有向無環圖
    while not nx.is_directed_acyclic_graph(DG):
        cycle = nx.find_cycle(DG, orientation="original")
        u,v,_ = random.choice(cycle)
        print(f"修正: 反轉邊 {u} -> {v} 變成 {v} -> {u}")
        DG.remove_edge(u,v)
        DG.add_edge(v,u)
    print("最終修正完成 線在是DAG")

    node_order = list(range(n_nodes))
    DG_array = nx.to_numpy_array(DG, nodelist=node_order)
    data_viz(DG,DG_array)
    # 確保遍歷順序
    topo_order = list(nx.topological_sort(DG))

    nodes={}
    distributions={}

    for node in topo_order:
        parents = list(DG.predecessors(node))
        node_name = f'X{node}'
        print(parents)
        #確保所有父節點的分布都生成了
        for parent in parents:
            parent_name = f'X{parent}'
            if parent_name not in distributions:
                # 如果父節點分佈尚未生成，立即生成
                prob_1 = np.random.uniform(0.1, 0.5)
                distributions[parent_name] = DiscreteDistribution({'1': prob_1, '0': 1 - prob_1})

        if not parents:  # 無父節點
            # 類似你的範例 a 和 c
            prob_1 = np.random.uniform(0.1, 0.5)  # 隨機生成概率
            distributions[node_name] = DiscreteDistribution({'1': prob_1, '0': 1 - prob_1})
        else:
            # 有父節點，生成條件概率表
            parent_names = [f'X{p}' for p in parents]
            parent_distributions = [distributions[p] for p in parent_names]

            # 根據父節點數量生成概率表
            num_parents = len(parents)
            states = ['0', '1']
            all_combinations = list(np.array(np.meshgrid(*([states] * num_parents))).T.reshape(-1, num_parents))

            table = []
            for combo in all_combinations:
                for state in states:
                    # 隨機生成概率，類似你的範例
                    prob_1 = np.random.uniform(0.1, 0.9)  # 隨機概率
                    table.append(list(combo) + [state, prob_1])
                    table.append(list(combo) + ['1' if state == '0' else '0', 1 - prob_1])

            distributions[node_name] = ConditionalProbabilityTable(table, parent_distributions)

    # 印出每個節點的機率分布 (CPT)
    for node_name, distribution in distributions.items():
        print(f"\nNode: {node_name}")
        print(distribution)  # 直接印出離散分布或 CPT

    bn_model=BayesianNetwork()
    node_obj={}

    for node_name, distribution in distributions.items():
        node = Node(distribution, name=node_name)
        node_obj[node_name] = node
        bn_model.add_node(node)

    for node in DG.nodes():
        node_name = f'X{node}'
        for parent in DG.predecessors(node):
            parent_name = f'X{parent}'
            bn_model.add_edge(node_obj[parent_name],node_obj[node_name])

    bn_model.bake()
    return bn_to_fac(bn_model, DG)




if __name__ == "__main__":
    construct_bayesian_network('bipart',9,True)
    """
    G,G_array=generate_struct_mask('loll', 9, False)
    nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=16)
    plt.show()

    plt.imshow(G_array, cmap='Blues', interpolation='none')
    plt.colorbar(label='Edge Weight')
    plt.title('Adjacency Matrix of Star Graph')
    plt.show()
"""

