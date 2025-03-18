import numpy as np
import networkx as nx
from networkx import edges
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from data_viz import data_viz
import random
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
    """Construct one binary MRF graphical model

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

    W = np.random.normal(0., 1., (n_nodes, n_nodes))
    W = (W+W.T)/2 #make adj asymmetric
    b = np.random.normal(0,0.25,n_nodes)
    W *=G_array
    # make graph direct
    DG = nx.DiGraph()
    for u, v in G.edges():
        if random.choice([True, False]):  # 50% 機率保留 (u, v)
            DG.add_edge(u, v)
        else:  # 50% 機率反轉 (v, u)
            DG.add_edge(v, u)
    print("show data-viz")
    data_viz(DG,G_array)

    #build bayesian network
    edges = [(f'X{u}', f'X{v}') for u, v in DG.edges()]
    bn_model = BayesianNetwork(edges)
    def compute_cpd(node, parents, DG, W, b):
        bias = b[node]
        values = []
        parent_states = [[-1,+1]] * len(parents) if parents else [[]] #0318 change
        for state_combo in np.array(np.meshgrid(*parent_states)).T.reshape(-1, len(parents)):
            row = []
            for state in [-1, +1]:
                energy = bias * state
                for i, parent in enumerate(parents):
                    weight = W[parent][node]  # 注意：W 需對應方向
                    energy += weight * state * state_combo[i]
                prob = 1 / (1 + np.exp(-energy))  # sigmoid
                row.append(prob)
            values.append(row)
        return TabularCPD(variable=f'X{node}', variable_card=2,
                          values=values, evidence=[f'X{p}' for p in parents],
                          evidence_card=[2] * len(parents))
    cpds=[]
    for node in DG.nodes():
        parents = list(DG.predecessors(node))
        if not parents:
            cpds.append(TabularCPD(variable=f'X{node}',variable_card=2, values=[[0.5],[0.5]]))
        else:
            cpds.append(compute_cpd(node, parents, DG, W, b))

    bn_model.add_cpds(*cpds)
    assert bn_model.check_model(), "Bayesian Network is invalid!"
    return model





if __name__ == "__main__":
    construct_bayesian_network('star',9,True)
    """
    G,G_array=generate_struct_mask('loll', 9, False)
    nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=16)
    plt.show()

    plt.imshow(G_array, cmap='Blues', interpolation='none')
    plt.colorbar(label='Edge Weight')
    plt.title('Adjacency Matrix of Star Graph')
    plt.show()
"""

