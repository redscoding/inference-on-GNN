# BeliefPropagation.py
import numpy as np
import networkx as nx
from bn_to_fac import bn_to_fac
from pomegranate import BayesianNetwork,Node
from pomegranate.distributions import DiscreteDistribution, ConditionalProbabilityTable
from bn_to_fac import FactorGraphObject
from itertools import product


class BeliefPropagation:
    def __init__(self, fg: FactorGraphObject):
        self.fg = fg
        self.messages = {}
        self.prev_messages = {}  # 用於檢查收斂性
        self.sorted_nodes = self._preprocess_nodes()  # 預處理：排序節點

    def _preprocess_nodes(self):
        """按邊數排序節點（變數和因子節點）"""
        node_degrees = {}
        # 計算每個節點的度數
        for node in self.fg.variables:
            node_degrees[node] = len(self.get_neighbors(node))
        for factor in self.fg.factors:
            node_degrees[factor] = len(self.get_neighbors(factor))
        # 按度數從小到大排序
        sorted_nodes = sorted(node_degrees.keys(), key=lambda x: node_degrees[x])
        return sorted_nodes

    def get_neighbors(self, node):
        if node in self.fg.factors:
            return [var for (f, var) in self.fg.edges if f == node]
        else:
            return [f for (f, var) in self.fg.edges if var == node]

    def initialize_messages(self):
        """初始化消息為 1"""
        for (factor, var) in self.fg.edges:
            num_states = len(self.fg.states[var])
            self.messages[(factor, var)] = np.ones(num_states)
            self.messages[(var, factor)] = np.ones(num_states)
            self.prev_messages[(factor, var)] = np.ones(num_states)
            self.prev_messages[(var, factor)] = np.ones(num_states)

    def normalize_message(self, msg):
        """正規化消息，避免過大或過小"""
        total = np.sum(msg)
        if total > 1e10 or total < 1e-10:  # 檢查過大或過小
            return msg / total if total != 0 else msg
        return msg

    def msg_vTf(self, var, factor):
        incoming = [self.messages[(f, var)] for f in self.get_neighbors(var) if f != factor]
        if not incoming:
            return self.normalize_message(np.ones(len(self.fg.states[var])))
        return self.normalize_message(np.prod(incoming, axis=0))

    def msg_fTv(self, factor, var):
        neighbors = self.get_neighbors(factor)
        other_vars = [v for v in neighbors if v != var]
        cpt = self.fg.factors[factor]
        msg = np.zeros(len(self.fg.states[var]))

        if isinstance(cpt, DiscreteDistribution):
            probs = cpt.parameters[0]
            if len(neighbors) > 1:
                raise ValueError(f"因子 {factor} 是 DiscreteDistribution，但有多於一個鄰居: {neighbors}")
            for val_idx, val in enumerate(self.fg.states[var]):
                msg[val_idx] = probs[str(val)]
        elif isinstance(cpt, ConditionalProbabilityTable):
            table = cpt.parameters[0]
            var_index = neighbors.index(var)
            other_indices = [neighbors.index(v) for v in other_vars]

            for val_idx, val in enumerate(self.fg.states[var]):
                total = 0
                other_states = [self.fg.states[v] for v in other_vars]
                for other_vals in product(*other_states):
                    assignment = [None] * len(neighbors)
                    for i, v in enumerate(other_vars):
                        assignment[other_indices[i]] = other_vals[i]
                    assignment[var_index] = val
                    for row in table:
                        if all(row[i] == assignment[i] for i in range(len(neighbors))):
                            prob = row[-1]
                            incoming_product = 1.0
                            for i, v in enumerate(other_vars):
                                incoming = self.messages[(v, factor)]
                                other_val_idx = self.fg.states[v].index(other_vals[i])
                                incoming_product *= incoming[other_val_idx]
                            total += prob * incoming_product
                            break
                msg[val_idx] = total
        return self.normalize_message(msg)

    def run(self, max_iter=10, convergence_threshold=1e-5):
        self.initialize_messages()
        for iteration in range(max_iter):
            converged = True
            # 按排序順序更新消息
            for node in self.sorted_nodes:
                if node in self.fg.variables:
                    # 變數節點更新到因子節點的消息
                    for factor in self.get_neighbors(node):
                        new_msg = self.msg_vTf(node, factor)
                        if not np.allclose(new_msg, self.prev_messages[(node, factor)], atol=convergence_threshold):
                            converged = False
                        self.prev_messages[(node, factor)] = self.messages[(node, factor)].copy()
                        self.messages[(node, factor)] = new_msg
                elif node in self.fg.factors:
                    # 因子節點更新到變數節點的消息
                    for var in self.get_neighbors(node):
                        new_msg = self.msg_fTv(node, var)
                        if not np.allclose(new_msg, self.prev_messages[(node, var)], atol=convergence_threshold):
                            converged = False
                        self.prev_messages[(node, var)] = self.messages[(node, var)].copy()
                        self.messages[(node, var)] = new_msg
            if converged:
                print(f"收斂於迭代 {iteration + 1}")
                break
        else:
            print(f"達到最大迭代次數 {max_iter}，未完全收斂")

    def get_marginal(self, var):
        incoming = [self.messages[(f, var)] for f in self.get_neighbors(var)]
        belief = np.prod(incoming, axis=0)
        return belief / np.sum(belief)

def run_belief_propagation(fg: FactorGraphObject, max_iter=10, convergence_threshold=1e-5):
    bp = BeliefPropagation(fg)
    bp.run(max_iter=max_iter, convergence_threshold=convergence_threshold)
    for var in fg.variables:
        marginal = bp.get_marginal(var)
        print(f"{var} 的邊際概率: {marginal}")

if __name__ == "__main__":
    model = BayesianNetwork()
    dist_x0 = DiscreteDistribution({'0': 0.7, '1': 0.3})
    dist_x2 = DiscreteDistribution({'0': 0.8, '1': 0.2})  # 這裡可能是問題來源，應該是 ConditionalProbabilityTable
    dist_x3 = ConditionalProbabilityTable(
        [['0', '0', 0.9], ['0', '1', 0.1], ['1', '0', 0.5], ['1', '1', 0.5]], [dist_x0]
    )
    dist_x7 = ConditionalProbabilityTable(
        [['0', '0', 0.6], ['0', '1', 0.4], ['1', '0', 0.3], ['1', '1', 0.7]], [dist_x0]
    )
    dist_x8 = ConditionalProbabilityTable(
        [['0', '0', 0.7], ['0', '1', 0.3], ['1', '0', 0.2], ['1', '1', 0.8]], [dist_x0]
    )
    dist_x9 = ConditionalProbabilityTable(
        [['0', '0', 0.5], ['0', '1', 0.5], ['1', '0', 0.6], ['1', '1', 0.4]], [dist_x0]
    )

    node_x0 = Node(dist_x0, name="X0")
    node_x2 = Node(dist_x2, name="X2")
    node_x3 = Node(dist_x3, name="X3")
    node_x7 = Node(dist_x7, name="X7")
    node_x8 = Node(dist_x8, name="X8")
    node_x9 = Node(dist_x9, name="X9")

    model.add_nodes(node_x0, node_x2, node_x3, node_x7, node_x8, node_x9)
    model.add_edge(node_x0, node_x2)
    model.add_edge(node_x0, node_x3)
    model.add_edge(node_x0, node_x7)
    model.add_edge(node_x0, node_x8)
    model.add_edge(node_x0, node_x9)
    model.bake()

    DG = nx.DiGraph([(0, 2), (0, 3), (0, 7), (0, 8), (0, 9)])

    # 檢查 bn_model 的結構
    print("檢查 bn_model 的節點分佈:")
    for i, state in enumerate(model.states):
        print(f"節點 {state.name}: 分佈類型 = {type(state.distribution)}, 內容 = {state.distribution}")

    graph = bn_to_fac(model, DG)
    algo = []
    algo.append(run_belief_propagation(graph, max_iter=10, convergence_threshold=1e-5))
