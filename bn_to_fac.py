import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pomegranate import DiscreteDistribution, ConditionalProbabilityTable, BayesianNetwork, Node

class FactorGraphObject:
    def __init__(self):
        self.variables = set()  # 變數節點
        self.factors = {}  # 存儲因子對應的CPT
        self.edges = []  # 存儲邊信息

    def add_variable(self, var_name):
        self.variables.add(var_name)

    def add_factor(self, factor_name, cpt):
        self.factors[factor_name] = cpt

    def add_edge(self, factor, var):
        self.edges.append((factor, var))

    def show(self):
        print("Variables:", self.variables)
        print("Factors:", list(self.factors.keys()))
        print("Edges:", self.edges)

def bn_to_fac(bn_model, DG):
    FG = FactorGraphObject()

    # 加入變數節點
    for node in DG.nodes():
        var_name = f'X{node}'
        FG.add_variable(var_name)

    # 加入因子節點
    for node in DG.nodes():
        parents = list(DG.predecessors(node))
        involved_vars = [node] + parents  # 這個因子涉及的變數
        factor_name = f'F{node}'  # 用 F 來代表因子節點

        # 連接該因子與所有涉及的變數
        for var in involved_vars:
            FG.add_edge(factor_name, f'X{var}')

        # 儲存因子的 CPT
        FG.add_factor(factor_name, bn_model.states[node].distribution)

    return FG  # 返回自定義的 FactorGraphObject
