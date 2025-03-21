# bn_to_fac.py
from pomegranate.distributions import DiscreteDistribution, ConditionalProbabilityTable
import networkx as nx

class FactorGraphObject:
    def __init__(self):
        self.variables = set()
        self.factors = {}
        self.edges = []
        self.states = {}

    def add_variable(self, var_name):
        self.variables.add(var_name)

    def add_factor(self, factor_name, cpt):
        self.factors[factor_name] = cpt

    def add_edge(self, factor, var):
        self.edges.append((factor, var))

    def set_states(self, states):
        self.states.update(states)

    def show(self):
        print("變數節點:", self.variables)
        print("因子節點:", list(self.factors.keys()))
        print("邊信息:", self.edges)
        print("因子CPT:", [self.factors[f] for f in self.factors])
        print("變數狀態:", self.states)

def bn_to_fac(bn_model, DG):
    FG = FactorGraphObject()

    # 建立節點名稱到索引的映射
    node_to_idx = {state.name: i for i, state in enumerate(bn_model.states)}

    # 加入變數節點
    for node in DG.nodes():
        var_name = f'X{node}'
        if var_name not in node_to_idx:
            raise ValueError(f"節點 {var_name} 在 bn_model 中不存在")
        FG.add_variable(var_name)

    # 加入因子節點和邊
    for node in DG.nodes():
        parents = list(DG.predecessors(node))
        factor_name = f'F{node}'
        var_name = f'X{node}'
        dist = bn_model.states[node_to_idx[var_name]].distribution

        # 一致性檢查
        if isinstance(dist, DiscreteDistribution) and parents:
            raise ValueError(f"節點 {node} 有父節點 {parents}，但分佈是 DiscreteDistribution")
        elif isinstance(dist, ConditionalProbabilityTable) and not parents:
            raise ValueError(f"節點 {node} 無父節點，但分佈是 ConditionalProbabilityTable")

        if isinstance(dist, DiscreteDistribution):
            FG.add_edge(factor_name, f'X{node}')
        elif isinstance(dist, ConditionalProbabilityTable):
            involved_vars = [node] + parents
            for var in involved_vars:
                FG.add_edge(factor_name, f'X{var}')
        else:
            raise ValueError(f"未知的分佈類型: {type(dist)}")
        FG.add_factor(factor_name, dist)

    # 提取變數狀態
    states = {}
    for node in DG.nodes():
        var_name = f'X{node}'
        dist = bn_model.states[node_to_idx[var_name]].distribution
        if isinstance(dist, DiscreteDistribution):
            states[var_name] = list(dist.parameters[0].keys())
        elif isinstance(dist, ConditionalProbabilityTable):
            table = dist.parameters[0]
            states[var_name] = sorted(set(row[-2] for row in table))
            parents = list(DG.predecessors(node))
            for i, parent in enumerate(parents):
                parent_var = f'X{parent}'
                if parent_var not in states:
                    states[parent_var] = sorted(set(row[i] for row in table))
    FG.set_states(states)

    return FG