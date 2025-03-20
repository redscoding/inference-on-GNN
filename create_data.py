from hashlib import algorithms_available

import numpy as np
from pgmpy.inference import BeliefPropagation
from tqdm import tqdm
from data_gen import construct_bayesian_network
#from BeliefPropagation import BeliefPropagation
low=9 #range of sizes, int the form 10_20
high=9
num=3 #number of graph to generate
graph_struct='star' #type of graph structure, such as star of fc
# algorithm to use for labeling, can be exact\bp\mcmc
#label_prop for label propagation, or label_sg for subgraph labeling
algo = 'exact'
mode = 'marginal' # type of inference to perform
verbose = True #whethter to display dataset statistics
size_range=np.arange(low,high+1)

#create new graph
graphs=[]
algo=[]
for _ in range(num):
    #sample n_nodes from range
    n_nodes = np.random.choice(size_range)
    graphs.append(construct_bayesian_network(graph_struct, n_nodes))
    print("-------------------showing graph----------------")
    print(graphs)

#chosing inference algorithm
#defult using bp
graph_iterators = tqdm(graphs)
for graph in graph_iterators:
    algo.append(BeliefPropagation(graph))


"""

#labeling algorithm selection
algo_obj = get_algorithm(algo, mode)
list_or_res = algo_obj.run(graphs, verbose)

#
"""