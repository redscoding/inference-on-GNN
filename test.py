import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

W = np.random.normal(0,1,(9,9))
W= (W+W.T)/2
print(W)
G = nx.from_numpy_array(W)

plt.figure(figsize=(6,6))

plt.imshow(W, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Edge Weight')
plt.title('Adjacency Matrix of Graph')
plt.show()

W_flat = W.flatten()
plt.figure()
plt.hist(W_flat, bins=30, density=True, color='skyblue', edgecolor='black')
plt.title(f'Distribution of W Before Mask star')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()