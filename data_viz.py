import networkx as nx
import matplotlib.pyplot as plt
def data_viz(G,G_array):
    nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=200, font_size=16)
    plt.show()

    plt.imshow(G_array, cmap='Blues', interpolation='none')
    plt.colorbar(label='Edge Weight')
    plt.title('Adjacency Matrix of Star Graph')
    plt.show()
    """
G,G_array=generate_struct_mask('loll', 9, False)
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=16)
plt.show()

plt.imshow(G_array, cmap='Blues', interpolation='none')
plt.colorbar(label='Edge Weight')
plt.title('Adjacency Matrix of Star Graph')
plt.show()
    """