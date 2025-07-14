# This class defines a layer in a NN and inherits from Nodes

from node import Node


class ExplicitLayer:
    def __init__(self, id, type, w_init, b_init, n_nodes):
        self.id = id
        self.type = type
        self.n_nodes = n_nodes
        self.node_list = [Node(w_init, b_init, self.id, self.type)]
