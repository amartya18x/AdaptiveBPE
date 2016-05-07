class trellis(object):
    node_list = [trellis_node('Phi')]
class trellis_node:
    def __init__(self, name, node_ind = -1,
                 num_inc = 0, out_arcs = [],
                 tokens = []
    ):
        self.name = name
        self.node_ind = node_ind
        self.num_inc = num_inc
        self.out_arcs = out_arcs
        self.tokens = tokens
