''' This is the data structure to create the trellis'''
import collections
class trellis_node:
    ''' Each node of the trellis'''
    def __init__(self, name, node_ind=-1,
                 num_inc=0, out_arcs=[],
                 tokens=[]):
        '''name : name of the token'''
        self.name = name
        self.node_ind = node_ind
        self.num_inc = num_inc
        self.out_arcs = out_arcs
        self.tokens = tokens

class trellis(object):
    ''' A data structure to build the trellis to segment the data'''
    node_list = collections.OrderedDict()
    root_node = trellis_node(name='Phi',
                             node_ind=0,
                             num_inc=0)
    node_list['Phi'] = root_node
