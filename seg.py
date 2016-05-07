''' This code aims to build an adaptive version of the BPE algorithm'''
import theano
import theano.tensor as T
import numpy as np
from trellis import trellis
X = T.scalar('X')

def build_trellis(train_data):
    trellis_G = trellis()
    return trellis_G
def build_lattice_hidden(num_state, dim_state):
    ''' Build the hidden state matrix'''
    hidden_states = theano.shared(np.zeros([num_state, dim_state]))
    return hidden_states

def step(hidden_states, t, trellis_G, num_inc,
         tokens, out_arcs, W_prob,
         b_prob, W, b, E, dim_hid):
    '''hidden_states : A matrix storing the hidden states for all nodes
    t : The current time step
    num_inc : Number of incoming connections
    tokens : Outgoing edges
    out_arcs : List of child nodes
    '''
    # Get information about the node of the graph
    trellis_node = trellis_G.node_list[t]
    node_ind = trellis_node.ind
    num_inc = trellis_node.num_inc
    out_arcs = trellis_node.out_arcs
    tokens = trellis_node.tokens
    # utility function to slice a tensor
    def _slice(_x, n, dim):
        return _x[:, n*dim:(n+1)*dim]
    # calculate the current hiddent state
    # Dont do if it is the root or has only one parent
    if num_inc > 1:
        hidden_states[:, node_ind*dim_hid:(node_ind+1)*dim_hid] /= num_inc
    # calculate the probability of the node
    prob_s = T.nnet.softmax(T.dot(W_prob, _slice(hidden_states, node_ind, dim_hid)) + b_prob)
    # fill in the prob_s of its children node
    # by adding to it the contribution of this node
    for out_arc, token in zip(out_arcs, tokens):
        out_state = _slice(hidden_states, node_ind, dim_hid) + W*E[token] + b
        hidden_states[out_arc] += T.nnet.sigmoid(out_state)
    return [prob_s]
