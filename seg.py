''' This code aims to build an adaptive version of the BPE algorithm'''
import theano
import theano.tensor as T
import numpy as np
X = T.scalar('X')

def build_lattice_hidden(num_state, dim_state):
    ''' Build the hidden state matrix'''
    hidden_states = theano.shared(np.zeros([num_state, dim_state]))
    return hidden_states

def step(hidden_states, t, num_inc,
         tokens, out_arcs, W2, W, E, dim_hid):
    '''hidden_states : A matrix storing the hidden states for all nodes
    t : The current time step
    num_inc : Number of incoming connections
    tokens : Current token
    out_arcs : List of outgoing connections
    '''
    # utility function to slice a tensor
    def _slice(_x, n, dim):
        return _x[:, n*dim:(n+1)*dim]
    # calculate the current hiddent state
    if num_inc > 1:
        hidden_states[:, t*dim_hid:(t+1)*dim_hid] /= num_inc
    prob_s = T.nnet.softmax(T.dot(W2, _slice(hidden_states, t, dim_hid)))
    for out_arc, token in zip(out_arcs, tokens):
        out_state = _slice(hidden_states, t, dim_hid) + W*E[token]
        hidden_states[out_arc] += out_state
    return [prob_s]
