''' This code aims to build an adaptive version of the BPE algorithm'''
import theano
import theano.tensor as T
import numpy as np
from trellis import trellis
X = T.scalar('X')

def build_trellis(train_data):
    trellis_g = trellis()
    return trellis_g
def build_lattice_hidden(num_state, dim_state):
    ''' Build the hidden state matrix'''
    hidden_states = theano.shared(np.zeros([num_state, dim_state]))
    return hidden_states

def traverse_graph(seqs, trellis_G, param):

    # Get the parameters of the model
    [W2, b2, W1, b1, Embed, dim_hidden] = param

    
    # Number of strings in the dataset
    n_strings = sequence.shape[0]

    # Calculate Token length( This will change ! I am not sure when)
    n_states = len(trellis_G.node_list)

    # Initialize the hidden state matrix
    hidden_lat = build_lattice_hidden(n_states, dim_hidden)

    
    # Shared parameters of the model
    shared_vars = [W2, b2, W1, b1, Embed, hidden_lat]
    
    def _step(hidden_states, t, trellis_G,
             W_prob, b_prob, W, b, E, dim_hid):
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
        def _slice(ten, ind, dim):
            ''' Access a row of the tensor'''
            return ten[:, ind*dim:(ind+1)*dim]
        # calculate the current hiddent state
        # Dont do if it is the root or has only one parent
        if num_inc > 1:
            hidden_states[:, node_ind*dim_hid:(node_ind+1)*dim_hid] /= num_inc
        # calculate the transition probabilities from the node
        probs = T.nnet.softmax(T.dot(W_prob, _slice(hidden_states, node_ind, dim_hid)) + b_prob)
        # fill in the prob_s of its children node
        # by adding to it the contribution of this node
        for out_arc, token in zip(out_arcs, tokens):
            out_state = _slice(hidden_states, node_ind, dim_hid) + W*E[token] + b
            hidden_states[out_arc] += T.nnet.sigmoid(out_state)
        return [probs]

    probs = theano.scan(_step,
                sequences=seqs,
                non_sequences=shared_vars,
                n_steps=n_strings)
    return probs
    
    
