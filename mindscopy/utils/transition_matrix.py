import numpy as np

def count_sub_epoch_state_trnasitions(state_seq, K):
    Ne = state_seq.shape[1]

    transitions = np.zeros((Ne-1, K, K))

    for i_e in range(Ne-1):
        idx = np.arange(i_e, i_e+1, Ne)
        transitions[i_e] = count_state_transitions(state_seq[idx], K, 2)

    return transitions

def count_state_transitions(state_seq, K, window_size):
    Nt = state_seq.shape[0] // window_size

    transitions = np.zeros((K, K))

    for i_t in range(Nt):
        for i_w in range(window_size-1):
            origin = state_seq[i_t*window_size + i_w]
            destination = state_seq[i_t*window_size + i_w + 1]
            transitions[origin, destination] += 1

    return transitions