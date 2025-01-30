import numpy as np

def count_sub_epoch_state_transitions(state_seq, K):
    Nt, Ne = state_seq.shape
    transitions = np.zeros((Ne-1, K, K))

    for i_e in range(Ne-1):
        transitions[i_e] = count_state_transitions(state_seq[:, i_e:i_e+2], K)

    return transitions

def count_state_transitions(state_seq, K):
    Nt, Ne = state_seq.shape

    transitions = np.zeros((K, K))

    for i_t in range(Nt):
        for i_w in range(Ne-1):
            origin = state_seq[i_t, i_w]
            destination = state_seq[i_t, i_w + 1]
            transitions[origin, destination] += 1

    return transitions