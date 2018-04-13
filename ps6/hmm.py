import numpy as np

def viterbi(sentence, init_probs, transition_probs, emission_probs):
    """
    Runs the Viterbi algorithm for generating part-of-speech tags from an HMM
    
    Parameters
    ----------
    sentence : numpy array of shape (m,)
        A vector of word indices corresponding to the observations (the words) 
        in the sentence. This vector should contain only the numerical word 
        indices, NOT the word strings themselves.
    
    init_probs : numpy array of shape (m,)
        An array of the probabilities for the value of the first hidden state
    
    transition_probs : numpy array of shape (n,n)
        The transition probability matrix. Entry (i,j) contains the probability of
        transitioning from hidden state j to hidden state 
    
    emission_probs : numpy array of shape (10,)
        A matrix containing the emission probabilities, where the 
        entry in row i and column j gives the probability P(wt=i|st=j)
        of observing word i when in state j.
    
    Output
    ------
    current_state : numpy array of shape (m,)
         a vector of hidden state indices corresponding to the most likely 
         sequence of hidden states (parts of speech) to have generated the input
         sentence. This vector contains the numerical hidden state indices, NOT 
         the part of speech strings themselves.
    """
    np.seterr(divide='ignore') # ignore divide by 0 error
    
    n_obs, n_states = emission_probs.shape
    trans = np.zeros((n_states+1,n_states+1))
    trans[1:, 1:] = transition_probs.T
    trans[0, 1:] = init_probs.T
    emis = np.vstack((np.zeros(n_obs), emission_probs.T))

    n_emissions = emis.shape[1]

    # work in log space to avoid underflow issues
    L = len(sentence)
    current_state = np.zeros(L)
    log_trans = np.log(trans)
    log_emis = np.log(emis)

    n_states = trans.shape[0]
    pTR = np.zeros((n_states,L))

    # assume that the model is in state 1 at step 0
    v = np.asarray([-np.inf]*n_states)
    v[0] = 0
    v_old = v.copy()

    # main loop
    for count in range(L):
        for state in range(n_states):
            # for each state we calculate
            # v[state] = emis[state, sentence[count]] * max_k(v_old[:] * trans[k, state])
            best_val = -np.inf
            best_pTR = 0

            # use a loop to avoid lots of calls to max
            for inner in range(n_states):
                val = v_old[inner] + log_trans[inner, state]
                if val > best_val:
                    best_val = val
                    best_pTR = inner

            # save the best transition information for later backtracking
            pTR[state, count] = best_pTR

            # update v
            v[state] = log_emis[state, sentence[count]] + best_val
        v_old = v.copy()

    # decide which of the final states is most probable
    final_state = np.argmax(v)

    # now back trace through the model
    current_state[L-1] = final_state
    for count in range(L-2, -1, -1):
        current_state[count] = pTR[int(current_state[count+1]), count+1]
    return current_state.astype(int)-1


