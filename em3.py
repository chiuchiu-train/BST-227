import numpy as np

def get_data(file):
    f = open(file, 'r')
    data = []
    for line in f:
        line = line.strip()
        data.append(line)
    return data

def init_EM(seq_length, motif_length):
    lmbda = np.random.uniform(0,1,size=(seq_length,))
    lmbda = lmbda/np.sum(lmbda)  # normalization
    psi_0 = np.random.uniform(0,1,size=(4,motif_length))
    psi_0 = psi_0/psi_0.sum(axis=0)
    psi_1 = np.random.uniform(0,1,size=(4,motif_length))
    psi_1 = psi_1/psi_1.sum(axis=0)
    theta = {'lmbda': lmbda, 'psi_0': psi_0, 'psi_1': psi_1}
    return theta

def E_step(data, theta, P):
    dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    C = []
    for i in range(len(data)):
        C_i = []
        for j in range(len(data[0])-P+1):   # 0 to 38-6+1
            print(len(theta['lmbda']))
            C_ij = theta['lmbda'][j]
            C_ij = np.log(theta['lmbda'][j])
            # Iterate through all positions of the motif
            for p in range(P):
                base = data[i][j+p]
                k = dict[base]
                C_ij += np.log(theta['psi_0'][k][p])
            # Iterate through all positions of the non-motif
            for jpr in range(len(data[0])-P+1): # j' is the start position of a non-motif sequence
                if jpr == j: # if j:j+p includes a base that is non motif, score it as background
                    continue
                for p in range(P):
                    base = data[i][jpr+p]
                    k = dict[base]
                    C_ij += np.log(theta['psi_0'][k][p])
            C_i.append(np.exp(C_ij))  # move cij back to probability space
        sm = sum(C_i) # denominator
        C_i = [item/sm for item in C_i]  # normalization
        C.append(C_i)
    return C

def M_step(data, C, P):
    dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    lmbda = np.array(C).sum(axis=0)  # sum by column of matrix
    lmbda = lmbda / 357  # divide all elements in list by N (normalization)

    # Initialize new psi matrices
    psi_1 = np.zeros((4, P))
    psi_0 = np.zeros((4, P))
    for p in range(P):
        for i in range(len(data)):
            for j in range(0, len(data[0])-P+1):
                base = data[i][j+p]
                k = dict[base]
                psi_1[k, p] += C[i][j]
                psi_0[k, p] += 1 - (C[i][j])
    theta = {'lmbda': lmbda, 'psi_1': psi_1, 'psi_0': psi_0}
    return theta

def LLH(data, theta, C, P):
    dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    # First term
    first = C @ theta['lmbda'][0:len(data)-P+1].T
    first = first.sum()  # sum across j's

    # Second term
    second = 0
    for i in range(len(data)):
        for j in range(len(data)-P+1):
            for p in range(P):
                base = data[i][j+p]  # what base? ACGT?
                k = dict[base]  # each base has a k and p
                second += C[i][j]*np.log(theta['psi_1'][k,p]) + (1-C[i][j])*np.log(theta['psi_0'][k,p])
    return (first + second)

# X is matrix of inputs, conv is convergence criterion
def EM(data, motif_length, conv=0.001):
    # Initialization
    LLH_prev = -(np.inf)
    LLH_curr = 0
    theta = init_EM(len(data), motif_length)

    # Main loop
    while abs(LLH_prev - LLH_curr) > conv:
        LLH_prev = LLH_curr
        print(LLH)
        C = E_step(data, theta, motif_length)
        theta = M_step(data, motif_length, C)
        # Recalculate LLH with theta from the M step
        LLH_curr = LLH(data, theta, C)
    return theta


data = get_data("small_seqs.txt")
theta = init_EM(len(data[0]), 6)
posteriors = E_step(data, theta, 6)


"""
elbo = evidence lower bound
# LLH = ELBO + entropy

#entropy
 qlogq = posteriors * np.log(posteriors);
 qlogq[np.where(posteriors == 0)] = 0 #0log0 = 0
 return (expected_complete_LL - np.sum(np.sum(qlogq,axis=1) * ohe_matrix)) #log likelihood = ELBO + entropy, when q=p

sum_i sum_j q(c_i)logq(ci)
"""