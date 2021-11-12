import numpy as np

def get_data(file):
    f = open(file, 'r')
    data = []
    for line in f:
        line = line.strip()
        data.append(line)
    return data

# L = sequence length, P = motif length
def init_EM(L, P):
    lmbda = np.random.uniform(0,1,size=(L,))
    lmbda = lmbda/np.sum(lmbda)  # normalization
    psi_0 = np.random.uniform(0,1,size=(4,P))
    psi_0 = psi_0/psi_0.sum(axis=0)
    psi_1 = np.random.uniform(0,1,size=(4,P))
    psi_1 = psi_1/psi_1.sum(axis=0)
    theta = {'lmbda': lmbda, 'psi_0': psi_0, 'psi_1': psi_1}
    return theta

def E_step(data, theta, P):
    dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    C = []
    for i in range(len(data)):
        C_i = []
        for j in range(len(data[0])-P+1):   # 0 to 38-6+1
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
            for j in range(len(data[0])-P+1):
                base = data[i][j+p]
                k = dict[base]
                psi_1[k, p] += C[i][j]
                psi_0[k, p] += 1 - (C[i][j])
    psi_1 /= len(data)  # normalization
    psi_0 /= len(data)*(len(data[0])-P)  # normalization
    theta = {'lmbda': lmbda, 'psi_1': psi_1, 'psi_0': psi_0}
    return theta

def LLH(data, theta, C, P):
    dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    # First term
    first = 0
    for i in range(len(data)):
        for j in range(len(C[i])):
            first += C[i][j] * np.log(theta['lmbda'][j])
    # Second term
    second = 0
    for i in range(len(data)):
        for j in range(len(data[0])-P+1):
            for p in range(P):
                base = data[i][j+p]  # what base? ACGT?
                k = dict[base]  # each base has a k and p
                second += C[i][j]*np.log(theta['psi_1'][k,p]) + (1-C[i][j])*np.log(theta['psi_0'][k,p])
    return (first + second)

# X is matrix of inputs, conv is convergence criterion
def EM(P, seed, conv=0.001, iter=5):
    data = get_data("sequence.padded.txt")

    # Initialization
    LLH_prev = -200000
    LLHs = []
    LLHs.append(LLH_prev)
    LLH_curr = 0
    np.random.seed(seed)
    theta = init_EM(len(data[0]), P)

    # 1st E step and M step
    posteriors = E_step(data, theta, P)
    theta = M_step(data, posteriors, P)

    # Main loop
    counter = 0
    while counter < iter:
        LLH_prev = LLH_curr
        print("Iteration", counter, "prev", LLH_prev, "curr", LLH_curr)
        posteriors = E_step(data, theta, P)
        theta = M_step(data, posteriors, P)
        # Recalculate LLH with theta from the M step
        LLH_curr = LLH(data, theta, posteriors, P)
        LLHs.append(LLH_curr)
        counter += 1
    return LLHs


# test code
#data = get_data("small_seqs.txt")
#theta = init_EM(len(data[0]), 6)
#print(len(theta['lmbda']))
#posteriors = E_step(data, theta, 6)
#theta = M_step(data, posteriors, 6)
#loglh = LLH(data, theta, posteriors, 6)
final_theta = EM(18, 1)
print(final_theta)


"""
elbo = evidence lower bound
# LLH = ELBO + entropy

#entropy
 qlogq = posteriors * np.log(posteriors);
 qlogq[np.where(posteriors == 0)] = 0 #0log0 = 0
 return (expected_complete_LL - np.sum(np.sum(qlogq,axis=1) * ohe_matrix)) #log likelihood = ELBO + entropy, when q=p

sum_i sum_j q(c_i)logq(ci)
"""
