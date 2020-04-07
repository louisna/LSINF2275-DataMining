import numpy as np

# first ligne == user  second colonne == item

result = np.array([[11, 12, 13],
                   [21, 22, 23],
                   [31, 32, 33]])

def kNN(R,k): # R = numpy matix 
    V = [] # list de user
    num_rows, num_cols = R.shape
    for i in range(num_rows):
        v = np.array([0.0]*num_cols)
        for j in range(num_cols):
            if R[i,j] > 0:
                v[j] = 1
            else :
                v[j] = 0
        V.append(v)


    nearest_neighbours = []
    for i in range(num_rows):
        kbest = [] # (similitude, number of client)
        for ii in range(num_rows):
            if i != ii :
                kbest.sort(key=lambda t: t[0])
                sim = sim_cosine(V[i],V[ii])
                if len(kbest) >= k and sim > kbest[0] :
                    kbest[0] = (sim, ii)
                elif len(kbest) < k :
                    kbest.append((sim, ii))
        nearest_neighbours.append(kbest)
    
    R_hat = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            sum = 0
            pred = 0
            for e in nearest_neighbours[i]:
                sum += e[0]
                pred += e[0] * R[e[0],j] 
            pred /= sum
            R_hat[i,j] = pred
    
    return R_hat

def sim_cosine(i,p): # i and p are numpy verctors i == 
    return np.dot(i.T,p)/(len(i)*len(p))

print(kNN(result,2))