import numpy as np
import sys
import random
from queue import PriorityQueue
random.seed(1998)
# first ligne == user  second colonne == item

result = np.array([[1, 0, 3],
                   [2, 0, 5],
                   [0, 2, 0],
                   [2, 1, 2],
                   [3, 3, 3],
                   [2, 0, 5]])


def kNN(R,k): # R = numpy matix
    # calculing user's vector V[i] = vi in the slides
    sim_dico = {}
    V = [] # list de user
    print('debut')
    num_rows, num_cols = R.shape
    for i in range(num_rows):
        v = np.array([0.0]*num_cols)
        for j in range(num_cols):
            if R[i,j] > 0:
                #v[j] = R[i,j] # slight better result with sim_cosine
                v[j] = 1
            else :
                v[j] = 0
        V.append(v)
    #print(V)

    # computing the k nearest neighbours
    nearest_neighbours = [] # nearest_neighbours[i] == k nearest neighbours of user i
    for i in range(num_rows):
        kbestPQ = PriorityQueue() # (similitude, number of client)
        for ii in range(num_rows):
            if i != ii :
                sim = sim_dico.get((i,ii),-1)
                if (sim == -1):
                    sim = sim_dico.get((ii,i),-1)
                    if (sim == -1):
                        sim = sim_cosine(V[i],V[ii])
                    sim_dico[(i,ii)]  = sim
                #print(sim)
                #print(np.dot(V[i],V[ii].T))
                #print(sim)
                if kbestPQ.qsize() < k :
                    kbestPQ.put((sim, ii))
                    continue
                # len(kbest) == k
                e = kbestPQ.get() # O(1) instead O(nlogn)
                if sim > e[0] :
                    kbestPQ.put((sim, ii)) # O(logn) instead O(1)
                else :
                    kbestPQ.put(e) # O(logn) instead O(1)
        kbest = []
        while(not kbestPQ.empty()):
            kbest.append(kbestPQ.get())
        nearest_neighbours.append(kbest)

    # computing the prediction
    R_hat = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            denom = 0
            pred = 0
            for e in nearest_neighbours[i]: # here PQ is inefficient because must creat a new PQ
                denom += abs(e[0])
                pred += e[0] * R[e[1],j]
            if denom == 0:
                pred = 0
            else:
                pred /= denom
            R_hat[i,j] = pred
    
    print('fin')

    return R_hat

def dummy(R):
    num_rows, num_cols = R.shape
    R_hat = np.ones((num_rows, num_cols))
    for i in range(num_cols):
        R_hat[:,i] = np.mean(R[:,i]) * R_hat[:,i]
    return R_hat

def confusion_matrix(vi,vj):
    conf = np.zeros((2, 2))
    for i in range(len(vi)):
        conf[int(vi[i]),int(vj[i])] +=  1
    return (conf[0,0], conf[0,1], conf[1,0], conf[1,1]) # slide 153 (a b c d)

#similitude slide 154
def sim_1(i,p):
    (a,b,c,d) = confusion_matrix(i,p)
    if (a+b+c+d)== 0:
        return 0.0
    else:
        return (a+d)/(a+b+c+d)

def sim_2(i,p):
    (a,b,c,d) = confusion_matrix(i,p)
    if (2*(a+d)+b+c) == 0:
        return 0.0
    else:
        return 2*(a+d)/(2*(a+d)+b+c)

def sim_3(i,p):
    (a,b,c,d) = confusion_matrix(i,p)
    if (a+2*(b+c)+d) == 0:
        return 0.0
    else:
        return (a+d)/(a+2*(b+c)+d)

def sim_4(i,p):
    (a,b,c,d) = confusion_matrix(i,p)
    if (a+b+c+d) == 0:
        return 0.0
    else:
        return (a)/(a+b+c+d)

def sim_5(i,p): #most popular
    (a,b,c,d) = confusion_matrix(i,p)
    if (a+b+c) == 0:
        return 0.0
    else :
        return (a)/(a+b+c)

def sim_6(i,p):
    (a,b,c,d) = confusion_matrix(i,p)
    if (2*a+b+c) == 0:
        return 0.0
    else:
        return (2*a)/(2*a+b+c)

def sim_7(i,p):
    (a,b,c,d) = confusion_matrix(i,p)
    if (a+2*(b+c)) == 0:
        return 0.0
    else :
        return (a)/(a+2*(b+c))

def sim_8(i,p):
    (a,b,c,d) = confusion_matrix(i,p)
    if (b+c) == 0:
        return 0.0
    else:
        return (a)/(b+c)

#similitude slide 155
def sim_cosine(i,p):
    return np.dot(i.T,p)/(len(i)*len(p))


def split_ratings(DB, cross=10):

    split = []
    nrow, _ = DB.shape

    total = list(range(nrow))

    for i in range(cross-1):
        print(i)
        length = round(len(total)/(cross-i))
        index_sample = random.sample(range(len(total)), length)
        split.append([total[j] for j in index_sample])
        total = [total[j] for j in range(len(total)) if j not in index_sample]
    split.append(list(total))
    split_DB = []
    for spl in split:
        a = [DB[i,:] for i in spl]
        split_DB.append(a)
    return split_DB


def load_indexes(DB, cross=10, filepath="ml-100k/u.data"):
    filepath = filepath[:-5] + "_indexes.data"
    try:
        with open(filepath, "r") as fd:
            split = []
            n = []
            for line in fd:
                if line == '\n':
                    if n != []:
                        split.append(n)
                    n = []
                    continue
                a = line.split('\t')
                n.append(np.array([float(a[0]), float(a[1]), float(a[2])]))
            if n != []:
                split.append(n)
            return split
    except IOError:
        print("No such file", filepath)
    split_DB = split_ratings(DB, cross)
    with open(filepath, "w+") as fd:
        for s in split_DB:
            for i in s:
                fd.write("" + str(i[0]) + "\t" + str(i[1]) + "\t" + str(i[2]) + "\t" + str(1234567890)+"\n")
            fd.write('\n')
            print('ok')
    print("rip")
    return split_DB
            


def build_R_from_DB(splits):
    nb_ratings = 100000
    # nb_ratings = 23
    nb_users = 943
    nb_items = 1682

    # Create the rating matrix
    R = np.zeros((nb_users, nb_items))

    for split in splits:
        for user, item, rating in split:
            R[int(user)-1, int(item)-1] = int(rating)
    
    return R


def cross_validation(DB, k, n_cross=10):
    MSE_g = np.zeros(10)
    MAE_g = np.zeros(10)

    split = load_indexes(DB, n_cross)

    for v in range(n_cross):
        # v is the testo set, splitted[-v] is the training set
        R = build_R_from_DB(split[:v] + split[v+1:])
        R_test = build_R_from_DB([split[v]])
        print(v)
        R_hat = kNN(R, k)
        print("okkk")
        #R_hat = dummy(R)
        nrow, ncol = R_test.shape
        MSE = 0.0
        MAE = 0.0
        for i in range(nrow):
            for j in range(ncol):
                if R_test[i,j] != 0 :
                    MSE += (R_test[i,j] - R_hat[i,j]) ** 2
                    MAE += abs(R_test[i,j] - R_hat[i,j])
        MSE /= len(split[v])
        MAE /= len(split[v])
        MSE_g[v] = MSE
        MAE_g[v] = MAE

    MSE = np.mean(MSE_g)
    MAE = np.mean(MAE_g)

    return MSE, MAE




def open_file(filepath):
    """
    The pattern in the file is the following:
    user id | item id | rating | timestamp
    """
    nb_ratings = 100000
    # nb_ratings = 23
    nb_users = 943
    nb_items = 1682

    DB = np.zeros((nb_ratings, 3))

    # Create the rating matrix
    R = np.zeros((nb_users, nb_items))

    with open(filepath, "r") as fd:
        count = 0
        for line in fd:
            if not line or line == "\n":
                continue
            user_id, item_id, rating, timestamp = list(line.split('\t'))
            R[int(user_id)-1, int(item_id)-1] = int(rating)
            DB[count] = np.array([int(user_id), int(item_id), int(rating)])
            count += 1

    return R, DB


if __name__ == "__main__":
    R, DB = open_file("ml-100k/u.data")
    #R_hat = kNN(result,3)
    #num_rows, num_cols = R_hat.shape
    #print(result)
    #print(R_hat)
    #print(R_hat[:,num_cols-1])
    #print(split_ratings(R))
    print(cross_validation(DB, 8))
    #print(split_ratings(100, cross=40))
    #print(len(result))

# (5.38067, 1.78837)                        k=1     cosine + V binary
# (3.7827014565738692, 1.5766381749075415)  k=3     cosine + V binary
# (3.5646839903019787, 1.5557354774322034)  k=6     cosine + V binary
# (3.567794058857217, 1.5642715439959054)   k=8     cosine + V binary
# (3.6052611051149492, 1.5779623255306814)  k=10    cosine + V binary
# (3.9054899999999995, 1.57295)             k=3     cosine + V binary + round pred
# (3.7516693338785023, 1.5449510928303942)  k=3     cosine + V rating
# (3.8575299999999997, 1.5387899999999999)  k=3     cosine + V rating+ round pred