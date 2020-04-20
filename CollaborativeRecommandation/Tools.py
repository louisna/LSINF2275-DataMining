import numpy as np
import random


def open_file(filepath):
    """
    The pattern in the file is the following:
    user id | item id | rating | timestamp
    """
    nb_ratings = 100000
    # nb_ratings = 20000
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
            R[int(user_id) - 1, int(item_id) - 1] = int(rating)
            DB[count] = np.array([int(user_id), int(item_id), int(rating)])
            count += 1

    return R, DB


def build_R_from_DB(splits):
    nb_ratings = 100000
    # nb_ratings = 20000
    nb_users = 943
    nb_items = 1682

    # Create the rating matrix
    R = np.zeros((nb_users, nb_items))

    for split in splits:
        for user, item, rating in split:
            R[int(user) - 1, int(item) - 1] = int(rating)

    return R


def load_indexes(DB, cross=10, filepath="ml-100k/u.data"):
    filepath = filepath[:-5] + "_indexes.data"
    try:
        with open(filepath, "r") as fd:
            split = []
            n = []
            for line in fd:
                if line == '\n':
                    if n:
                        split.append(n)
                    n = []
                    continue
                a = line.split('\t')
                n.append(np.array([float(a[0]), float(a[1]), float(a[2])]))
            if n:
                split.append(n)
            return split
    except IOError:
        print("No such file", filepath)
    split_DB = split_ratings(DB, cross)
    with open(filepath, "w+") as fd:
        for s in split_DB:
            for i in s:
                fd.write("" + str(i[0]) + "\t" + str(i[1]) + "\t" + str(i[2]) + "\t" + str(1234567890) + "\n")
            fd.write('\n')
            print('ok')
    print("rip")
    return split_DB


def split_ratings(DB, cross=10):
    split = []
    nrow, _ = DB.shape

    total = list(range(nrow))

    for i in range(cross - 1):
        print(i)
        length = round(len(total) / (cross - i))
        index_sample = random.sample(range(len(total)), length)
        split.append([total[j] for j in index_sample])
        total = [total[j] for j in range(len(total)) if j not in index_sample]
    split.append(list(total))
    split_DB = []
    for spl in split:
        a = [DB[i, :] for i in spl]
        split_DB.append(a)
    return split_DB