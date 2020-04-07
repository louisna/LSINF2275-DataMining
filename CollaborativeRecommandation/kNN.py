import numpy as np

def kNN(R):



    return 0

def sim_cosine(i,p):

    return 0








def open_file(filepath):
    """
    The pattern in the file is the following:
    user id | item id | rating | timestamp
    """
    nb_ratings = 100000
    nb_users = 943
    nb_items = 1982

    # Create the rating matrix
    R = np.zeros((nb_users, nb_items))

    with open(filepath, "r") as fd:
        for line in fd:
            if not line or line == "\n":
                continue
            user_id, item_id, rating, timestamp = list(line.split('\t'))
            R[int(user_id)-1, int(item_id)-1] = int(rating)

    return R


if __name__ == "__main__":
    R = open_file("ml-100k/u.data")
