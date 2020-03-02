import numpy as np

epsilon = 10 ** (-4)

def markovDecision_matrix(layout, circle):
    safe, safe_cost = markov_decision_matrix_safe(layout, circle)
    normal, normal_cost = markov_decision_matrix_normal(layout, circle)
    print(normal_cost)

    Dice = np.array([0] * 14)
    Expec = np.array([10] * 15)
    Expec[14] = 0
    Old = np.array([0.0] * 15)
    count = 0

    while np.sum((Expec-Old)**2) > epsilon:
        if count == 0:
            Old = np.array(([0.0] * 15))
        else:
            Old = Expec.copy()
        count += 1
        Expec = np.array([0.0] * 15)
        expec_safe = np.dot(safe, Old) + safe_cost
        #print(safe_cost)
        expec_normal = np.dot(normal, Old) + normal_cost
        for i in range(14):
            if expec_safe[i] < expec_normal[i]:
                Expec[i] = expec_safe[i]
                Dice[i] = 1
            else:
                Expec[i] = expec_normal[i]
                Dice[i] = 2
        Expec[14] = 0
    return Expec, Dice

def markov_decision_matrix_safe(layout, circle):
    nb = len(layout)
    transition = np.zeros((nb, nb))
    tier = 1/3

    # First compute the entry for the last square of the layout
    a = [0] * 15
    a[14] = 1
    transition[14] = a

    # Then compute the transition matrix for all other squares
    for i in range(14):
        b = [0.0] * 15
        b[i] = 0.5
        if i == 2:  # Special square
            b[3] = 0.5 * 0.5
            b[10] = 0.5 * 0.5
        else:
            b[i+1] = 0.5
        transition[i] = b
    cost = np.ones(15)
    cost[14] = 0.0
    return transition, np.transpose(cost)


def markov_decision_matrix_normal(layout, circle):
    nb = len(layout)
    transition = np.zeros((nb, nb))
    tier = 1 / 3

    # First compute the entry for the last square of the layout
    a = [0] * 15
    a[14] = 1
    transition[14] = a

    # Then compute the transition matrix for all other squares
    for i in range(14):
        b = [0] * 15
        b[i] = 0
        # Stay on its place
        trap_probability_matrix(i, layout, circle, b)

        # Make a step of 1
        if i == 2:  # Two paths
            trap_probability_matrix(3, layout, circle, b, split=0.5)
            trap_probability_matrix(10, layout, circle, b, split=0.5)
        elif i == 9:  # Reaches 14
            trap_probability_matrix(14, layout, circle, b)
        else:
            trap_probability_matrix(i+1, layout, circle, b)

        # Make a step of 2
        if i == 2:  # Two paths
            trap_probability_matrix(4, layout, circle, b, split=0.5)
            trap_probability_matrix(11, layout, circle, b, split=0.5)
        elif i == 8:  # Reaches 14
            trap_probability_matrix(14, layout, circle, b)
        elif i == 9:
            trap_probability_matrix(15, layout, circle, b)
        else:
            trap_probability_matrix(i+2, layout, circle, b)
        transition[i] = b
    cost = np.ones(15)
    cost[14] = 0.0
    for i in range(14):
        if layout[i] == 3:
            cost[i] += 1
    return transition, np.transpose(cost)


def trap_probability_matrix(pos_arrival, layout, circle, b, split=1.):
    pos_arrival = position(pos_arrival, circle)
    tier = 1/3
    traps = [0.0]*4
    traps[0] = pos_arrival
    traps[1] = 0
    if 10 <= pos_arrival <= 12:
        pos_trap_2 = pos_arrival - 10
    else:
        pos_trap_2 = pos_arrival - 3
    traps[2] = max(0, pos_trap_2)
    traps[3] = pos_arrival
    if layout[pos_arrival] == 4:
        b[int(traps[1])] += (tier * tier) * split
        b[int(traps[2])] += (tier * tier) * split
        b[int(traps[3])] += (tier * tier) * split
    else:
        b[int(traps[layout[pos_arrival]])] += tier * split


def position(new, circle):
    if not circle and new >= 14:
        return 14
    elif circle and new == 14:
        return 14
    return new % 15