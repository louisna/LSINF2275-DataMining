import numpy as np
import random
import matplotlib.pyplot as plt
from markovDecisionMatrix import markovDecision_matrix

safe_proba = np.array([0.5, 0.5])
normal_proba = np.array([1/3, 1/3, 1/3])
epsilon = 10 ** (-7)

random.seed(1998)


def markovDecision(layout, circle):
    """
    This function launches the Markov Decision Process algorithm to determine the optimal strategy regarding the
    choice of the dice in the Snakes and Ladders game, using the “value iteration” method.
    :param layout: a vector of type numpy.ndarray that represents the layout of the game, containing 15 values
                    representing the 15 squares of the Snakes and Ladders game:
                        layout[i] = 0 if it is an ordinary square
                        = 1 if it is a “restart” trap (go back to square 1)
                        = 2 if it is a “penalty” trap (go back 3 steps)
                        = 3 if it is a “prison” trap (skip next turn)
                        = 4 if it is a “mystery” trap (random effect among the three previous)
    :param circle: circle: a boolean variable (type bool), indicating if the player must land exactly on the finish
                    square to win (circle = True) or still wins by overstepping the final square (circle = False)
    :return:a type list containing the vectors [Expec,Dice]:
                - Expec: a vector of type numpy.ndarray containing the expected cost (= number of turns) associated to
                the 14 squares of the game, excluding the finish one.
                - Dice: a vector of type numpy.ndarray containing the choice of the dice to use for each of the 14
                squares of the game (1 for “security” dice, 2 for “normal” dice), excluding the finish one.
    """
    Dice = np.array([0]*14)
    Expec = np.array([10]*15)
    Expec[14] = 0
    Old = np.array([0.0]*15)

    count = 0
    tier = 1/3

    while np.sum((Expec-Old)**2) > epsilon:  # no improvement between Expec and Old
        if count == 0:
            Old = np.array([0.0] * 15)
        else:
            Old = Expec.copy()  # ?
        count += 1
        Expec = np.array([0.0] * 15)
        Expec[14] = 0
        for i in range(len(Dice)-1, -1, -1):
            # Safe dice value computation
            safe_value = 1 + (0.5 * Old[i])
            if i == 2:
                safe_value += 0.5 * (0.5 * Old[3] + 0.5 * Old[10])
            elif i == 9:
                safe_value += 0  # Useless
            else:
                safe_value += 0.5 * Old[i+1]
            # Normal dice value computation
            normal_value = 1 + (tier * trap_probability(Old, i, layout, circle))  # Stay
            # Movement by 1
            if i == 2:
                normal_value += tier * 0.5 * (trap_probability(Old, 3, layout, circle) + trap_probability(Old, 10, layout, circle))
            elif i == 9:  # +1 => finish
                normal_value += 0  # Useless
            else:
                normal_value += tier * trap_probability(Old, i+1, layout, circle)
            # Movement by 2
            if i == 2:
                normal_value += tier * 0.5 * (trap_probability(Old, 4, layout, circle) + trap_probability(Old, 11, layout, circle))
            elif i == 8:  # +1 => finish
                normal_value += 0  # Useless
            else:
                if i == 9:
                    new_pos = 15
                else:
                    new_pos = i + 2
                normal_value += tier * trap_probability(Old, new_pos, layout, circle)

            # Compute min_arg and min_value
            if safe_value < normal_value:
                Expec[i] = safe_value
                Dice[i] = 1
            else:
                Expec[i] = normal_value
                Dice[i] = 2
        # Update values
    # print(Expec)
    # print(Dice)
    return Expec, Dice


def trap_probability(Old, pos_arrival, layout, circle):
    pos_arrival = position(pos_arrival, circle)
    tier = 1/3
    traps = [0.0]*5
    traps[0] = Old[pos_arrival]
    traps[1] = Old[0]
    if 10 <= pos_arrival <= 12:
        pos_trap_2 = pos_arrival - 10
    else:
        pos_trap_2 = pos_arrival - 3
    traps[2] = Old[max(0, pos_trap_2)]
    traps[3] = 1 + Old[pos_arrival]
    traps[4] = tier * sum(traps[1:4])
    return traps[layout[pos_arrival]]


def position(new, circle):
    if not circle and new >= 14:
        return 14
    elif circle and new == 14:
        return 14
    return new % 15


def random_dice():
    """
    Make always a random choice between the safe and the normal dices
    :return: 0 for the safe dice, and 1 for the normal dice
    """
    return [random.randint(1, 2) for i in range(15)]


def safe_dice():
    """
    Make always the choice of the safe dice
    :return: 1 for the safe dice
    """
    return [1] * 14


def normal_dice():
    """
    Make always the choice of the normal dice
    :return: 2 for the normal dice
    """
    return [2] * 14


def play_game(layout, circle, dice_strategy):
    """
    Determine a random strategy to the Snake and ladder game
    This random strategy is implemented as follow: at each turn, select randomly the dice to throw
    :param layout: See above
    :param circle: See above
    :param dice_strategy:
    :return: See above
    """

    current = 0
    freeze = False
    turn = 0

    while (circle and current != 14) or (not circle and current < 14):
        turn += 1
        if turn > 50:
            return 50
        if freeze:
            freeze = False
            continue  # Frozen
        else:
            new, freeze = make_movement(current, dice_strategy[current], layout, circle)
            current = new
    return turn


def make_movement(current, dice, layout, circle):
    """
    Function that makes the movement, according to the current position of the player, the chosen dice, the
    layout of the grid, and if we authorize to circle
    :param current: current position of the player
    :param dice: The chosen dice, can be safe or normal
    :param layout: The layout of the grid
    :param circle: Boolean value indicating if we can circle in the grid
    :return: The new position on the grid, and a boolean value indicating if the player is frozen for the next round
    """
    on_fast = False
    if 10 <= current <= 13:
        on_fast = True
    elif current == 2:
        on_fast = random.randint(0, 1) == 1  # Suppose 1 => take fast
    if dice == 1:  # Safe dice
        movement = random.randint(0, 1)
        if current == 2 and on_fast and movement > 0:
            return current + movement + 7, False  # Add fast index
        elif 3 <= current <= 9 and current + movement >= 10:
            return current + movement + 4, False
        else:
            return current + movement, False
    else:
        movement = random.randint(0, 2)
        new = current + movement
        if current == 2 and on_fast and movement > 0:
            new += 7
        elif 3 <= current <= 9 and new >= 10:  # Consider 11 as the 15
            new += 4
        # Check circle value
        if new >= 14 and not circle:
            return new, False
        if new == 14:  # Should be useless
            return 14, False
        new = new % 15
        if layout[new] == 4:
            trap = random.randint(1, 3)
        else:
            trap = layout[new]
        if trap == 0:  # Safe place
            return new, False
        if trap == 1:
            return 0, False  # Go back to start
        if trap == 2:
            # Pay attention to if it is the fast path
            if 10 <= new <= 12:
                return new - 3 - 7, False
            else:
                return max(0, new-3), False
        if trap == 3:
            return new, True  # Maybe find a better idea
        else:
            raise ArithmeticError  # Should not happen


def gen_map():
    m = [0] * 15
    for i in range(1, 14):
        if random.randint(0, 1) == 0:
            m[i] = random.randint(1, 4)
    return m


def gen_map_with_probabilities(prob, trap):
    """
    prob == [0,100]
    trap == [1,4]
    """
    m = [0] * 15
    for i in range(1, 14):
        if random.randint(0, 100) <= prob:
            m[i] = trap
    return m


def gen_map_with_nb(nb, trap):
    """
    prob == [1,13]
    trap == [1,4]
    """
    m = [0] * 15
    for i in range(0, nb):
        while True:
            case = random.randint(1, 13)
            if m[case] == 0:
                if trap == 5:
                    m[case] = random.randint(1, 4)
                break
    return m


def box_plot():
    circle = False
    layout = gen_map_with_probabilities(20, 1)
    print("map", layout)
    always_safe = safe_dice()
    always_normal = normal_dice()
    always_random = random_dice()
    markov = markovDecision(layout, circle)[1]

    nb = 1000  # Nb of experiments
    safe_result = np.array([0.0] * nb)
    normal_result = np.array([0.0] * nb)
    random_result = np.array([0.0] * nb)
    markov_result = np.array([0.0] * nb)

    for i in range(nb):
        safe_result[i] = play_game(layout, circle, always_safe)
        normal_result[i] = play_game(layout, circle, always_normal)
        random_result[i] = play_game(layout, circle, always_random)
        markov_result[i] = play_game(layout, circle, markov)

    plt.boxplot([markov_result, safe_result, normal_result, random_result], showmeans=True, meanline=True,
                showfliers=False)
    plt.title("Boxplot of empirical cost for the 4 strategies")
    plt.xticks([1, 2, 3, 4], ['Markov', 'Always safe', 'Always normal', 'Always random'])
    plt.ylabel("Experimental cost")
    plt.show()


def simu(trap, circle=False):
    exp = 21
    step = 5
    expected = np.array([0.0] * exp)
    experimental = np.array([0.0] * exp)
    always_one = np.array([0.0] * exp)
    always_two = np.array([0.0] * exp)
    always_random = np.array([0.0] * exp)
    one = safe_dice()
    two = normal_dice()
    for i in range(exp):
        ran = random_dice()
        nb_maps = 20
        markov_by_map = np.array([0.0] * nb_maps)
        markov_exp_map = np.array([0.0] * nb_maps)
        always_one_by_map = np.array([0.0] * nb_maps)
        always_two_by_map = np.array([0.0] * nb_maps)
        always_random_by_map = np.array([0.0] * nb_maps)
        for k in range(nb_maps):
            nb = 300  # Nb of experiments
            markov_result = np.array([0.0] * nb)
            layout = gen_map_with_probabilities(i * step, trap)
            tmp = markovDecision(layout, circle)
            markov_exp_map[k] = tmp[0][0]
            expected[i] = tmp[0][0]
            markov = tmp[1]
            one_result = np.array([0.0] * nb)
            two_result = np.array([0.0] * nb)
            random_result = np.array([0.0] * nb)
            for j in range(nb):
                markov_result[j] = play_game(layout, circle, markov)
                one_result[j] = play_game(layout, circle, one)
                two_result[j] = play_game(layout, circle, two)
                random_result[j] = play_game(layout, circle, ran)

            markov_by_map[k] = np.mean(markov_result)
            always_one_by_map[k] = np.mean(one_result)
            always_two_by_map[k] = np.mean(two_result)
            always_random_by_map[k] = np.mean(random_result)
        print(i)
        expected[i] = np.mean(markov_exp_map)
        experimental[i] = np.mean(markov_by_map)
        always_one[i] = np.mean(always_one_by_map)
        always_two[i] = np.mean(always_two_by_map)
        always_random[i] = np.mean(always_random_by_map)

    print(experimental)
    plt.plot(list(range(0, exp)), expected, marker='s', color="C0",
             label="Expected", alpha=1.0, linestyle='dashed')
    plt.plot(list(range(0, exp)), experimental, marker='^', color="C1",
             label="Experimental", alpha=0.8, linestyle='dashed')

    plt.plot(list(range(0, exp)), always_one, marker='o', color="C2", markersize=7,
             label="Always safe", alpha=0.8, linestyle='dashed')
    plt.plot(list(range(0, exp)), always_two, marker='D', color="C3", markersize=6,
             label="Always normal", alpha=0.7, linestyle='dashed')
    plt.plot(list(range(0, exp)), always_random, marker='X', color="C4", markersize=5,
             label="Always random", alpha=0.6, linestyle='dashed')

    plt.ylim(top=20)
    plt.legend()
    plt.xlabel("Number of traps on the map")
    plt.ylabel("Number of turns to finish the game")
    plt.title("Map only composed of traps of type " + str(trap) + " with circle=" + str(circle))
    plt.grid()
    # plt.savefig("exp_vs_exp_perct_" + str(trap) + "_" + str(circle) + ".svg")
    plt.show()


def simu2(trap, circle=False):
    exp = 13
    ewa = 1
    expected = np.array([0.0] * exp)
    experimental = np.array([0.0] * exp)
    always_one = np.array([0.0] * exp)
    always_two = np.array([0.0] * exp)
    always_random = np.array([0.0] * exp)
    one = safe_dice()
    two = normal_dice()
    for i in range(exp):
        ran = random_dice()
        nb_maps = 30
        markov_by_map = np.array([0.0] * nb_maps)
        markov_exp_map = np.array([0.0] * nb_maps)
        always_one_by_map = np.array([0.0] * nb_maps)
        always_two_by_map = np.array([0.0] * nb_maps)
        always_random_by_map = np.array([0.0] * nb_maps)
        for k in range(nb_maps):
            nb = 400  # Nb of experiments
            markov_result = np.array([0.0] * nb)
            circle = False
            layout = gen_map_with_nb(i*ewa, trap)
            tmp = markovDecision(layout, circle)
            markov_exp_map[k] = tmp[0][0]
            expected[i] = tmp[0][0]
            markov = tmp[1]
            one_result = np.array([0.0] * nb)
            two_result = np.array([0.0] * nb)
            random_result = np.array([0.0] * nb)
            for j in range(nb):
                markov_result[j] = play_game(layout, circle, markov)
                one_result[j] = play_game(layout, circle, one)
                two_result[j] = play_game(layout, circle, two)
                random_result[j] = play_game(layout, circle, ran)

            markov_by_map[k] = np.mean(markov_result)
            always_one_by_map[k] = np.mean(one_result)
            always_two_by_map[k] = np.mean(two_result)
            always_random_by_map[k] = np.mean(random_result)
        print(i)
        expected[i] = np.mean(markov_exp_map)
        experimental[i] = np.mean(markov_by_map)
        always_one[i] = np.mean(always_one_by_map)
        always_two[i] = np.mean(always_two_by_map)
        always_random[i] = np.mean(always_random_by_map)

    print(experimental)
    plt.plot(list(range(0, exp)), expected, marker='s', color="C0",
             label="Expected", alpha=1.0, linestyle='dashed')
    plt.plot(list(range(0, exp)), experimental, marker='^', color="C1",
             label="Experimental", alpha=0.8, linestyle='dashed')

    plt.plot(list(range(0, exp)), always_one, marker='o', color="C2", markersize=7,
             label="Always safe", alpha=0.8, linestyle='dashed')
    plt.plot(list(range(0, exp)), always_two, marker='D', color="C3", markersize=6,
             label="Always normal", alpha=0.7, linestyle='dashed')
    plt.plot(list(range(0, exp)), always_random, marker='X', color="C4", markersize=5,
             label="Always random", alpha=0.6, linestyle='dashed')

    plt.ylim(top=20)
    plt.legend()
    plt.xlabel("Number of traps on the map")
    plt.ylabel("Number of turns to finish the game")
    plt.title("Map only composed of traps of type " + str(trap) + " with circle=" + str(circle))
    plt.grid()
    # plt.savefig("exp_vs_exp_" + str(trap) + "_" + str(circle) + ".svg")
    plt.show()


if __name__ == "__main__":
    # layout_0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # layout_1 = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    # layout_2 = np.array([0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0])
    # layout_3 = np.array([0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0])
    # layout_4 = np.array([0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0])
    # layout_5 = np.array([0, 4, 0, 1, 0, 2, 3, 4, 0, 0, 2, 1, 0, 4, 0])
    # circle_0 = True
    # play_game(gen_map(), circle_0)
    # print(markovDecision(layout_4, circle_0))
    # simu(1)
    simu2(5, circle=True)
