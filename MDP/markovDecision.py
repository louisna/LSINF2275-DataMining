import numpy as np
import random

safe_proba = np.array([0.5, 0.5])
normal_proba = np.array([1/3, 1/3, 1/3])
epsilon = 10 ** (-4)


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
    # np.sum((Expec-Old)**2) > epsilon

    while count < 100:  # no improvement between Expec and Old
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
    traps[4] = tier * sum(traps[0:3])
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
    return random.randint(1, 2)


def safe_dice():
    """
    Make always the choice of the safe dice
    :return: 1 for the safe dice
    """
    return 1


def normal_dice():
    """
    Make always the choice of the normal dice
    :return: 2 for the normal dice
    """
    return 2


def other_strategy(layout, circle):
    """
    Determine a random strategy to the Snake and ladder game
    This random strategy is implemented as follow: at each turn, select randomly the dice to throw
    :param layout: See above
    :param circle: See above
    :return: See above
    """

    Dice = np.array([0]*len(layout))
    current = 1
    freeze = False
    turn = 0

    while (circle and current != 15) or (not circle and current < 15):
        turn += 1
        if freeze:
            print("FROZEN")
            freeze = False
            continue  # Frozen
        dice_choice = random_dice()  # CHOSE MOVE
        movement = random.randint(0, dice_choice)
        new, freeze = make_movement(current, movement, dice_choice, layout, circle)
        Dice[current] = new
        print("turn:", turn, current, "improvement of", new-current, new)
        current = new


def make_movement(current, movement, dice, layout, circle):
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
    if 11 <= current <= 14:
        on_fast = True
    elif current == 3:
        on_fast = random.randint(0, 1) == 1  # Suppose 1 => take fast
        print("make decision", on_fast)
    if dice == 1:  # Random dice
        if current == 3 and on_fast and movement > 0:
            return current + movement + 7, False  # Add fast index
        elif 4 <= current <= 10 and current + movement >= 11:
            return current + movement + 4, False
        else:
            return current + movement, False
    else:
        new = current + movement
        if current == 3 and on_fast and movement > 0:
            new += 7
        elif 4 <= current <= 10 and new >= 11:  # Consider 11 as the 15
            new += 4
        # Check circle value
        if new >= 15 and not circle:
            return new, False
        if new == 15:  # Should be useless
            return 15, False
        new = new % 16
        if layout[new-1] == 4:
            trap = random.randint(1, 3)
        else:
            trap = layout[new-1]
        if trap == 0:  # Safe place
            return new, False
        if trap == 1:
            return 1, False  # Go back to start
        if trap == 2:
            # Pay attention to if it is the fast path
            if 11 <= new <= 13:
                return new - 3 - 7, False
            else:
                return max(0, new-3), False
        if trap == 3:
            return new, True  # J'AI MIS CA COMME CA MAIS CA PUE EN VRAI
        else:
            raise ArithmeticError  # Should not happen


if __name__ == "__main__":
    layout_0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    layout_1 = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    layout_2 = np.array([0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0])
    layout_3 = np.array([0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0])
    layout_4 = np.array([0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0])
    circle_0 = True
    # other_strategy(layout_4, circle_0)
    print(markovDecision(layout_4, circle_0))
