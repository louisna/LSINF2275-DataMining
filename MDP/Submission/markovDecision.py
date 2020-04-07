import numpy as np
import random

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

    # Number of iterations
    count = 0
    tier = 1/3

    while np.sum((Expec-Old)**2) > epsilon:  # no improvement between Expec and Old
        if count == 0:
            Old = np.array([0.0] * 15)
        else:
            Old = Expec.copy()
        count += 1
        Expec = np.array([0.0] * 15)
        # Enforces the last state to 0 (should be useless)
        Expec[14] = 0
        # For each state from last to first
        for i in range(len(Dice)-1, -1, -1):
            # Safe dice value computation
            # Cost of the turn + half a chance not to move
            safe_value = 1 + (0.5 * Old[i])
            if i == 2:  # Half a chance to take the fast path or not
                safe_value += 0.5 * (0.5 * Old[3] + 0.5 * Old[10])
            elif i == 9:  # Completes the game, special case because 9 => 14
                safe_value += 0  # Useless
            else:  # Normal movement by 1 without risk of trap
                safe_value += 0.5 * Old[i+1]
            # Normal dice value computation
            # Cost of the turn + 1/3 chance not to move
            normal_value = 1 + (tier * trap_probability(Old, i, layout, circle))  # Stay
            # Movement by 1
            if i == 2:  # Half a chance to take the fast path or not
                normal_value += tier * 0.5 * (trap_probability(Old, 3, layout, circle) +
                                              trap_probability(Old, 10, layout, circle))
            elif i == 9:  # Completes the game, special case because 9 => 14
                normal_value += 0  # Useless
            else:  # Normal movement by 1 with risk
                normal_value += tier * trap_probability(Old, i+1, layout, circle)
            # Movement by 2
            if i == 2:  # Half a chance to take the fast path or not
                normal_value += tier * 0.5 * (trap_probability(Old, 4, layout, circle) +
                                              trap_probability(Old, 11, layout, circle))
            elif i == 8:  # +1 => finish
                normal_value += 0  # Useless
            else:
                if i == 9:  # 9 + 2 => 15 (could come back to 0 if circle=True)
                    new_pos = 15
                else:
                    new_pos = i + 2  # Normal movement by 2
                normal_value += tier * trap_probability(Old, new_pos, layout, circle)

            # Compute min_arg and min_value
            if safe_value < normal_value:
                Expec[i] = safe_value
                Dice[i] = 1
            else:
                Expec[i] = normal_value
                Dice[i] = 2
        # Update values
    return [Expec[:-1], Dice]


def trap_probability(Old, pos_arrival, layout, circle):
    """
    Similar to the 'f' function presented during the report. This function handles the traps, supposing that the player
    arrives at 'pos_arrival'.
    :param Old: The expected values of the last iteration
    :param pos_arrival: the arrival position of the player after the thrown of the dice, but before the triggering
                        of the traps
    :param layout: the board
    :param circle: the circle value
    :return:
    """
    # Format the position
    pos_arrival = position(pos_arrival, circle)
    tier = 1/3
    traps = [0.0]*5
    # traps[0] = no trap on the board
    traps[0] = Old[pos_arrival]
    # First trap: go back to 0
    traps[1] = Old[0]
    # Second trap
    if 10 <= pos_arrival <= 12:  # Special handling if at these positions
        pos_trap_2 = pos_arrival - 10
    else:
        pos_trap_2 = pos_arrival - 3
    traps[2] = Old[max(0, pos_trap_2)]
    # Third trap. The additional cost of the freeze is represented here
    traps[3] = 1 + Old[pos_arrival]
    # The fourth trap is a sum of the 3 first
    traps[4] = tier * sum(traps[1:4])
    return traps[layout[pos_arrival]]


def position(new, circle):
    """
    Format the position to stay in the board
    :param new: the position of the player after the dice, but before the trigger of the traps
    :param circle:
    :return:
    """
    if not circle and new >= 14:
        return 14
    elif circle and new == 14:
        return 14
    return new % 15
