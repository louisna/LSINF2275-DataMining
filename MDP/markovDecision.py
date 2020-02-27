import numpy as np
import random


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
    pass


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
        new, freeze = make_movement(current, dice_choice, layout, circle)
        Dice[current] = new
        print("turn:", turn, current, "improvement of", new-current, new)
        current = new


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
    if 11 <= current <= 14:
        on_fast = True
    elif current == 3:
        on_fast = random.randint(0, 1) == 1  # Suppose 1 => take fast
        print("make decision", on_fast)
    if dice == 1:  # Random dice
        movement = random.randint(0, 1)
        if current == 3 and on_fast and movement > 0:
            return current + movement + 7, False  # Add fast index
        elif 4 <= current <= 10 and current + movement >= 11:
            return current + movement + 4, False
        else:
            return current + movement, False
    else:
        movement = random.randint(0, 2)
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
    layout_1 = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    layout_2 = np.array([0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    layout_3 = np.array([0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
    layout_4 = np.array([0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
    circle = False
    other_strategy(layout_4, circle)
