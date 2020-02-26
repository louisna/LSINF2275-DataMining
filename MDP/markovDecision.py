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

def random_strategy(layout, circle):
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
            continue  # Frozen
        dice_choice = random.randint(0, 1)
        new, freeze = make_movement(current, dice_choice, layout, circle)
        Dice[current] = new
        current = new
        print(new)


def make_movement(current, dice, layout, circle):
    if not dice:  # random dice
        movement = random.randint(0, 1)
        return current + movement, False
    else:
        movement = random.randint(0, 2)
        new = current + movement
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
            return max(0, new-3), False
        if trap == 3:
            return new, True  # J'AI MIS CA COMME CA MAIS CA PUE EN VRAI
        else:
            raise ArithmeticError  # Should not happen


if __name__ == "__main__":
    layout = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    circle = False
    print(random_strategy(layout, circle))