#!/usr/bin/env python
"""
"""

from api import State, util
import random

class Bot:

    __max_depth = -1
    __randomize = True

    def __init__(self, randomize=True, depth=6):
        """
        :param randomize: Whether to select randomly from moves of equal value (or to select the first always)
        :param depth:
        """
        self.__randomize = randomize
        self.__max_depth = depth

    def get_move(self, state):
        # type: (State) -> tuple[int, int]
        print('EXECUTING MINIMAX, BUILDING TREE..')
        val, move = self.value(state)
        print('\n__________\nFOUND BEST MOVE:', val, move, '\n_________\n')
        return move

    def value(self, state: State, depth = 0):
        # type: (State, int) -> tuple[float, tuple[int, int]]
        """
        Return the value of this state and the associated move
        :param state:
        :param depth:
        :return: A tuple containing the value of this state, and the best move for the player currently to move
        """
        # print('depth:', depth)
        if state.finished():
            winner, points = state.winner()
            return (points, None) if winner == 1 else (-points, None)

        if depth == self.__max_depth:
            return heuristic(state)

        moves = state.moves()

        if self.__randomize:
            random.shuffle(moves)

        best_value = float('-inf') if maximizing(state) else float('inf')
        best_move = None

        for move in moves:

            next_state = state.next(move)

            # IMPLEMENT: Add a recursive function call so that 'value' will contain the
            # minimax value of 'next_state'
            value, _ = self.value(state=next_state, depth=depth+1)

            if maximizing(state):
                if value > best_value:
                    best_value = value
                    best_move = move
            else:
                if value < best_value:
                    best_value = value
                    best_move = move

        return best_value, best_move

def maximizing(state):
    # type: (State) -> bool
    """
    Whether we're the maximizing player (1) or the minimizing player (2).
    :param state:
    :return:
    """
    return state.whose_turn() == 1

def heuristic(state):
    # type: (State) -> tuple[float, None]
    """
    Estimate the value of this state. Strategy is to play lowest strength 
    legal cards first. To keep the highest strength hand for later in phase 2.

    :param state:
    :return: A heuristic evaluation for the given state (between -1.0 and 1.0)
    """
    return get_hand_strength(state), None

def get_hand_strength(state):
    hand = state.hand()
    current_trump_suit = state.get_trump_suit()

    hand_strength = 0
    for card in hand:
        hand_strength = hand_strength + 
            get_card_strength(card, current_trump_suit)
    return hand_strength

def get_card_strength(card: tuple, current_trump_suit: str):
    card_strength = 0
    rank = util.get_rank(card)
    
    if rank == 'A':
        card_strength = card_strength + 32
    elif rank == '10':
        card_strength = card_strength + 16
    elif rank == 'K':
        card_strength = card_strength + 8
    elif rank == 'Q':
        card_strength = card_strength + 4
    elif rank == 'J':
        card_strength = card_strength + 2

    if util.get_suit(card) == current_trump_suit:
        card_strength = card_strength * 2
    return card_strength
