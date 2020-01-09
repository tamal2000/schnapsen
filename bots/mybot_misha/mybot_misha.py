"""
RandomBot -- A simple strategy: enumerates all legal moves, and picks one
uniformly at random.
"""

# Import the API objects
from api import Deck

import random


class Bot:

    def __init__(self):
        pass

    def get_move(self, state):
        # type: (State) -> tuple[int, int]
        """
        My own strategy, based on how I would play personally.

        :param State state: An object representing the gamestate. This includes a link to
            the states of all the cards, the trick and the points.
        :return: A tuple of integers or a tuple of an integer and None,
            indicating a move; the first indicates the card played in the trick, the second a
            potential spouse.
        """

        # All legal moves
        moves = state.moves()

        oppo_card = state.get_opponents_played_card()
        trump_moves = []
        non_trump_moves = []
        suit_following_moves = []

        for move in moves:
            # Can I trump jack exchange?
            if type(move[0]) is None:
                return move

            # Can I play a marriage?
            if type(move[0]) is int and type(move[1]) is int:
                return move

            # Can I play a trump ace and win by 66?
            ''' Not implemented yet '''

            # Fill the lists containing cards by condition
            if type(move[0]) is int:
                suit = Deck.get_suit(move[0])

                # Get all suit following moves
                if oppo_card and suit == Deck.get_suit(oppo_card):
                    suit_following_moves.append(move)

                # Get all trump moves
                if suit == state.get_trump_suit():
                    trump_moves.append(move)

                # Get all non trump moves
                if suit != state.get_trump_suit():
                    non_trump_moves.append(move)

        # -------------- If the opponent played a card first -----
        if oppo_card:
            # Can I follow suit or trump and win?
            for move in suit_following_moves:

                # Can I follow a non-trump suit and win the trick?
                if move in non_trump_moves and oppo_card > move[0]:
                    return move

                # Can I follow a trump suit with high yield and win the trick?
                if move in trump_moves and \
                   Deck.get_rank(oppo_card) in ['A', '10'] and \
                   oppo_card > move[0]:
                    return move

            # ------- Cant win the trick. -------

            # Can I sacrifice a non-trump Jack?
            for move in non_trump_moves:
                if Deck.get_rank(move[0]) == 'J':
                    return move

            # Can I sacrifice by following suit with a queen
            # when the played card is a king?
            for move in suit_following_moves:
                if Deck.get_rank(move[0]) == 'Q' and \
                   Deck.get_rank(oppo_card) == 'K':
                    return move

        # --------------- If I have to play a card first -----------------

        # Lead non trump jack
        for move in non_trump_moves:
            if Deck.get_rank(move[0]) == 'J':
                return move

        # Lead non-marriageable queen/king
        ''' Not implemented yet '''

        # Lead non trump ace.
        for move in non_trump_moves:
            if Deck.get_rank(move[0]) == 'A':
                return move

        # Play anything legal at random.
        return random.choice(moves)
