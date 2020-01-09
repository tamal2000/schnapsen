"""
RandomBot -- A simple strategy: enumerates all legal moves, and picks one
uniformly at random.
"""

# Import the API objects
from api import State
from api import Deck

import random


class Bot:

    def __init__(self):
        pass

    def get_move(self, state):
        # type: (State) -> tuple[int, int]
        """
        Function that gets called every turn. This is where to implement the strategies.
        Be sure to make a legal move. Illegal moves, like giving an index of a card you
        don't own or proposing an illegal mariage, will lose you the game.
        TODO: add some more explanation
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
        # oppo_is_trump = Deck.get_suit(oppo_card) == Deck.get_trump_suit()

        for move in moves:
            # print('move:', move)
            # Can I trump jack exchange?
            if type(move[0]) is None:
                print('EXCHANGING TRUMP JACK')
                return move

            # Can I play a marriage?
            if type(move[0]) is int and type(move[1]) is int:
                print('PLAYING MARRIAGE')
                return move

            # Can only play cards, i.e. no option for exchanging or marriage
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

        if oppo_card:
            # Can I follow suit or trump and win?
            for move in suit_following_moves:
                # Can I follow a non-trump suit and win the trick?
                if move in non_trump_moves and oppo_card > move[0]:
                    print('NON TRUMP TRICK WIN')
                    return move
                # Can I follow a trump suit with high yield and win the trick?
                if move in trump_moves and \
                   Deck.get_rank(oppo_card) in ['A', '10'] and \
                   oppo_card > move[0]:
                    print('TRUMP TRICK WIN')
                    return move
                
            # Cant follow suit.. Can I throw in a non-trump Jack?
            for move in non_trump_moves:
                if Deck.get_rank(move[0]) == 'J':
                    return move
                

        # Can I follow a trump suit and win the trick
        for move in non_trump_moves:
            pass


        # Can I throw in a non-trump King or Queen that cannot be married?

        # Play anything legal at random.
        return random.choice(moves)
