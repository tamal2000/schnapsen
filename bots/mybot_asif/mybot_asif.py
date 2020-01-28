"""
RandomBot -- A simple strategy: enumerates all legal moves, and picks one
uniformly at random.
"""

# Import the API objects
from api import State, Deck, util
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
        print ("list of moves",moves)

        trump_cards = []
        other_cards = []
        low_cards = []

        for move in moves:
            # Performs a trump jack exchange
            if move[0] is None:
                return move
            # Melds a marriage
            if move[1] is not None:
                return move

            if Deck.get_rank(move[0]) == 'A' and Deck.get_suit(move[0]) != state.get_trump_suit():
                return move

            # Find trump cards
            if move[0] is not None and Deck.get_suit(move[0]) == state.get_trump_suit():
                trump_cards.append(move)
            else:
                other_cards.append(move)
                
            if move[0] == 'J' or move[0] == 'Q':
                if Deck.get_suit(move[0]) != state.get_trump_suit():
                    low_cards.append(move)

        if len(low_cards) > 0:
            return low_cards[-1]

        # If the opponent has played a card
        oppo_card = state.get_opponents_played_card()
        if oppo_card is not None:
            rank = Deck.get_rank(oppo_card)
            moves_same_suit = []
            # Get all moves of the same suit as the opponent's played card
            for move in moves:
                if move[0] is not None and Deck.get_suit(move[0]) == Deck.get_suit(state.get_opponents_played_card()):
                    moves_same_suit.append(move)

            if len(trump_cards) > 0:
                if rank == 'A' or rank == '10':
                    return trump_cards[0]
                else:
                    if len(low_cards) > 0:
                        return low_cards[-1]

                    if len(moves_same_suit) > 0:
                        return moves_same_suit[-1]

            if len(moves_same_suit) > 0:
                if rank == 'A':
                    if len(low_cards) > 0:
                        return low_cards[0]
                else:
                    return moves_same_suit[0]


        return moves[0]