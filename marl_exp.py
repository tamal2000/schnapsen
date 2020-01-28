from matplotlib import pyplot as plt
from api import State, util
import random

state = State.generate()
moves = State.moves(state)
rand_move = random.choice(moves)
next_sate = State.next(state,rand_move)

finished = state.finished()
revoked = state.revoked() # When player makes any illigal moves
winner = state.winner() # Works when the game is Finished
hand = state.hand()
clone = state.clone() # Clone the given state
opponent_card = state.get_opponents_played_card() #after the opponent plays a cards
pre_move = next_sate.get_prev_trick()
turn = state.whose_turn()
perspective = state.get_perspective(turn)
leader = next_sate.leader()
point_1 = next_sate.get_points(1)
point_2 = next_sate.get_points(2)
pending_points = next_sate.get_pending_points(turn)
trump_suite = state.get_trump_suit()
stock = next_sate.get_stock_size()
phase = State.get_phase(state)
if phase != 1:
    assumption = next_sate.make_assumption() #not sure yet
    print("Assumption:", assumption)
other = util.other(turn)
suit = util.get_suit(rand_move[0])
rank = util.get_rank(rand_move[0])
card_name = util.get_card_name(rand_move[0])
player = util.load_player('rand')
point_ratio = util.ratio_points(state,1)


#print('Current State:', state)
print('Current Moves:', moves)
print('Random Move:', rand_move)
#print('Next State:\n',next_sate)
print('Game Finished:', finished)
print('Revoked:', revoked)
print('Winner:', winner)
print('Hand:', hand)
#print('Clone:', clone)
print("Opponent Play's cards:", opponent_card)
print("Previous move:", pre_move )
print("Whose turn:", turn)
print("perspective:", perspective)
print("Leader:", leader)
print("player 1 Points:", point_1)
print("player 2 Points:", point_2)
print("Pending poitns:", pending_points)
print("Trump Suit:", trump_suite)
print("Stock Size:", stock)
print("Game in now in Phase:", phase)
print("Other:", other)
print("Suit of the played card", suit)
print('Card Rank', rank)
print("Card Name:", card_name)
print("Loaded bot", player)
print("Ratio", point_ratio)
