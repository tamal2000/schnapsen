#!/usr/bin/env python

import os
import random
import time
from collections import deque
from itertools import chain

import numpy as np
from sklearn.externals import joblib
from tensorflow.keras.models import load_model

from api import Deck, State, util

DIR_NAME = 'two_____3.00max____1.86avg____1.00min__1580290549.model'
MODEL_FILE = f'models_test/{DIR_NAME}'


# DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000
# MIN_REPLAY_MEMORY_SIZE = 1_000
# MINIBATCH_SIZE = 64
# UPDATE_TARGET_EVERY = 5
# MODEL_NAME = 'two'
# # why do we need mini reward to be defined
# MIN_REWARD = -200
# MEMORY_FRACTION = 0.20

# # Environment settings
# EPISODES = 1000 #20_000
# # Exploration settings
# epsilon = 1 # not a constant, going to be decayed
# EPSILON_DECAY = 0.99975
# MIN_EPSILON = 0.001

# #  Stats settings
# AGGREGATE_STATS_EVERY = 50  # episodes
# SHOW_PREVIEW = False

# # For stats
# ep_rewards = [-200]

# # For more repetitive results
# random.seed(1)
# np.random.seed(1)
# tf.random.set_seed(1)

class DQNAgent:

    def __init__(self, model_path=MODEL_FILE):

        # Main model
        self.model = load_model(model_path)

        # Target network
        self.target_model = load_model(model_path)
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

class Bot:

    def __init__(self, model_file=MODEL_FILE):
        self.agent = DQNAgent()
        self.moves_from_nn = 0
        self.total_moves = 0
        
    def get_move(self, state):
        return self.obtain_move_from_deep_network(state)

    def obtain_move_from_deep_network(self, state):
        # Count total moves made
        self.total_moves = self.total_moves + 1

        current_state_features = np.array(features(state))
        moves = state.moves()
    
        qs = self.agent.get_qs(current_state_features)

        valid_q_indices = []
        for move in moves:
            card_index = move[0]
            valid_q_indices.append(card_index)

        highest_valid_q = 0
        card_index_of_best_move = None

        for index, q_value in enumerate(qs):
            if index in valid_q_indices:  # if move is valid
                if q_value > highest_valid_q:
                    highest_valid_q = q_value
                    card_index_of_best_move = index

        # print('qs:', qs)        
        # print('moves:', moves)        
        # print('valid_q_indices:', valid_q_indices)        
        # print('index_of_best_move:', index_of_best_move)  

        if card_index_of_best_move is None:  # No move has Q value > 0
            action = random.choice(moves)
        else:
            # Obtain the move where the card index matches the best move
            action = [move for move in moves if move[0] == card_index_of_best_move][0]
            self.moves_from_nn = (self.moves_from_nn + 1)
              
        # print('move ratio:', f'{self.moves_from_nn}/{self.total_moves}')
    
        return action
        
# Len of feature vector = 167
def features(state):
    # type: (State) -> tuple[float, ...]
    """
    Extract features from this state. Remember that every feature vector returned should have the same length.

    :param state: A state to be converted to a feature vector
    :return: A tuple of floats: a feature vector representing this state.
    """

    feature_set = []

    # Add player 1's points to feature set
    p1_points = state.get_points(1)
    # Add player 2's points to feature set
    p2_points = state.get_points(2)
    # Add player 1's pending points to feature set
    p1_pending_points = state.get_pending_points(1)
    # Add plauer 2's pending points to feature set
    p2_pending_points = state.get_pending_points(2)
    # Get trump suit
    trump_suit = State.get_trump_suit(state)
    # Add phase to feature set
    phase = state.get_phase()
    # Add stock size to feature set
    stock_size = state.get_stock_size()
    # Add leader to feature set
    leader = state.leader()
    # Add whose turn it is to feature set
    whose_turn = state.whose_turn()
    # Add opponent's played card to feature set
    opponents_played_card = state.get_opponents_played_card()

    ############################# CUSTOM #################################

    # Add hand to feature set
    hand = state.hand()

    # Number of trump cards in hand
    number_of_trump_cards_in_hand = 0
    for h in hand:
        if Deck.get_suit(h) == trump_suit:
            number_of_trump_cards_in_hand = number_of_trump_cards_in_hand + 1

    # Number of suit-following cards in hand
    number_of_suit_following_cards_in_hand = 0
    for h in hand:
        if state.get_opponents_played_card():  # if the opponent has played first
            suit_of_played_card = Deck.get_suit(state.get_opponents_played_card())
            if Deck.get_suit(h) == suit_of_played_card:
                number_of_suit_following_cards_in_hand = \
                    number_of_suit_following_cards_in_hand + 1

    ################## You do not need to do anything below this line ########################

    perspective = state.get_perspective()

    # Perform one-hot encoding on the perspective.
    # Learn more about one-hot here: https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
    perspective = [card if card != 'U'   else [1, 0, 0, 0, 0, 0] for card in perspective]
    perspective = [card if card != 'S'   else [0, 1, 0, 0, 0, 0] for card in perspective]
    perspective = [card if card != 'P1H' else [0, 0, 1, 0, 0, 0] for card in perspective]
    perspective = [card if card != 'P2H' else [0, 0, 0, 1, 0, 0] for card in perspective]
    perspective = [card if card != 'P1W' else [0, 0, 0, 0, 1, 0] for card in perspective]
    perspective = [card if card != 'P2W' else [0, 0, 0, 0, 0, 1] for card in perspective]
    # Append one-hot encoded perspective to feature_set
    feature_set += list(chain(*perspective))
    # Append normalized points to feature_set
    total_points = p1_points + p2_points
    feature_set.append(p1_points/total_points if total_points > 0 else 0.)
    feature_set.append(p2_points/total_points if total_points > 0 else 0.)
    # Append normalized pending points to feature_set
    total_pending_points = p1_pending_points + p2_pending_points
    feature_set.append(p1_pending_points/total_pending_points if total_pending_points > 0 else 0.)
    feature_set.append(p2_pending_points/total_pending_points if total_pending_points > 0 else 0.)
    # Convert trump suit to id and add to feature set
    # You don't need to add anything to this part
    suits = ["C", "D", "H", "S"]
    trump_suit_onehot = [0, 0, 0, 0]
    trump_suit_onehot[suits.index(trump_suit)] = 1
    feature_set += trump_suit_onehot
    # Append one-hot encoded phase to feature set
    feature_set += [1, 0] if phase == 1 else [0, 1]
    # Append normalized stock size to feature set
    feature_set.append(stock_size/10)
    # Append one-hot encoded leader to feature set
    feature_set += [1, 0] if leader == 1 else [0, 1]
    # Append one-hot encoded whose_turn to feature set
    feature_set += [1, 0] if whose_turn == 1 else [0, 1]
    # Append one-hot encoded opponent's card to feature set
    opponents_played_card_onehot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    opponents_played_card_onehot[opponents_played_card if opponents_played_card is not None else 20] = 1
    feature_set += opponents_played_card_onehot

    ############################# CUSTOM #################################

    # Append one-hot encoded number of trump suits in hand to feature set
    trumps_in_hand_onehot = [0 for x in range(6)]
    trumps_in_hand_onehot[number_of_trump_cards_in_hand] = 1
    feature_set += trumps_in_hand_onehot

    # Append one-hot encoded number of suit following cards in hand to
    # feature set (max = 4, vector/list len is 5)
    suit_following_in_hand_onehot = [0 for x in range(5)]
    suit_following_in_hand_onehot[number_of_suit_following_cards_in_hand] = 1
    feature_set += suit_following_in_hand_onehot

    # Return feature set
    return feature_set
