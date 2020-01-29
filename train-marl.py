import os
import os.path
import random
import time
from collections import deque
#from bots.marl.marl_asif import Bot
#from bots.marl.marl_asif import features
from itertools import chain

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Activation, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from api import State, util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Only needed if game doesn't work wihtout second player
#from bots.rand import rand
# player=rand.Bot()

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 64
UPDATE_TARGET_EVERY = 5
MODEL_NAME = 'three'
# why do we need mini reward to be defined
MIN_REWARD = -200
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 5000  # 20_000
# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

# For stats
ep_rewards = [-200]


# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Create models folder
if not os.path.isdir('models_test3'):
    os.makedirs('models_test3')


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        #self.writer = tf.summary.FileWriter(self.log_dir)
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        #self._write_logs(stats, self.step)
        pass


# Agent class
class DQNAgent:
    def __init__(self):

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Dense(32, input_shape=(156,)))
        model.add(Activation('relu'))
        model.add(Dense(300))
        model.add(Activation('relu'))
        model.add(Dense(20, activation='softmax'))
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action[0]] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def obtain_move_from_deep_network(self, state):

        # Count total moves made
        #self.total_moves = self.total_moves + 1

        current_state_features = np.array(features(state))
        moves = state.moves()

        qs = self.get_qs(current_state_features)

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
            #self.moves_from_nn = (self.moves_from_nn + 1)

        # print('move ratio:', f'{self.moves_from_nn}/{self.total_moves}')

        return action


agent = DQNAgent()

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode
    agent.tensorboard.step = episode
    #print("Episode number", episode)

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = State.generate(phase=1)

    # Reset flag and start iterating until episode ends
    counter = 0
    done = False
    # while not done:
    while not current_state.finished():

        current_state_features = np.array(features(current_state))

        moves = current_state.moves()
        if np.random.random() > epsilon:
            action = agent.obtain_move_from_deep_network(current_state)
        else:
            action = random.choice(moves)

        ############### find a way to insert move ##########
        new_state = current_state.next(action)
        winner, score = new_state.winner()
        done = current_state.finished()
        reward = 0

        if winner == 1:
            reward += score
        elif winner == 2:
            reward -= score

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        new_state_features = np.array(features(new_state))

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state_features, action, reward, new_state_features, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

        #print('episode_reward', episode_reward)
        #print('step', step)

    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only whenb9632890- min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models_test/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
