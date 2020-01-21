from bots.ml import ml
from api._state import State
from api import engine
import tournament

from argparse import Namespace


player1 = ml.Bot(model_file='./bots/ml/rand.pkl')
player2 = ml.Bot(model_file='./bots/ml/rdeep.pkl')
player3 = ml.Bot(model_file='./bots/ml/trained_on_rdeep.pkl')

def play():
    options = Namespace(fast=True, max_time=5, phase=1, players='ml_rand,ml_rdeep,ml_ml', repeats=1000)
    
    tournament.run_tournament(options)

#     # Generate or load the map
#     state = State.generate(phase=1)

#     # Play the game
#     engine.play(player1, player2, state=state)

play()