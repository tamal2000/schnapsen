from bots.ml import ml
from api._state import State
from api import engine
import tournament

from argparse import Namespace



def play():
    options = Namespace(fast=True, max_time=5, phase=1, players='ml_rand,ml_rdeep,ml_ml,ml_additional_features', repeats=250)
    tournament.run_tournament(options)
play()