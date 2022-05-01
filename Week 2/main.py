import numpy as np
import numpy.random as rand

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib import colors
from src.sarsa import SARSA
from src.gridworld import Gridworld
#TODO:
# Avoid block in start position


if __name__ == '__main__':
    sarsa = SARSA(1,(5,5),5)

    sarsa.go()



