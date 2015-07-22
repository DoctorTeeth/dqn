"""This class describes the actual agent
It gets called by the experiments
when we have other emulators besides just ALE
"""

import numpy as np
import sys
from random import randrange

# we will have to fill this in alongside the other guy

# the experiment never talks to the memory

# only this agent ever talks to the memory

# and different experiments call this agent

# then the experiment will take an agent

# we want a person to be able to specify their own agent that
# inherits from the base agent class
# for now we can just have the one, but 

# another thing is that the downsampling is part of 
# the environment and not the agent, so we should make sure to 
# keep that in the experiment layer 


class Agent():
    def __init__(self):
        #initialize the state of the agent
        print "initializing agent"

    def step(self, reward, observation, actions):
        #for now, always return the null action
        a = actions[randrange(len(actions))]
        return a


