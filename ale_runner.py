"""
This class trains an arbitrary agent using ALE.
It also provides functions for testing the agent.
"""
import numpy as np
from ale_python_interface import ALEInterface

class ALERunner(object):
    """screenFilter is a function that takes in the full screen as input
        and turns that into what we actually use in the agent
        we will need to pass this in when we instantiate the ALERunner
    """
    #TODO: describe the arg list in more detail
    
    def __init__(self, agent, screenFilter, romFile):
        self.ale = ALEInterface()

        # we need to set this up before we can get the action set
        self.ale.setInt('random_seed', 123)
        self.ale.loadROM(romFile)

        self.agent = agent
        self.screenFilter = screenFilter
        self.actions = self.ale.getMinimalActionSet()
        self.screenWidth, self.screenHeight = self.ale.getScreenDims()
        self.screenBuffer = np.empty((self.screenHeight, self.screenWidth, 3),
                                 dtype=np.uint8)


        #TODO:
        # maybe allow for people to watch the whole training run?


        # TODO
        # should this include all of the loading of the ROM etc?
        # TODO 
        # there ought to be somewhere in this file where we call ALEresetGame

    def train_agent(self):
        """Run a full training cycle, which is broken into training epochs
        and optional testing epochs
        """

        for epoch in range(0,self.num_epochs):
            # sprague has an agent finish the epoch separately
            # I want that to happen inside these calls
            self.run_training_epoch()

            self.run_testing_epoch()

    def run_epoch():
        """an epoch is actually defined by a discrete number of timesteps
        """

    def run_game():
        """ I do episode -> game, because that makes way more sense
        """

    def get_image():
        """ lifted straight from deep_q_rl
            get a screencap from ALE and scale it down
        """
        
        self.ale.getScreenRGB(self.screenBuffer)

        return self.screenFilter(self.screenBuffer)

