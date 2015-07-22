"""
This class trains an arbitrary agent using ALE.
It also provides functions for testing the agent.
"""
import numpy as np
from ale_python_interface import ALEInterface
import logging
import sys

# configure the logger to print all messages
# we could write this to a file if we wanted
logging.basicConfig(level=logging.DEBUG)

class ALERunner(object):
    """screenFilter is a function that takes in the full screen as input
        and turns that into what we actually use in the agent
        we will need to pass this in when we instantiate the ALERunner
    """
    #TODO: describe the arg list in more detail
    
    def __init__(self, agent, screenFilter, romFile, epochs, epoch_steps,
                    watch_training):
        self.ale = ALEInterface()

        #need to start watching before ROM is loaded
        if watch_training:
            self.start_watching() 

        # we need to set this up before we can get the action set
        self.ale.setInt('random_seed', 123)
        self.ale.loadROM(romFile)

        self.agent = agent
        self.screenFilter = screenFilter
        self.actions = self.ale.getMinimalActionSet()
        self.screenWidth, self.screenHeight = self.ale.getScreenDims()
        self.screenBuffer = np.empty((self.screenHeight, self.screenWidth, 3),
                                 dtype=np.uint8)

        self.epoch_steps = epoch_steps
        self.epochs = epochs


        #TODO:
        # maybe allow for people to watch the whole training run?

        # TODO 
        # there ought to be somewhere in this file where we call ALEresetGame

    def train_agent(self):
        """Run a full training cycle, which is broken into training epochs
        and optional testing epochs
        """

        logging.info("Starting training cycle.")
        #print "starting training cycle"
        for i in range(0,self.epochs):
            # sprague has an agent finish the epoch separately
            # I want that to happen inside these calls
            self.run_epoch(i)

    def run_epoch(self, epoch_counter):
        """an epoch is actually defined by a discrete number of timesteps
        """

        logging.info("Running epoch: " + str(epoch_counter))
        steps_left = self.epoch_steps
        while steps_left > 0:
            steps_taken = self.run_game(steps_left)
            steps_left -= steps_taken 
            
    def run_game(self, steps_left):
        """ Run a single game. A single game may consist of multiple lives
            I say game instead of episode, because I think it's more natural
        """
        
        # keep track of steps taken in this game
        steps_taken = 0
        
        # we set the initial action, but it can be null for now
        action = 0 
        while steps_taken < steps_left and not self.ale.game_over():
           
            # take an action in the emulator
            # TODO: take a random action instead of the first action
            reward = self.ale.act(action)

            # TODO: what do we do about stacking frames
            # generate the representation of the environment state
            # (the observation) that the agent will see 
            # TODO: actually generate this observation
            observation = 0
             

            # an agent generates the next action based on the reward,
            # observation pair
            action = self.agent.step(reward, observation,self.actions)
            logging.debug("Agent chooses action: " + str(action))

            steps_taken += 1

        return steps_taken
         

    def get_image(self):
        """ lifted straight from deep_q_rl
            get a screencap from ALE and scale it down
        """
        #TODO: it's not clear that we need this function
        # rather, it might be better to have a more general one
        
        self.ale.getScreenRGB(self.screenBuffer)

        return self.screenFilter(self.screenBuffer)

    def start_watching(self):
        
        logging.info("Starting up game watcher.")        

        if sys.platform == 'darwin':
            print "darwin"
            import pygame
            pygame.init()
            self.ale.setBool('sound', False) # Sound doesn't work on OSX
        elif sys.platform.startswith('linux'):
            print "linux"
            self.ale.setBool('sound', True)
        self.ale.setBool('display_screen', True)
