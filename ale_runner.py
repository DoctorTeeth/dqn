"""
This class trains an arbitrary agent using ALE.
It also provides functions for testing the agent.
"""
import sys
from random import randrange
from ale_python_interface import ALEInterface

class ALERunner(object):
    """screenFilter is a function that takes in the full screen as input
        and turns that into what we actually use in the agent
        we will need to pass this in when we instantiate the ALERunner
    """

    def __init__(self, agent, screenFilter):
        self.ale = ALEInterface()
        self.agent = agent
        self.screenFilter = screenFilter
        self.actions = ale.getMinimalActionSet()
        self.screenWidth, self.screenHeight = ale.getScreenDims()
        self.screenBuffer = np.empty((self.screenHeight, self.screenWidth, 3),
                                 dtype=np.uint8)

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



if len(sys.argv) < 2:
  print 'Usage:', sys.argv[0], 'rom_file'
  sys.exit()

ale = ALEInterface()

# Get & Set the desired settings
ale.setInt('random_seed', 123)

# Set USE_SDL to true to display the screen. ALE must be compiled
# with SDL enabled for this to work. On OSX, pygame init is used to
# proxy-call SDL_main.
USE_SDL = False
if USE_SDL:
  if sys.platform == 'darwin':
    import pygame
    pygame.init()
    ale.setBool('sound', False) # Sound doesn't work on OSX
  elif sys.platform.startswith('linux'):
    ale.setBool('sound', True)
  ale.setBool('display_screen', True)

# Load the ROM file
ale.loadROM(sys.argv[1])

# Get the list of legal actions
legal_actions = ale.getLegalActionSet()

# Play 10 episodes
for episode in xrange(10):
  total_reward = 0
  while not ale.game_over():
    a = legal_actions[randrange(len(legal_actions))]
    # Apply an action and get the resulting reward
    reward = ale.act(a);
    total_reward += reward
  print 'Episode', episode, 'ended with score:', total_reward
  ale.reset_game()
