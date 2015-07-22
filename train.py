import sys
from random import randrange
from ale_runner import ALERunner
from ale_python_interface import ALEInterface
from agent import Agent

if len(sys.argv) < 2:
  print 'Usage:', sys.argv[0], 'rom_file'
  sys.exit()

#initialize the agent
# right now it's just a dummy agent
# soon it will be an agent with a simple linear model
# for the value function estimator
# then it will have a convnet as the value function estimator

agent = Agent()


# set some training params for the ALERunner
screenFilter = "foo"
romFile = sys.argv[1]
epochs = 2
epoch_steps = 1000
watch_training = True

#initialize the runner, which marshalls the ALE process
runner = ALERunner(agent, screenFilter, romFile, epochs, epoch_steps,
            watch_training)

#start training
runner.train_agent()

