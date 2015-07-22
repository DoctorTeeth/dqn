import sys
from random import randrange
from ale_runner import ALERunner
from ale_python_interface import ALEInterface

if len(sys.argv) < 2:
  print 'Usage:', sys.argv[0], 'rom_file'
  sys.exit()

agent = "foo"
screenFilter = "foo"
romFile = sys.argv[1]


runner = ALERunner(agent, screenFilter, romFile)

