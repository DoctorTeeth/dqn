import replay_memory
import numpy as np
import sys

def sample_and_print(replayMem, count):
     
    print "sample of size: ", count
    befores, afters, actions, rewards, terminal = replayMem.sample_tuples(count)
    print "phi_befores: ", befores
    print "phi_afters: ", afters
    print "actions: ", actions 
    print "rewards: ", rewards 
    print "terminal: ", terminal 
    print

print "testing replay memory"
print

# for this test, the type of the state tensor is uint8
t = np.dtype('uint8')
phiDims = (2,2)

# declare memory of 10 experience tuples
mem = replay_memory.ReplayMemory(10,phiDims,t)

#insert a single tuple, then print it, result should always be the same
mem.insert_tuple(1, 2, 3, 4)
sample_and_print(mem,1)

# insert some tuples, but don't fill up the memory
for i in range(0,4):
    phi_before = i      # this and below does element-wise assign
    phi_after  = i + 1
    action     = 888
    reward     = 3.14159
    mem.insert_tuple(phi_before, action, reward, phi_after)

# sample a random batch from the memory
# check that it is the right size, action values are as expected, etc
sample_and_print(mem,2)

# insert exactly enough tuples to fill up the memory
for i in range(5,10):
    phi_before = i
    phi_after  = i + 1
    action     = 777 
    reward     = 0 
    mem.insert_tuple(phi_before, action, reward, phi_after)

# samples should now come from both 0-5 and 6-10  
sample_and_print(mem,6)

# insert one more tuple to test wrap around
mem.insert_tuple(0,0,0,0)

#if you insert 9 more times, all samples should have 0 for all values
for i in range(0,9):
    mem.insert_tuple(0,0,0,0)

xs = mem.sample_tuples(1)
print "wrap around test: ",
if xs[2] == 0:
    print "PASS"
else:
    print "FAIL"

# test that we can allocate a memory big enough to train the nature model
# this allocates space for 1 million tuples
# where each tuple represents stores 84 by 84 by 4 
# stack of 4 most recent grayscaled images
# so we do have enough memory to just use
bigDims = (84,84,4) 
big_t = np.dtype('uint8')
big = replay_memory.ReplayMemory(1000 * 1000,bigDims,big_t)

print "big memory allocation: PASS"
