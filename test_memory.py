import replay_memory
import numpy as np

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
phiDims = (5,1)

# declare memory of 10 experience tuples
mem = replay_memory.ReplayMemory(10,phiDims,t)

# insert some tuples, but don't fill up the memory
for i in range(0,5):
    phi_before = i
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
