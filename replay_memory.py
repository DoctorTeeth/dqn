"""This module defines classes involving the replay memory.
    It will mostly have to be tossed when we move to distributed training
    It supports the following operations:
        insert_tuple:
            inserts a tuple (phi_t, action, reward, phi_{t+1})
            phi represents the state before and after the action is taken
            phi will be stored in pre-processed mode, so that tuples
            can be taken directly from the memory and used to train
            therefore, you will have to pre-process states before inserting them. 
            The upshot is that, as long as you represent your actions
            as a vector indexable by ints
            and as long as your reward in R^1, this code makes no assumptions about how
            you will implement the rest of the DQN.
            action is an int, and reward is some sort of float
        
        sample_tuples:
            grabs some number of tuples from the replay memory
            ***optionally, delete tuples we've already trained on***
            a more sophisticated replay memory will pose some malloc-esque problems
            in terms of memory fragmentation etc,
            but for now we just have this simple memory
        
"""

import numpy as np

class ReplayMemory(object):
    """This class holds the actual physical replay memory
    """

    def __init__(self, size, phiDims, phiType):
        self.size = size

        # the size of the phi tensor is:
        # <Number of samples> * <dims of an individual sample>
        dims    = (size,) + (phiDims)
        self.phiType = phiType
        self.phiDims = phiDims

        # TODO: what do we do about inserting terminal states?
        # so that they can be used to generate a random sample
        self.phi_befores    = np.zeros(dims, dtype=self.phiType)
        self.phi_afters     = np.zeros(dims, dtype=self.phiType)
        self.actions        = np.zeros(self.size, dtype='int16')
        self.rewards        = np.zeros(self.size, dtype='float')
        self.terminal       = np.zeros(self.size, dtype='bool')

        # currently, no tuples live in the replay memory
        self.occupants = 0

        # currently, we are writing tuples at the beginning of replay memory
        self.index = 0


    def insert_tuple(self, phi_before, action, reward, phi_after):
        self.phi_befores[self.index, ...] = phi_before 
        self.phi_afters[self.index, ...]  = phi_after
        self.actions[self.index]      = action 
        self.rewards[self.index]      = reward 

        # if we haven't yet filled up the memory,
        # increment the occupant count
        if self.occupants < self.size:
            self.occupants += 1
            
        # always increment the index
        self.index += 1
                 
        # if the index has been incremented past the end of the memory,
        # wrap back around to the front
        if self.index == self.size:
            self.index = 0


    def sample_tuples(self,count):
        # set aside memory for the sample        

        dims = (count,) + self.phiDims        

        phi_befores    = np.zeros(dims, dtype=self.phiType)
        phi_afters     = np.zeros(dims, dtype=self.phiType)
        actions        = np.zeros(count, dtype='int16')
        rewards        = np.zeros(count, dtype='float')
        terminal       = np.zeros(count, dtype='bool')

        # write the sample into memory
        
        # start writing at the beginning
        sample_index = 0
        
        while sample_index < count:

            # choose a random index between the first elt of replay
            # memory and the last elt of replay memory
            choice = np.random.randint(0, self.occupants)            
            #TODO: quasi-random sampling might improve training results
            
            phi_befores[sample_index] = self.phi_befores[choice]
            phi_afters[sample_index]  = self.phi_afters[choice]
            actions[sample_index]     = self.actions[choice]
            rewards[sample_index]     = self.rewards[choice]
            terminal[sample_index]    = self.terminal[choice]
            
            sample_index += 1 

        # We give these back as a tuple (is there a nicer way?)
        return phi_befores, phi_afters, actions, rewards, terminal
