"""
    Replaybuffer used for storing the agent's interactions with the environment
    Stores:
        -- State
        -- Action
        -- Reward
        -- Next State (State_)
        -- Terminal (boolean)

    Samples of the replaybuffer can be returned to the DDQN model when required.
"""


import numpy as np


# Create a Replay Buffer to store  all the states, actions and rewards for all the plays the agent will make
# -- nn_type: [vision, tactile, kinematics, raw]
class ReplayBuffer(object):
    def __init__(   self,
                    n_actions, 
                    vision_dims = None,
                    tact_dims   = None,
                    num_inp     = 0,
                    nn_type     = [True, False, False, False],
                    max_size    = 10000,
                    goal        = 0
                    ):
        
        self.memory_size            = max_size
        self.memory_count           = 0
        self.nn_type                = nn_type
        self.nn_count               = np.count_nonzero(self.nn_type)
        self.goal                   = goal

        # kinematics and raw are wrapped into the same dimension
        # -- so don't count both
        # -- also, tactile is len 2 (two gelsight)
        if nn_type[2] and nn_type[3]:
            self.nn_count -= 1
        if nn_type[1]:
            self.nn_count +=1

        # get the state memories we are storing
        self.state_memory           = self.__setup_memory_state(vision_dims, tact_dims, num_inp)

        # make a copy of the state space
        self.next_state_memory      = self.state_memory.copy()

        # Create blank arrays for storing the actions, rewards and terminal memories
        self.action_memory          = np.zeros((self.memory_size, n_actions), dtype=np.int8)
        self.reward_memory          = np.zeros(self.memory_size)
        self.terminal_memory        = np.zeros(self.memory_size, dtype=np.float32)


    # setup the state memories to be stored
    def __setup_memory_state(self, vision_dims, tact_dims, num_inp):
        state_memory = []

        # Create arrays to store states, next states, actions, rewards and the terminal memorys
        # -- vision
        if self.nn_type[0]:
            state_memory.append(np.zeros((self.memory_size, vision_dims[0], vision_dims[1], vision_dims[2])))

        # -- tactile
        if self.nn_type[1]:
            state_memory.append(np.zeros((self.memory_size, tact_dims[0], tact_dims[1], tact_dims[2])))
            state_memory.append(np.zeros((self.memory_size, tact_dims[0], tact_dims[1], tact_dims[2])))

        # -- arm inputs
        if self.nn_type[2] or self.nn_type[3]:
            state_memory.append(np.zeros((self.memory_size, num_inp)))

        if len(state_memory)==1:
            return state_memory[0]
        else:
            return state_memory


    # return the type we are storing
    def get_buffer_type(self):
        return self.nn_
    

    # return the number of observations in play
    def get_buffer_count(self):
        return self.nn_count


    # Store the transition to be used later later in the sample (pulled out in batch size selected)
    def store_transition(self, state, action, reward, state_, done):
        # which memory index should we overwrite
        i = self.memory_count % self.memory_size

        # multi input types
        if self.nn_count>1:
            for input in range(len(self.state_memory)):
                self.state_memory[input][i]      = state[input]
                self.next_state_memory[input][i] = state_[input]
        else:
            self.state_memory[i]                 = state
            self.next_state_memory[i]            = state_
        

        # store one hot encoding of actions
        actions                     = np.zeros(self.action_memory.shape[1])
        actions[action]             = 1.0
        self.action_memory[i]       = actions

        # Store rewards at default values
        self.reward_memory[i]       = reward

        # Flag the terminal states as 0 - these will be ignored when applying the 'learn' function
        self.terminal_memory[i]     = 1 - done

        self.memory_count          += 1


    # Pull out random random samples
    # -- outputs: [S, A, R, S_, T]
    def sample_buffer(self, batch_size):
        max_mem = min(self.memory_count, self.memory_size)

        # Choose a random batch from the memory (of size batch_size)
        batch       = np.random.choice(max_mem, batch_size)
        
        # multi input
        if self.nn_count>1:
            states  = []
            states_ = []
            for i in range(len(self.state_memory)):
                states.append(self.state_memory[i][batch]) 
                states_.append(self.next_state_memory[i][batch])
        else:
            states  = self.state_memory[batch]
            states_ = self.next_state_memory[batch]


        # actions rewards and terminal identifier
        actions     = self.action_memory[batch]
        rewards     = self.reward_memory[batch]
        terminal    = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal