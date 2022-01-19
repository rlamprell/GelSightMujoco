"""
    This is the Double Deep Q-Learning Model (DDQN)
    Here two Neural Networks are created one to select actions and the other to retreive Q-values
    DDQN helps to avoid the overestimation bias found in DQN.

    This code is adapted from coursework-2 of COMP532.
"""


import  numpy as np
from    tensorflow.keras.models import load_model


# Double Deep Q-Learning Network - The greedy option means always choose the highest Q value
class ddqn(object):
    def __init__(self, 
                 gamma, 
                 n_actions, 
                 epsilon, 
                 batch_size, 
                 epsilon_decay, 
                 epsilon_minim,
                 eval_neuralnetwork, 
                 targ_neuralnetwork, 
                 replaybuffer,
                 target_update=1000, 
                 model_name=None, 
                 greedy=False,
                 model_output_dir=""
                 ):

        # File name for saving and loading
        self.model_file         = model_name
        self.model_output       = model_output_dir

        # If playing greedily then use the best weights every time
        # In this case the agent should never activate the learn function
        # - however, the minimum is set to 0.0 just in case the user wants to run the learn function
        if greedy:
            self.epsilon        = 0.0
            self.epsilon_minim  = 0.0
        else:
            self.epsilon        = epsilon
            self.epsilon_decay  = epsilon_decay
            self.epsilon_minim  = epsilon_minim

        # How many actions available
        self.action_space       = [i for i in range(n_actions)]
        #elf.n_actions          = n_actions

        # Batch_size to pick from memory
        self.batch_size         = batch_size

        # Discount Factor
        self.gamma              = gamma

        # Load the memory config - Records all the plays from the environment
        self.memory              = replaybuffer

        # Create two DQNs - One to be updated every step, the other every 100
        self.q_evaluation        = eval_neuralnetwork
        self.q_target            = targ_neuralnetwork
       

        # Update rate for the q_target network
        self.target_update      = target_update


    # Store the values of what just happened
    def store_memory(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)


    # Choose the action to perform - Uses the Evaluation network to make greedy decisions
    def choose_action(self, state):
 
        rand        = np.random.random()
        state_copy  = state.copy()

        # Epsilon Greedy
        # -- if epsilon value is lower than the random number then choose randomly
        if rand < self.epsilon:
            action  = np.random.choice(self.action_space)
        # -- else choose the best action from the prediction batch
        else:
            # the number of items stored in the replay buffer
            # -- vision, tactile and/or kinematics
            data_count = self.memory.get_buffer_count()

            # reshape the states so tensorflow will accept them (1, x, x, x)
            # -- if we have multiple obs types stored
            if data_count>1:
                for i in range(data_count):
                    state_copy[i] = state_copy[i][np.newaxis, :]

            # -- if we have only a single obs type stored
            else:
                state_copy  = state_copy[np.newaxis, :]

            actions = self.q_evaluation.predict(state_copy)
            action  = np.argmax(actions)

        return action


    # Make the agent learn
    # Calc Q(s,a) and fit to network
    # Update the target network every set number of steps
    # Decay the epsilon value
    def learn(self, update_eps=True):
        # if the memory count if greater than the batch size then update
        if self.memory.memory_count > self.batch_size:
            # the sample batches from the replaybuffer
            state, action, reward, state_, done = self.memory.sample_buffer(self.batch_size)

            # Make the new q_target our q_predicted value
            q_target = self.q_evaluation.predict(state)

            # Action selected per batch
            action_indices = np.dot(action, np.array(self.action_space, dtype=np.int8))

            # Batch Index (local 0 - 7)
            batch = np.arange(self.batch_size, dtype=np.int32)

            # Best actions for each batch in the next state
            max_actions = np.argmax(self.q_evaluation.predict(state_), axis=1)

            # RHS Q-Value - Predict the action output for each batch based on the next state
            q_ = self.q_target.predict(state_)

            # q_target Q(s,a) - wraps the reward and actions together (*done will remove everything after end of game)
            q_target[batch, action_indices] = reward + self.gamma*q_[batch, max_actions.astype(int)]*done

            # Fit to eval network using q_target - verbose=0 stops keras printing to console
            loss = self.q_evaluation.fit(state, q_target, batch_size=self.batch_size, epochs=1, verbose=0, shuffle=True)#, callbacks=[self.tensorboard_callback])
            #loss    = history[1]

            # Update the target network if we've done 100 steps (update using the eval table)
            # -- arbitary value selected as a init param - 'target_update'
            if self.memory.memory_count % self.target_update == 0:
                self.update_network()

            # Update to the new epsilon value if epsilon has not yet decayed to the lowest allowed value
            if update_eps:
                self.update_epsilon()

            return loss


    # set the epsilon value
    def set_epsilon(self, eps):
        self.epsilon = eps


    # decay the epsilon by a multiple
    def update_epsilon(self):
        if (self.epsilon <= self.epsilon_minim):
            self.epsilon = self.epsilon_minim
        else:
            self.epsilon = self.epsilon * self.epsilon_decay


    # decay the epsilon  by a value
    def update_epsilon_by_value(self, value):
        if (self.epsilon <= self.epsilon_minim):
            self.epsilon = self.epsilon_minim
        else:
            self.epsilon = self.epsilon - value


    # return the current epsilon value
    def get_epsilon(self):
        return self.epsilon


    # Update the Target Network with the Evaluation Network's weights
    def update_network(self):
        self.q_target.set_weights(self.q_evaluation.get_weights())


    # Save the model
    def save_model(self, i, model_name="ddqn_model"):
        self.q_evaluation.save(f'outputs/ddqn-models/{i}-{model_name}.h5')

    # save the replaybuffer
    def save_replaybuffer(self, i):
        self.memory.save_buffer(i)


    # Load the model
    def load_model(self, model_name):
        self.q_evaluation = load_model(model_name)

        # Load the imported weights into the target network when playing greedily
        if self.epsilon == 0.0:
            self.update_network()