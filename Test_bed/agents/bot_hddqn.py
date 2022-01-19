"""
    This is the Robot Agent's model for Hierarchical Double Deep Q-Learning
    It inherits from bot_ddqn

    Unlike DDQN, this creates two agents, one intrinsic and one extrinsic.
    It was adapted from this paper by Kulkarni:
        https://arxiv.org/pdf/1604.06057.pdf
"""


from gym.core import Wrapper
import numpy as np
from numpy.core import numeric


if __name__ == '__main__':
    from  utils.plotter import PlotResults
else:
    from .utils.plotter import PlotResults

from agents.bot_ddqn import Bot



# Create a virtual bot and an environment for them to play
# Link the bot to a neural network and track whether or not it's learning as it plays
class Bot(Bot):
    def __init__(self, environment, int_returns, ext_returns, results_filename='Output.txt'):
        
        super().__init__(environment, results_filename)

        # meta goals
        self.metacontroller_goals = self.get_env_goals()

        # get the environment return types
        self.return_types         = self.env.get_return_mode()
        self.int_returns          = int_returns
        self.ext_returns          = ext_returns

        # if either int or ext is true then true for the state returns
        self.env_returns = int_returns.copy()
        for i in range(len(int_returns)):
            if int_returns[i] or ext_returns[i]:
                self.env_returns[i] = True


    # split the observations 
    # -- if the intrinsic and extrinsic agents don't match
    def _split_observations(self, state):
        # only one state type so return it for both 
        # instrinsic and extrinsic
        if self.int_returns==self.ext_returns:
            return state, state

        in_state        = []
        ex_state        = []
        state_increment = 0

        for s in range(len(self.env_returns)):
            if self.env_returns[s]==True:
                offset = 1
                # intrinsic state
                if self.int_returns[s]==True:
                    # if tactile
                    if s==1:
                        in_state.append(state[state_increment])
                        in_state.append(state[state_increment+1])
                        offset=2
                    else:
                        in_state.append(state[state_increment])
                # extrinsic state
                if self.ext_returns[s]==True:
                    # if tactile
                    if s==1:
                        ex_state.append(state[state_increment])
                        ex_state.append(state[state_increment+1])
                        offset=2
                    else:
                        ex_state.append(state[state_increment])   
            
                state_increment+=offset

        # remove the extra dimension if the state space is only length 1
        if len(ex_state)==1:
            ex_state = ex_state[0]
        if len(in_state)==1:
            in_state = in_state[0]

        return ex_state, in_state

    
    # get the initial epsilon value from the intrinsic network and store them in an array
    def __ini_eps_values(self, in_model, test=False):
        meta_eps = []
        for i in range(len(self.metacontroller_goals)):
            if test:
                meta_eps.append(0)
            else:
                meta_eps.append(in_model.get_epsilon())
        
        return meta_eps


    # get the goals from the environment
    def get_env_goals(self):
        return self.env.get_goals()


    # wrap the state and goal
    def wrap_state(self, state, goal):

        wrapped_state   = state.copy()

        # If there is no real or tactile inputs then the information is only arm related
        # and so the goal should be appended directly to the state
        no_real_or_tact = self.int_returns[0]==False and self.int_returns[1]==False      
        if no_real_or_tact:
            wrapped_state        = np.append(wrapped_state, goal)

        # else append to the end     
        else:
            # add to the end
            index                = len(wrapped_state)-1
            wrapped_state[index] = np.append(wrapped_state[index], goal)

        return wrapped_state


    # Train using game model -- Runs the loops for the network and agent tasks
    def train(self, intrinsic_model, extrinsic_model, n_episodes=100000, model0_weights=None, model1_weights=None):
        # instantiate the inital epsilon values
        self.meta_eps = self.__ini_eps_values(intrinsic_model)

        # Load a model if selected
        if model0_weights  is not None:
            intrinsic_model.load_model(model0_weights)
        if model1_weights is not None:
            extrinsic_model.load_model(model1_weights)

        # Setup arrays to dump game information
        ddqn_times      = []
        ddqn_success    = []
        ddqn_rewards    = []

        # Initialise graphing
        plotter = PlotResults()

        # play for i to n+1 games (+1 to ensure the graphs capture the last number)
        for i in range(n_episodes):
            
            # Reset for next run
            state, done         = self.reset()
            actuaction_force    = [0,0,0,0,0,0,0,0]

            # extrinsic and intrinsic states
            ext_state, int_state = self._split_observations(state)

            # pick a goal for this state
            goal            = extrinsic_model.choose_action(ext_state)
            current_epsilon = self.meta_eps[goal]
            intrinsic_model.set_epsilon(current_epsilon)
            
            # run until the episode's end
            while not done:
                # extrinsic agent's reward 
                # initial state of this goal
                # intrinsic agent's reward itialisation
                # has the goal been reached
                extrinstic_reward   = 0        
                ini_state           = ext_state
                reached             = False

                # run until the episode ends or the intrinsic agent reaches its goal
                while (not done and not reached):
                    # wrap the current state with the goal
                    wrapped_state = self.wrap_state(int_state, goal)

                    # Pick from the number of buttons allowed (Discretizer)
                    action = intrinsic_model.choose_action(wrapped_state)

                    # convert the int output to an array
                    # -- actuator force is in both the state and info arrays 
                    #    (easier to get it out of the latter)
                    action_array = self.env.action_handler(action, actuaction_force)

                    # Take the action and record the outputs
                    state_, reward, done, info  = self.env.step(action_array)
                    actuaction_force            = info['actuator_force']
                    ext_state_, int_state_      = self._split_observations(state_)

                    # reward for the intrinsic agent
                    # -- has the current goal been reached
                    # -- what is the associated reward
                    reached, intrinsic_reward = self.env.get_intrinsic_reward(reward, self.metacontroller_goals[goal])
                    
                    # wrap the next state and goal
                    wrapped_state_            = self.wrap_state(int_state_, goal)

                    # Make the agent remember what's done, seen and the associated rewards with those things
                    intrinsic_model.store_memory(wrapped_state, action, intrinsic_reward, wrapped_state_ , int(done))

                    # Make the agents learn from what they now know
                    extrinsic_model.learn(update_eps=False)
                    intrinsic_model.learn(update_eps=False)

                    # append the culmulative reward
                    extrinstic_reward   += reward
                    self.score          += reward

                    # Make the next state the current state
                    ext_state   = ext_state_
                    int_state   = int_state_


                # update the extrinsic agent parameters
                extrinsic_model.store_memory(ini_state, goal, extrinstic_reward, ext_state_, done)
                
                # if we reached the target then decay the epsilon value for this goal
                # if not done, set a new goal and get its associated epsilon value
                if not done and reached:
                    # update the subgoal eps by the average success rate
                    intrinsic_model.update_epsilon()
                    self.meta_eps[goal] = intrinsic_model.get_epsilon()
                    
                    # pick a new goal
                    goal                = extrinsic_model.choose_action(ext_state)
                    current_epsilon     = self.meta_eps[goal]
                    intrinsic_model.set_epsilon(current_epsilon)

            # update the extrinsic model by standard epsilon greedy
            extrinsic_model.update_epsilon()
            eps = extrinsic_model.get_epsilon()

            # append metrics to arrays
            ddqn_success.append(info['success'])
            ddqn_rewards.append(self.score)
            ddqn_times.append(info['step_adj'])

            # Display results in the console
            self.print_results_to_console(i, ddqn_success[i], self.score, ddqn_times[i], eps)
            self.print_results_to_file(i, ddqn_success[i], ddqn_times[i])

            # Save the Model and Graph every 100 episodes
            # Also save after the first 10 - just to make sure the model and outputs are working
            if (i==2) or (i == 10) or (i % 100 == 0 and i > 0):

                h_eps = extrinsic_model.get_epsilon()

                # Save the models
                intrinsic_model.save_model(i, "intrinsic_agent")
                extrinsic_model.save_model(i, "extrinsic_agent")

                # Save the graph of rewards acheived and the time spent in the environment
                x = [i for i in range(i + 1)]
                filename0 = f'outputs/ddqn-output/ddqnRobotArm-TimeScores--Episodes{i}.png'
                plotter.plot_results(x, ddqn_rewards, ddqn_times, filename0, "Rewards", "Times")


    # Test using game model -- Runs the loops for the network and agent tasks
    # Test assumes the epsilon values to be 0 and the model agent to be acting entirely greedily hence:
    # -- there are no learn() method calls
    # -- there are no epsilon_sets() or updates
    def test(self, intrinsic_model, extrinsic_model, model0_weights, model1_weights, n_episodes=100):
        # Load a model if selected
        intrinsic_model.load_model(model0_weights)
        extrinsic_model.load_model(model1_weights)

        # Setup arrays to dump game information
        ddqn_times      = []
        ddqn_success    = []
        ddqn_rewards    = []

        # Initialise graphing
        plotter = PlotResults()

        # play for i to n+1 games (+1 to ensure the graphs capture the last number)
        for i in range(n_episodes):
            
            # Reset for next run
            state, done         = self.reset()
            actuaction_force    = [0,0,0,0,0,0,0,0]

            # extrinsic and intrinsic states
            ext_state, int_state = self._split_observations(state)

            # pick a goal for this state
            goal            = extrinsic_model.choose_action(ext_state)
            
            # run until the episode's end
            while not done:
                # extrinsic agent's reward 
                # initial state of this goal
                # intrinsic agent's reward itialisation
                # has the goal been reached
                extrinstic_reward   = 0        
                ini_state           = ext_state
                reached             = False

                # run until the episode ends or the intrinsic agent reaches its goal
                while (not done and not reached):
                    # wrap the current state with the goal
                    wrapped_state = self.wrap_state(int_state, goal)

                    # Pick from the number of buttons allowed (Discretizer)
                    action = intrinsic_model.choose_action(wrapped_state)

                    # convert the int output to an array
                    # -- actuator force is in both the state and info arrays 
                    #    (easier to get it out of the latter)
                    action_array = self.env.action_handler(action, actuaction_force)

                    # Take the action and record the outputs
                    state_, reward, done, info  = self.env.step(action_array)
                    actuaction_force            = info['actuator_force']
                    ext_state_, int_state_      = self._split_observations(state_)

                    # reward for the intrinsic agent
                    # -- has the current goal been reached
                    # -- what is the associated reward
                    reached, intrinsic_reward = self.env.get_intrinsic_reward(reward, self.metacontroller_goals[goal])
                    
                    # append the culmulative reward
                    extrinstic_reward   += reward
                    self.score          += reward

                    # Make the next state the current state
                    ext_state   = ext_state_
                    int_state   = int_state_


                # update the extrinsic agent parameters
                extrinsic_model.store_memory(ini_state, goal, extrinstic_reward, ext_state_, done)
                
                # if we reached the target then decay the epsilon value for this goal
                # if not done, set a new goal and get its associated epsilon value
                if not done and reached:
                    # update the subgoal eps by the average success rate
                    self.meta_eps[goal] = intrinsic_model.get_epsilon()
                    
                    # pick a new goal
                    goal                = extrinsic_model.choose_action(ext_state)

            # update the extrinsic model by standard epsilon greedy
            eps = extrinsic_model.get_epsilon()

            # append metrics to arrays
            ddqn_success.append(info['success'])
            ddqn_rewards.append(self.score)
            ddqn_times.append(info['step_adj'])

            # Display results in the console
            self.print_results_to_console(i, ddqn_success[i], self.score, ddqn_times[i], eps)
            self.print_results_to_file(i, ddqn_success[i], ddqn_times[i])

            # Save the Model and Graph every 100 episodes
            # Also save after the first 10 - just to make sure the model and outputs are working
            if (i==2) or (i == 10) or (i % 100 == 0 and i > 0):

                h_eps = extrinsic_model.get_epsilon()

                # Save the models
                intrinsic_model.save_model(i, "intrinsic_agent")
                extrinsic_model.save_model(i, "extrinsic_agent")

                # Save the graph of rewards acheived and the time spent in the environment
                x = [i for i in range(i + 1)]
                filename0 = f'outputs/ddqn-output/ddqnRobotArm-TimeScores--Episodes{i}.png'
                plotter.plot_results(x, ddqn_rewards, ddqn_times, filename0, "Rewards", "Times")