"""
    This is the Robot Agent's model for Double Deep Q-Learning
    It creates the agent, who interacts with the environment and tracks all their results
    (This is a modified version of the COMP532's Assignment-2 submission)

    This also acts as a parent to bot_hddqn.
"""


import os


if __name__ == '__main__':
    from  utils.plotter import PlotResults
else:
    from .utils.plotter import PlotResults




# Create a virtual bot and an environment for them to play
# Link the bot to a neural network and track whether or not it's learning as it plays
class Bot:
    def __init__(self, environment, results_filename='Output.txt'):
        # Game name - This can also contain extra arguments, such as which stage to start on
        self.env                 = environment

        # Are we logging the q values if so Log the outputs
        self.results_filename    = results_filename

        # create output dirs
        self.__create_output_dirs()


    # create all the directories required for outputs
    def __create_output_dirs(self):
        # create the outputs folders for results and model weights
        ddqn_output     = os.path.join('./outputs/ddqn-output')
        ddqn_models     = os.path.join('./outputs/ddqn-models')

        # Make sure there is a folder for outputs, models, videos and a file for logging the outputs
        if not os.path.exists(ddqn_output):   os.makedirs(ddqn_output)
        if not os.path.exists(ddqn_models):   os.makedirs(ddqn_models)


    # Reset for next episode
    def reset(self):
        # Score and Time are not seen by the agent, but used instead of graphing the performance of the agent
        self.score  = 0
        self.time   = 0

        # initial 'done' and state
        done        = False
        
        # initial state observation
        state, _    = self.env.reset()

        return state, done
    

    # Output the results to the console
    def print_results_to_console(self, i, done, score, time, eps):

        print(  'Episode: ',            i,
                '   Successful:',       done,
                '   Score: %.2f'%       score,
                '   Time:',             time,
                '   epsilon:',          eps,
        )


    # Output to a file
    def print_results_to_file(self, i, success, time):
        # Write to the output
        #dir_path        = os.path.dirname(os.path.realpath(__file__))
        #output          = os.path.join(dir_path, './outputs/', self.results_filename)
        output          = os.path.join('./outputs/', self.results_filename)

        with open(output, "a") as myfile:
            myfile.write(f'Episode {i}     Success {success}      Score {self.score}     Time {time}\n' )


    # Train using game model -- Runs the loops for the network and agent tasks
    def train(self, deep_model, n_episodes=100000, model_weights=None):

        # Load a model if selected
        if model_weights is not None:
            deep_model.load_model(model_weights)

        # Setup arrays to dump game information
        ddqn_times      = []
        ddqn_success    = []
        ddqn_rewards    = []

        # Initialise graphing
        plotter = PlotResults()

        # play for i to n+1 episodes (+1 to ensure the graphs capture the last number)
        for i in range(n_episodes):
            
            # Reset for next run
            state, done      = self.reset()
            actuaction_force = [0,0,0,0,0,0,0,0]

            # Run until the agent dies or loses
            while not done:
                # Pick from the number of buttons allowed (Discretizer)
                action = deep_model.choose_action(state)

                # convert the int output to an array
                # -- actuator force is in both the state and info arrays 
                #    (easier to get it out of the latter)
                action_array = self.env.action_handler(action, actuaction_force)

                # Take the action and record the outputs
                state_, reward, done, info = self.env.step(action_array)

                # current forces applied to the robotic arm
                actuaction_force = info['actuator_force']

                # add this eps reward to the total
                self.score += reward

                # Make the agent remember what's done, seen and the associated rewards with those things
                deep_model.store_memory(state, action, reward, state_, int(done))

                # Make the agent learn from what it now knows
                deep_model.learn()

                # Make the next state the current state
                state = state_

            # append metrics to arrays
            ddqn_success.append(info['success'])
            ddqn_rewards.append(self.score)
            ddqn_times.append(info['step_adj'])

            # Display results in the console
            self.print_results_to_console(i, ddqn_success[i], self.score, ddqn_times[i], deep_model.get_epsilon())
            self.print_results_to_file(i, ddqn_success[i], ddqn_times[i])

            # Save the Model and Graph every 100 episodes
            # Also save after the first 10 - just to make sure the model and outputs are working
            if (i==2) or (i == 10) or (i % 100 == 0 and i > 0):
                # Save the model
                deep_model.save_model(i)

                # Save the graph of rewards acheived and the time spent in the environment
                x = [i for i in range(i + 1)]
                filename0 = f'./outputs/ddqn-output/RobotArm-TimeScores--Episodes{i}.png'
                plotter.plot_results(x, ddqn_rewards, ddqn_times, filename0, "Rewards", "Times")


    # Test running game model - loading a model
    def test(self, deep_model, model_name, n_episodes=10):

        # Initialise graphing
        plotter = PlotResults()

        # Load the model
        deep_model.load_model(model_name)

        # Setup arrays to dump game information
        ddqn_times      = []
        ddqn_success    = []
        ddqn_rewards    = []

        # Loop for the No. of runs selected
        for i in range(n_episodes):
            
            # Reset for next run
            state, done      = self.reset()
            actuaction_force = [0,0,0,0,0,0,0,0]

            # Run until the agent dies or loses
            while not done:
                # Pick from the number of buttons allowed (Discretizer)
                action = deep_model.choose_action(state)

                # convert the int output to an array
                # -- actuator force is in both the state and info arrays 
                #    (easier to get it out of the latter)
                action_array = self.env.action_handler(action, actuaction_force)

                # Take the action and record the outputs
                state_, reward, done, info = self.env.step(action_array)

                # current forces applied to the robotic arm
                actuaction_force = info['actuator_force']

                # add this eps reward to the total
                self.score += reward

                # Make the next state the current state
                state = state_


            # append metrics to arrays
            ddqn_success.append(info['success'])
            ddqn_rewards.append(self.score)
            ddqn_times.append(info['step_adj'])

            # Display results in the console
            self.print_results_to_console(i, ddqn_success[i], self.score, ddqn_times[i], eps=0)

            # Save the Model and Graph every 100 episodes
            # Also save after the first 10 - just to make sure the model and outputs are working
            if (i==2) or (i == 10) or (i % 100 == 0 and i > 0):
                # Save the model
                deep_model.save_model(i)
                #deep_model.save_replaybuffer(i)

                # Save the graph of rewards acheived and the time spent in the environment
                x = [i for i in range(i + 1)]
                filename0 = f'outputs/ddqn-output/RobotArm-TimeScores--Testing--Episodes{i}.png'
                plotter.plot_results(x, ddqn_rewards, ddqn_times, filename0, "Rewards", "Times")