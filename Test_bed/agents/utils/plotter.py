"""
    Plot the outputs from the training and testing of an Agent.

    Similar to DDQN_Model, this is a modified version of code submitted 
    for assignment-2 of COMP532.
"""


import numpy            as np
import matplotlib.pylab as plt


# This class constructs tables from the score and time results of agent/bot DDQN model
class PlotResults:
    def __init__(self):
        # Colours
        self.col1, self.col2 = 'k', 'r'


    # plot the results of the agent
    def plot_results(self, x, scores, times, filename, title0="Rewards", title1="Times", lines=None):

        # Create a figure
        fig     = plt.figure()

        # Add two subplots
        axis1_  = fig.add_subplot(111, label="1")
        axis2_  = fig.add_subplot(111, label="2", frame_on=False)

        # How many timestamps (plots across the x axis)
        n       = len(times)

        # Empty arrays for time and scores
        avg_times   = np.empty(n)
        avg_scores  = np.empty(n)

        # Average across 20 episodes for each (first 19 steps will be over n)
        for t in range(n):
            avg_times[t]    = np.mean(times[max(0, t-100):(t+1)])
            avg_scores[t]   = np.mean(scores[max(0, t-100):(t+1)])

        # Format and draw Axis 1
        axis1_.plot(x, avg_times, color=self.col1)
        axis1_.set_xlabel("Episode", color=self.col1)
        axis1_.set_ylabel(title1, color=self.col1)
        axis1_.tick_params(axis='x', colors=self.col1)
        axis1_.tick_params(axis='y', colors=self.col1)

        # Format and draw Axis 2
        axis2_.axes.get_xaxis().set_visible(False)
        axis2_.yaxis.tick_right()
        axis2_.set_ylabel(title0, color=self.col2)
        axis2_.yaxis.set_label_position('right')
        axis2_.tick_params(axis='y', colors=self.col2)
        axis2_.plot(x, avg_scores, color=self.col2)

        # Remove the lines
        if lines is not None:
            for line in lines:
                plt.axvline(x=line)

        # Save the figure
        plt.savefig(filename, bbox_inches="tight")

        # Need to explicitly close or it won't drop the memory
        plt.close()