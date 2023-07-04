import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import get_logger
from utils.test_env import EnvTest
from q3_schedule import LinearExploration, LinearSchedule
from q4_linear_torch import Linear
import logging


from configs.q5_nature import config


class NatureQN(Linear):
    """
    Implementing DQN that will solve MinAtar's environments.

    Model configuration can be found in the assignment PDF, section 4a.
    """

    def initialize_models(self):
        """Creates the 2 separate networks (Q network and Target network). The input
        to these models will be an img_height * img_width image
        with channels = n_channels * self.config.state_history

        1. Set self.q_network to be a model with num_actions as the output size
        2. Set self.target_network to be the same configuration self.q_network but initialized from scratch
        3. What is the input size of the model?

        Use the following architecture:
        • One convolution layer with 16 output channels, a kernel size of 3, stride 1, and no padding.
        • A ReLU activation.
        • A dense layer with 128 hidden units.
        • Another ReLU activation.
        • The final output layer.

        Hints:
            1. Simply setting self.target_network = self.q_network is incorrect.
            2. The following functions might be useful
                - nn.Sequential
                - nn.Conv2d
                - nn.ReLU
                - nn.Flatten
                - nn.Linear
            3. To calculate the size of the input to the first linear layer, you
               can use online tools that calculate the output size of a
               convolutional layer (e.g. https://madebyollin.github.io/convnet-calculator/)
        """
        state_shape = self.env.state_shape()
        img_height, img_width, n_channels = state_shape
        input_channels = n_channels * self.config.state_history
        num_actions = self.env.num_actions()

        # calculate the output shape after the Conv2d layer
        output_height = img_height - 3 + 1
        output_width = img_width - 3 + 1
        conv_output_size = output_height * output_width * 16

        # Define the model architecture
        self.q_network = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

        # Initialize Q Network and Target Network
        self.target_network = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def get_q_values(self, state, network):
        """
        Returns Q values for all actions

        Args:
            state: (torch tensor)
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            network: (str)
                The name of the network, either "q_network" or "target_network"

        Returns:
            out: (torch tensor) of shape = (batch_size, num_actions)

        Hint:
            1. What are the input shapes to the network as compared to the "state" argument?
            2. You can forward a tensor through a network by simply calling it (i.e. network(tensor))
        """
        out = None
        network_dic = {"q_network": self.q_network, "target_network": self.target_network}
        out = network_dic[network](state.permute(0, 3, 1, 2))
        return out


"""
Use deep Q network for test environment.
"""
if __name__ == "__main__":
    logging.getLogger(
        "matplotlib.font_manager"
    ).disabled = True  # disable font manager warnings
    env = EnvTest((8, 8, 6))

    # exploration strategy
    exp_schedule = LinearExploration(
        env, config.eps_begin, config.eps_end, config.eps_nsteps
    )

    # learning rate schedule
    lr_schedule = LinearSchedule(config.lr_begin, config.lr_end, config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule, run_idx=1)
