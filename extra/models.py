from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.utils import get_activation_fn
import torch
import torch.nn as nn

class ActorCriticMLP(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        activation = get_activation_fn(model_config.get("fcnet_activation", "relu"))

        # Define a custom MLP model
        self.fc1 = nn.Linear(obs_space.shape[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, num_outputs)

        self.activation = activation
        self.value_branch = nn.Linear(256, 1)  # Separate branch for value function

    def forward(self, input_dict, state, seq_lens):
        x = self.activation(self.fc1(input_dict["obs"]))
        x = self.activation(self.fc2(x))
        self._last_x = self.activation(self.fc3(x))
        logits = self.fc4(x)
        return logits, state

    def value_function(self):
        return self.value_branch(self._last_x).squeeze(1)
