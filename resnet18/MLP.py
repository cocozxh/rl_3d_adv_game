from functools import partial
from telnetlib import GA
from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 1000)
        self.hidden_fc = nn.Linear(1000, 500)
        self.hidden_fc2 = nn.Linear(500, 500)
        self.hidden_fc3 = nn.Linear(500, 1000)
        self.output_fc = nn.Linear(1000, output_dim)

    def forward(self, x):

        # x = [batch size, height, width]

        batch_size = x.shape[0]

        h_1 = F.relu(self.input_fc(x))

        h_2 = F.relu(self.hidden_fc(h_1))

        h_3 = F.relu(self.hidden_fc2(h_2))

        h_4 = F.relu(self.hidden_fc3(h_3))



        y_pred = self.output_fc(h_4)



        return y_pred