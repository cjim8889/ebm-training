import torch
import torch.nn as nn

class TimeVelocityField(nn.Module):

    def __init__(self, input_dim, hidden_dim, depth=3):
        super(TimeVelocityField, self).__init__()

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim+1, hidden_dim))
        for _ in range(depth - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        h = torch.cat([x, t], dim=-1)

        for layer in self.layers:
            h = torch.sigmoid(layer(h))

        return self.output_layer(h)