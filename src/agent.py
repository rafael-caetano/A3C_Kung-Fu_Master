import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Agent(nn.Module):
    def __init__(self, input_dims, n_actions, lr):
        super(Agent, self).__init__()
        
        self.input_dims = input_dims
        self.n_actions = n_actions

        self.l1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(3,3), stride=2)
        self.l2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=2)
        self.l3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=2)
        self.l4 = nn.Flatten()
        self.l5 = nn.Linear(512, 128)
        self.a = nn.Linear(128, n_actions)
        self.v = nn.Linear(128, 1)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        x = F.relu(self.l5(x))
        logits = self.a(x)
        state_values = self.v(x).squeeze(-1)

        return logits, state_values

    def choose_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float32, device=self.device)
        
        if state.dim() == 3:
            state = state.unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.forward(state)
            probabilities = F.softmax(logits, dim=-1)

        if state.shape[0] == 1:  # Single environment
            action = np.random.choice(self.n_actions, p=probabilities[0].cpu().numpy())
        else:  # Multiple environments
            actions = [np.random.choice(self.n_actions, p=prob.cpu().numpy()) for prob in probabilities]
            action = np.array(actions)

        return action