import torch as T
import torch.nn as nn

from collections import dequ
import random
from math import exp


class Q(nn.Module):
    def __init__(self, n_obs, n_actions):
        super().__init__()

        self.Q = nn.Sequential(
            nn.Linear(n_obs, 64),
            nn.RELU(),
            nn.Linear(64, 64),
            nn.RELU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, obs):
        return self.Q(obs)


class agent(nn.Module):
    def __init__(
        self, n_obs, n_actions,
        epochs, n_episodes, batch_size,
        discount_rate, eps_start, eps_end, eps_decay
     ):
        super().__init__()

        self.Q = Q(n_obs, n_actions)

        # For each Episode: State, Action, Reward, New State
        self.replay_buffer = T.Tensor([n_episode, 4])
        self.batch_size = batch_size

        self.discount_rate = discount_rate
        self.steps_done = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay #lower is faster


    def forward(self, obs):
        return T.argmax(Q(obs))

    def train(self):
        for epoch in epochs:
            # Generating Training Data
            replay_buffer.clear()
            for _ in range(n_episodes):
            while len(replay_buffer) n_episodes 
                self.run() # Fills replay_buffer

            #Optimizing Model
            for _ in range(n_episodes):
                episodes = T.multinomial(replay_buffer, batch_size) # Without Replacement!
                state_buffer, action_buffer, reward_buffer, new_state_buffer = episodes.transpose(0, 1)

                pred_Q = self(state_buffer)
                target_Q = reward_buffer
                target_Q += T.where(
                    new_state_buffer != None,
                    self(new_state_buffer), # If True
                    0                       # if False
                )

                loss = loss(pred_Q, target_Q)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(self.parameters(), CLIP_VALUE) # TODO: Get Clip Value
                optimizer.step()


    def choose_action(self, state):
        decay = exp(-1 * self.steps_done / eps_decay)
        eps_threshold = eps_end + decay*(eps_start - eps_end)
        if random.random() > eps_threshold:
            action = self(state)
        else:
            action = random.randrange(n_actions)

        return action
