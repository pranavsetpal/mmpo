import torch as T
import torch.nn as nn

from collections import deque
import random


class replay_buffer():
    def __init__(self, state_dims, action_dims, maxlen):
        self.state = T.zeros(maxlen, *state_dims)
        self.action = T.zeros(maxlen, *action_dims)
        self.reward = T.zeros(maxlen)
        self.new_state = T.zeros(maxlen, *state_dims)

        self.len = 0
        self.maxlen = maxlen

    # Useless
    def __len__():
        return self.len

    def append(s, a, r, s_):
        idx = (T.arange(s.shape[0]) + self.len) % self.maxlen
        self.state[idx] = s
        self.action[idx] = a
        self.reward[idx] = r
        self.new_state[idx] = s_

        self.len += 1

    def sample(batch_size):
        idx = random.sample(range(self.len), batch_size)
        return ( self.state[idx], self.action[idx], self.reward[idx], self.new_state[idx] ) 

    def clear():
        self.__init__(state.shape[1:], action.shape[1:], self.maxlen)


class agent(nn.Module):
    def __init__(self, n_obs, n_actions, max_buffer_size, batch_size, discount_rate):
        super().__init__()

        self.Q = nn.Sequential(
            nn.Linear(n_obs, 64),
            nn.RELU(),
            nn.Linear(64, 64),
            nn.RELU(),
            nn.Linear(64, n_actions)
        )
        self.softmax = nn.Softmax()

        self.replay_buffer = replay_buffer(n_obs, n_actions, max_buffer_size)
        self.batch_size = batch_size


    def forward(self, obs):
        actions = Q(obs)
        return actions

    def train(self):
        for epoch in epochs:
            replay_buffer.clear()
            for _ in range(max_buffer_size): #TODO
                optimize()

    def optimize(self):
        if replay_buffer.len < batch_size:
            return

        state_buffer, action_buffer, reward_buffer, new_state_buffer = replay_buffer.sample(batch_size)
        target_Q = reward_buffer + (self.discount_rate * agent(new_state_buffer))

        loss = nn.MSELoss(Q, target_Q)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.parameters(), CLIP_VALUE) # TODO: Get Clip Value
        optimizer.step()


    def choose_action(self, state):
        #TODO: Calculate EPS_TRHESHOLD
        if random.random() > eps_threshold: action = agent(state)
                                      else: action = random.randrange(n_actions)
        return action
