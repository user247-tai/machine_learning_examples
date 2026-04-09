import gymnasium as gym
import numpy as np
from collections import namedtuple, deque
import random
import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DeepModel(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_shape, 128),
            nn.ReLU(),
            nn.Linear(128, action_shape) 
        )

    def forward(self, x):
        return self.net(x)


class Agent:
    def __init__(self):
        self.num_envs = 5
        self.envs = gym.vector.SyncVectorEnv(self.make_env_list(self.num_envs))

        self.memory = ReplayMemory(10000)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = 0.99
        self.epsilon = 0.1
        self.batch_size = 128
        self.tau = 0.005

        self.policy_model = DeepModel(4, 2).to(self.device)
        self.target_model = DeepModel(4, 2).to(self.device)

        self.target_model.load_state_dict(self.policy_model.state_dict())

        self.optimizer = optim.AdamW(self.policy_model.parameters(), lr=1e-3)
        self.loss_fn = nn.HuberLoss()

    # ===== ENV =====
    def make_env(self):
        def thunk():
            env = gym.make("CartPole-v1")
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env
        return thunk

    def make_env_list(self, n):
        return [self.make_env() for _ in range(n)]
    
    def exp_schedule(self, start, end, decay_rate, t):
        return end + (start - end) * np.exp(-decay_rate * t)

    # ===== ACTION =====
    def epsilon_greedy(self, obs):
        if np.random.random() < self.epsilon:
            return self.envs.action_space.sample()

        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().to(self.device)
            q_values = self.policy_model(obs_tensor)
            actions = torch.argmax(q_values, dim=1)
            return actions.cpu().numpy()

    # ===== OPTIMIZE =====
    def optimize(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # mask
        non_final_mask = torch.tensor(
            [s is not None for s in batch.next_state],
            dtype=torch.bool,
            device=self.device
        )

        # batch
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        # next states
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        ).to(self.device)

        # Q(s,a)
        q_values = self.policy_model(state_batch)
        state_action_values = q_values.gather(1, action_batch).squeeze(1)

        # Q(s',a')
        next_q_values = torch.zeros(self.batch_size, device=self.device)

        with torch.no_grad():
            if len(non_final_next_states) > 0:
                next_q_values[non_final_mask] = (self.target_model(non_final_next_states).max(1).values)

        # target
        td_target = reward_batch + self.gamma * next_q_values

        # optimize
        self.optimizer.zero_grad()
        loss = self.loss_fn(state_action_values, td_target)
        loss.backward()
        self.optimizer.step()

    # ===== TRAIN =====
    def train(self, total_steps=30000):
        obs, _ = self.envs.reset()

        episode_count = 0
        exploration_duration = int(0.1 * total_steps)

        for step in range(total_steps):
            #self.epsilon = self.exp_schedule(1.0, 0.05, 1e-4, step)
            actions = self.epsilon_greedy(obs)

            next_obs, rewards, terminateds, truncateds, infos = self.envs.step(actions)

            for i in range(self.num_envs):
                done = terminateds[i] or truncateds[i]

                if done:
                    episode_count += 1
                    print(f"Step={step}, Episode={episode_count}, Return={infos['episode']['r'][i]}")

                state = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0)
                action = torch.tensor([[actions[i]]], dtype=torch.int64)
                reward = torch.tensor([rewards[i]], dtype=torch.float32)

                next_state = None if done else torch.tensor(next_obs[i], dtype=torch.float32).unsqueeze(0)

                self.memory.push(state, action, reward, next_state)

            self.optimize()
            obs = next_obs

            # soft update
            for target_param, policy_param in zip(self.target_model.parameters(), self.policy_model.parameters()):
                target_param.data.copy_(self.tau * policy_param.data +(1.0 - self.tau) * target_param.data)

    # ===== TEST =====
    def test(self):
        env = gym.make("CartPole-v1", render_mode="human")
        env = gym.wrappers.RecordEpisodeStatistics(env)

        obs, _ = env.reset()
        done = False

        while not done:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                action = torch.argmax(self.policy_model(obs_tensor), dim=1).item()

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if done:
                print(f"Test Reward: {info['episode']['r']}")

        env.close()

    def save_model(self, path="cartpole.pth"):
        torch.save(self.policy_model.state_dict(), path)

    def load_model(self, path="cartpole.pth"):
        self.policy_model.load_state_dict(torch.load(path, map_location=self.device))
        self.policy_model.to(self.device)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.policy_model.eval() 


def main():
    agent = Agent()
    agent.train()
    agent.save_model()
    # agent.load_model()
    # agent.test()


if __name__ == "__main__":
    main()