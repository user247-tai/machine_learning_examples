import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque

class DeepModel(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_shape, 128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, action_shape)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        hidden = self.network(x)
        return self.actor(hidden), self.critic(hidden)


class A2CAgent:
    def __init__(self):
        self.num_envs = 5
        self.envs = gym.vector.SyncVectorEnv(self.make_env_list(self.num_envs))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = 0.99
        self.value_weight = 0.25
        self.entropy_weight = 0.01
        self.max_norm = 0.5

        obs_dim = self.envs.single_observation_space.shape[0]
        action_dim = self.envs.single_action_space.n
        self.model = DeepModel(obs_dim, action_dim).to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)

    # ===== ENV =====
    def make_env(self):
        def thunk():
            env = gym.make("CartPole-v1")
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env

        return thunk

    def make_env_list(self, n):
        return [self.make_env() for _ in range(n)]

    # ===== ACTION =====
    def sample_action(self, logits):
        dist = Categorical(logits=logits)
        actions = dist.sample()
        return actions, dist

    # ===== OPTIMIZE =====
    def optimize(self, obs, actions, rewards, next_obs, dones):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        done_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        logits, values = self.model(obs_tensor)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(action_tensor)
        entropy = dist.entropy().mean()

        with torch.no_grad():
            _, next_values = self.model(next_obs_tensor)
            next_values = next_values.squeeze(-1)

        values = values.squeeze(-1)
        targets = reward_tensor + self.gamma * next_values * (1.0 - done_tensor)
        advantages = targets - values

        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean() # MSE Loss

        loss = policy_loss + self.value_weight * value_loss - self.entropy_weight * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
        self.optimizer.step()

    # ===== TRAIN =====
    def train(self, total_steps=100_000):
        obs, _ = self.envs.reset()
        episode_count = 0
        episode_returns = []

        for step in range(total_steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                logits, _ = self.model(obs_tensor)
                action_tensor, _ = self.sample_action(logits)

            actions = action_tensor.cpu().numpy()
            next_obs, rewards, terminateds, truncateds, infos = self.envs.step(actions)
            dones = np.logical_or(terminateds, truncateds)

            for i in range(self.num_envs):
                if dones[i]:
                    episode_count += 1
                    print(f"Step={step}, Episode={episode_count}, Return={infos['episode']['r'][i]}")
                    episode_returns.append(infos['episode']['r'][i])

            self.optimize(obs, actions, rewards, next_obs, dones)
            obs = next_obs

            if len(episode_returns) > 10 and np.all(np.equal(episode_returns[-10:], 500)):
                print("max reward achieved!")
                break
            
    # ===== TEST =====
    def test(self):
        env = gym.make("CartPole-v1", render_mode="human")
        env = gym.wrappers.RecordEpisodeStatistics(env)

        obs, _ = env.reset()
        done = False

        while not done:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                logits, _ = self.model(obs_tensor)
                action = torch.argmax(logits, dim=1).item()

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if done:
                print(f"Test Reward: {info['episode']['r']}")

        env.close()

    def save_model(self, path="cartpole.pth"):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path="cartpole.pth"):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()


def main():
    agent = A2CAgent()
    agent.train()
    agent.save_model()
    # agent.load_model()
    # agent.test()


if __name__ == "__main__":
    main()
