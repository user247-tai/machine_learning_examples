import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common import atari_wrappers
from stable_baselines3.common.buffers import ReplayBuffer
import ale_py


class DeepCNNModel(nn.Module):
    def __init__(self, action_shape):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), #64x7x7
            nn.ReLU(),
            nn.Linear(512, action_shape),
        )

    def forward(self, x):
        return self.network(x / 255.0)


class Agent:
    def __init__(self):
        self.num_envs = 5
        self.envs = gym.vector.SyncVectorEnv(self.make_env_list(self.num_envs))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory = ReplayBuffer(buffer_size=50_000, 
                                   observation_space=self.envs.single_observation_space,
                                   action_space=self.envs.single_action_space,
                                   n_envs=self.num_envs,
                                   optimize_memory_usage=True,
                                   handle_timeout_termination=False)

        self.gamma = 0.99
        self.epsilon = 0.1
        self.batch_size = 128
        self.tau = 0.005

        action_dim = self.envs.single_action_space.n
        self.policy_model = DeepCNNModel(action_dim).to(self.device)
        self.target_model = DeepCNNModel(action_dim).to(self.device)

        self.target_model.load_state_dict(self.policy_model.state_dict())

        self.optimizer = optim.AdamW(self.policy_model.parameters(), lr=1e-3)
        self.loss_fn = nn.HuberLoss()

    # ===== ENV =====
    def make_env(self):
        def thunk():
            env = gym.make("BreakoutNoFrameskip-v4")
            env = atari_wrappers.NoopResetEnv(env)
            env = atari_wrappers.EpisodicLifeEnv(env)
            env = atari_wrappers.MaxAndSkipEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = atari_wrappers.FireResetEnv(env)
            env = atari_wrappers.ClipRewardEnv(env)
            env = gym.wrappers.GrayscaleObservation(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.FrameStackObservation(env, 4)
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
        if self.memory.size() < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        state_batch = batch.observations.float().to(self.device)
        next_state_batch = batch.next_observations.float().to(self.device)
        action_batch = batch.actions.long().to(self.device)
        reward_batch = batch.rewards.flatten().to(self.device)
        done_batch = batch.dones.flatten().to(self.device)

        q_values = self.policy_model(state_batch)
        state_action_values = q_values.gather(1, action_batch).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_model(next_state_batch).max(1).values
            next_q_values = next_q_values * (1.0 - done_batch)

        td_target = reward_batch + self.gamma * next_q_values
        self.optimizer.zero_grad()
        loss = self.loss_fn(state_action_values, td_target)
        loss.backward()
        self.optimizer.step()

    # ===== TRAIN =====
    def train(self, total_steps=100_000):
        obs, _ = self.envs.reset()

        episode_count = 0

        for step in range(total_steps):
            self.epsilon = self.exp_schedule(1.0, 0.05, 1e-4, step)
            actions = self.epsilon_greedy(obs)

            next_obs, rewards, terminateds, truncateds, infos = self.envs.step(actions)

            for i in range(self.num_envs):
                done = terminateds[i] or truncateds[i]
                
                if done:
                    episode_count += 1 
                    print(f"Step={step}, Episode={episode_count}, Return={infos['episode']['r'][i]}")

            # SB3 ReplayBuffer.add expects vectorized NumPy batches, not Python lists of tensors.
            dones = np.logical_or(terminateds, truncateds)
            self.memory.add(
                obs=obs,
                next_obs=next_obs,
                action=np.asarray(actions).reshape(self.num_envs, 1),
                reward=np.asarray(rewards, dtype=np.float32),
                done=dones,
                infos=[{} for _ in range(self.num_envs)],
            )

            self.optimize()
            obs = next_obs

            # soft update
            for target_param, policy_param in zip(self.target_model.parameters(), self.policy_model.parameters()):
                target_param.data.copy_(self.tau * policy_param.data +(1.0 - self.tau) * target_param.data)

    # ===== TEST =====
    def make_eval_env(self):
        env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
        env = atari_wrappers.NoopResetEnv(env)
        env = atari_wrappers.MaxAndSkipEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = atari_wrappers.FireResetEnv(env)
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.FrameStackObservation(env, 4)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    def test(self):
        env = self.make_eval_env()
        obs, _ = env.reset()
        done = False

        self.policy_model.eval()
        with torch.no_grad():
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                action = torch.argmax(self.policy_model(obs_tensor), dim=1).item()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

        if "episode" in info:
            print(f"Test Reward: {info['episode']['r']}")
        env.close()

    def save_model(self, path="breakout.pth"):
        torch.save(self.policy_model.state_dict(), path)

    def load_model(self, path="breakout.pth"):
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
