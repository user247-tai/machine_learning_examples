import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from stable_baselines3.common import atari_wrappers
from stable_baselines3.common.buffers import RolloutBuffer
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
        )

        self.actor = nn.Linear(512, action_shape)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        hidden = self.network(x / 255.0)
        return self.actor(hidden), self.critic(hidden)


class A2CAgent:
    def __init__(self):
        self.num_envs = 5
        self.n_steps = 256
        self.envs = gym.vector.SyncVectorEnv(self.make_env_list(self.num_envs))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        action_dim = self.envs.single_action_space.n

        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.value_weight = 0.5
        self.entropy_weight = 0.01
        self.max_grad_norm = 0.5

        self.memory = RolloutBuffer(
            buffer_size=self.n_steps,
            observation_space=self.envs.single_observation_space,
            action_space=self.envs.single_action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.num_envs
        )

        self.model = DeepCNNModel(action_dim).to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=3e-4)

    # ===== ENV =====
    def make_env(self):
        def thunk():
            env = gym.make("BreakoutNoFrameskip-v4")
            env = atari_wrappers.NoopResetEnv(env, noop_max=30)
            env = atari_wrappers.EpisodicLifeEnv(env)
            env = atari_wrappers.MaxAndSkipEnv(env, skip=4)
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

    # ===== ACTION =====
    def sample_action(self, logits):
        dist = Categorical(logits=logits)
        actions = dist.sample()
        return actions, dist

    def get_real_next_obs(self, next_obs, infos, dones):
        """
        In vector envs, next_obs may already be the reset observation for finished envs.
        If available, use final_observation for more accurate value bootstrapping.
        """
        real_next_obs = np.array(next_obs, copy=True)

        if "final_observation" not in infos:
            return real_next_obs

        final_obs = infos["final_observation"]
        final_obs_mask = infos.get("_final_observation", dones)

        for i in range(self.num_envs):
            if final_obs_mask[i] and final_obs[i] is not None:
                real_next_obs[i] = final_obs[i]

        return real_next_obs

    def log_finished_episodes(self, infos, update_idx, step_idx, episode_count):
        if "episode" not in infos:
            return episode_count

        episode_mask = infos.get("_episode", None)

        for i in range(self.num_envs):
            should_log = True if episode_mask is None else bool(episode_mask[i])
            if should_log:
                episode_count += 1
                print(
                    f"Update={update_idx}, RolloutStep={step_idx}, "
                    f"Episode={episode_count}, Return={infos['episode']['r'][i]}"
                )

        return episode_count

    # ===== OPTIMIZE =====
    def optimize(self, next_obs, dones):
        with torch.no_grad():
            next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
            _, last_values = self.model(next_obs_tensor)
            last_values = last_values.squeeze(-1)

        self.memory.compute_returns_and_advantage(last_values=last_values, dones=dones)

        for memory in self.memory.get(batch_size=None):
            observations_arr = memory.observations
            actions_arr = memory.actions.long().flatten()
            advantages_arr = memory.advantages
            returns_arr = memory.returns

            new_logits, new_values = self.model(observations_arr)
            dist = Categorical(logits=new_logits)
            new_log_probs = dist.log_prob(actions_arr)
            entropy = dist.entropy().mean()
            new_values = new_values.flatten()
            policy_loss = -(advantages_arr.detach() * new_log_probs).mean()
            value_loss = 0.5 * (returns_arr - new_values).pow(2).mean()
            loss = policy_loss + self.value_weight * value_loss - self.entropy_weight * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

        self.memory.reset()

    # ===== TRAIN =====
    def train(self, total_env_steps=5_000_000):
        self.model.train()
        obs, _ = self.envs.reset()
        episode_count = 0
        updates = max(1, total_env_steps // (self.num_envs * self.n_steps))
        episode_start = np.ones(self.num_envs, dtype=np.float32)

        for update_idx in range(updates):
            for step_idx in range(self.n_steps):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

                with torch.no_grad():
                    logits, values = self.model(obs_tensor)
                    action_tensor, dist = self.sample_action(logits)
                    log_probs = dist.log_prob(action_tensor)

                actions = action_tensor.cpu().numpy()
                next_obs, rewards, terminateds, truncateds, infos = self.envs.step(actions)
                dones = np.logical_or(terminateds, truncateds)
                real_next_obs = self.get_real_next_obs(next_obs, infos, dones)

                self.memory.add(
                    obs=obs,
                    action=actions,
                    reward=rewards.astype(np.float32),
                    episode_start=episode_start,
                    value=values.squeeze(-1),
                    log_prob=log_probs
                )

                episode_start = dones.astype(np.float32)
                obs = next_obs

                episode_count = self.log_finished_episodes(
                    infos=infos,
                    update_idx=update_idx,
                    step_idx=step_idx,
                    episode_count=episode_count,
                )

            self.optimize(real_next_obs, dones)

            
    # ===== TEST =====
    def make_eval_env(self):
        env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
        env = atari_wrappers.NoopResetEnv(env, noop_max=30)
        env = atari_wrappers.MaxAndSkipEnv(env, skip=4)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = atari_wrappers.FireResetEnv(env)
        env = atari_wrappers.ClipRewardEnv(env)
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.FrameStackObservation(env, 4)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    def test(self):
        env = self.make_eval_env()
        obs, _ = env.reset()
        done = False

        self.model.eval()
        with torch.no_grad():
            while not done:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                logits, _ = self.model(obs_tensor)
                action = torch.argmax(logits, dim=1).item()
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

        if "episode" in info:
            print(f"Test Reward: {info['episode']['r']}")
        env.close()

    def save_model(self, path="a2c_breakout.pth"):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path="a2c_breakout.pth"):
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
