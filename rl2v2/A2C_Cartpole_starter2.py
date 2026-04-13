import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class DeepModel(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_shape, 128),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, action_shape)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        hidden = self.net(x)
        logits = self.policy_head(hidden)
        values = self.value_head(hidden)
        return logits, values


class A2CAgent:
    def __init__(self):
        self.num_envs = 5
        self.n_steps = 5

        self.envs = gym.vector.SyncVectorEnv(self.make_env_list(self.num_envs))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.value_weight = 0.5
        self.entropy_weight = 0.01
        self.max_grad_norm = 0.5

        obs_dim = self.envs.single_observation_space.shape[0]
        action_dim = self.envs.single_action_space.n

        self.model = DeepModel(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=3e-4)

    # ===== ENV =====
    def make_env(self):
        def thunk():
            env = gym.make("CartPole-v1")
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env
        return thunk

    def make_env_list(self, n):
        return [self.make_env() for _ in range(n)]

    # ===== SEED =====
    def seed_everything(self, seed=123):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ===== ACTION / VALUE =====
    def sample_action_and_value(self, obs):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits, values = self.model(obs_tensor)
            dist = Categorical(logits=logits)
            actions = dist.sample()

        return actions.cpu().numpy(), values.squeeze(-1).cpu().numpy()

    def predict_values(self, obs):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, values = self.model(obs_tensor)

        return values.squeeze(-1).cpu().numpy()

    # ===== INFO HELPERS =====
    def get_real_next_obs(self, next_obs, infos, dones):
        """
        In vector envs, when an env ends, next_obs may already be the reset observation.
        If Gymnasium provides final_observation, use that for correct bootstrapping.
        """
        real_next_obs = np.array(next_obs, copy=True)

        if "final_observation" not in infos:
            return real_next_obs

        final_obs = infos["final_observation"]
        final_obs_mask = infos.get("_final_observation", dones)

        for i in range(self.num_envs):
            if final_obs_mask[i]:
                if final_obs[i] is not None:
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

    # ===== GAE =====
    def compute_gae(self, rewards, values, next_values, terminateds, dones):
        """
        rewards, values, next_values, terminateds, dones:
            shape [T, N]
        """
        T, N = rewards.shape
        advantages = np.zeros((T, N), dtype=np.float32)
        gae = np.zeros(N, dtype=np.float32)

        for t in reversed(range(T)):
            # Bootstrap from next value unless true termination
            bootstrap_mask = 1.0 - terminateds[t].astype(np.float32)

            # Do not propagate advantage recursion across episode boundary
            continuation_mask = 1.0 - dones[t].astype(np.float32)

            delta = rewards[t] + self.gamma * next_values[t] * bootstrap_mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * continuation_mask * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    # ===== OPTIMIZE =====
    def optimize(self, rollout):
        obs = torch.as_tensor(rollout["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(rollout["actions"], dtype=torch.int64, device=self.device)
        advantages = torch.as_tensor(rollout["advantages"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(rollout["returns"], dtype=torch.float32, device=self.device)

        # Flatten [T, N, ...] -> [T*N, ...]
        obs = obs.reshape(-1, obs.shape[-1])
        actions = actions.reshape(-1)
        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logits, values = self.model(obs)
        values = values.squeeze(-1)

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        policy_loss = -(log_probs * advantages).mean()
        value_loss = 0.5 * (returns - values).pow(2).mean()

        loss = policy_loss + self.value_weight * value_loss - self.entropy_weight * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

    # ===== TRAIN =====
    def train(self, total_env_steps=200000, seed=123):
        self.seed_everything(seed)
        self.model.train()

        obs, _ = self.envs.reset(seed=seed)

        episode_count = 0
        updates = max(1, total_env_steps // (self.num_envs * self.n_steps))

        for update_idx in range(updates):
            rollout_obs = []
            rollout_actions = []
            rollout_rewards = []
            rollout_values = []
            rollout_next_values = []
            rollout_terminateds = []
            rollout_dones = []

            for step_idx in range(self.n_steps):
                actions, values = self.sample_action_and_value(obs)

                next_obs, rewards, terminateds, truncateds, infos = self.envs.step(actions)
                dones = np.logical_or(terminateds, truncateds)

                real_next_obs = self.get_real_next_obs(next_obs, infos, dones)
                next_values = self.predict_values(real_next_obs)

                rollout_obs.append(obs.copy())
                rollout_actions.append(actions.copy())
                rollout_rewards.append(rewards.astype(np.float32))
                rollout_values.append(values.astype(np.float32))
                rollout_next_values.append(next_values.astype(np.float32))
                rollout_terminateds.append(terminateds.astype(np.float32))
                rollout_dones.append(dones.astype(np.float32))

                episode_count = self.log_finished_episodes(
                    infos=infos,
                    update_idx=update_idx,
                    step_idx=step_idx,
                    episode_count=episode_count
                )

                obs = next_obs

            rollout_rewards = np.asarray(rollout_rewards, dtype=np.float32)         # [T, N]
            rollout_values = np.asarray(rollout_values, dtype=np.float32)           # [T, N]
            rollout_next_values = np.asarray(rollout_next_values, dtype=np.float32) # [T, N]
            rollout_terminateds = np.asarray(rollout_terminateds, dtype=np.float32) # [T, N]
            rollout_dones = np.asarray(rollout_dones, dtype=np.float32)             # [T, N]

            advantages, returns = self.compute_gae(
                rewards=rollout_rewards,
                values=rollout_values,
                next_values=rollout_next_values,
                terminateds=rollout_terminateds,
                dones=rollout_dones,
            )

            rollout = {
                "obs": np.asarray(rollout_obs, dtype=np.float32),           # [T, N, obs_dim]
                "actions": np.asarray(rollout_actions, dtype=np.int64),     # [T, N]
                "advantages": advantages,                                   # [T, N]
                "returns": returns,                                         # [T, N]
            }

            self.optimize(rollout)

    # ===== TEST =====
    def test(self, seed=123):
        self.model.eval()

        env = gym.make("CartPole-v1", render_mode="human")
        env = gym.wrappers.RecordEpisodeStatistics(env)

        obs, _ = env.reset(seed=seed)
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

    def save_model(self, path="cartpole_a2c.pth"):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path="cartpole_a2c.pth"):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()


def main():
    agent = A2CAgent()
    agent.train(total_env_steps=200000, seed=123)
    agent.save_model()

    # agent.load_model()
    # agent.test(seed=123)


if __name__ == "__main__":
    main()