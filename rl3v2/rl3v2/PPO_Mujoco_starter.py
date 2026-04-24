import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPS = 1e-6


def init_linear_layer(linear_layer, final_layer=False):
    if final_layer:
        nn.init.uniform_(linear_layer.weight, -1e-3, 1e-3)
        nn.init.uniform_(linear_layer.bias, -1e-3, 1e-3)
    else:
        nn.init.xavier_uniform_(linear_layer.weight)
        nn.init.zeros_(linear_layer.bias)


class CriticModel(nn.Module):
    def __init__(self, observation_shape, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=observation_shape, out_features=hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=1),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            init_linear_layer(module, final_layer=module.out_features == 1)

    def forward(self, state):
        return self.network(state)


class ActorModel(nn.Module):
    def __init__(self, observation_shape, action_shape, hidden_dim=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_features=observation_shape, out_features=hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(in_features=hidden_dim, out_features=action_shape)
        self.log_std = nn.Parameter(torch.zeros(1, action_shape))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            init_linear_layer(module, final_layer=module is self.mean_head)

    def forward(self, x):
        features = self.backbone(x)
        mean = self.mean_head(features)
        log_std = self.log_std.expand_as(mean)
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        return mean, log_std


class PPOAgent:
    def __init__(self):
        self.num_envs = 10
        self.env_name = "HumanoidStandup-v5"

        self.envs = gym.vector.SyncVectorEnv(
            self.make_env_list(self.num_envs),
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )
        self.envs = gym.wrappers.vector.RecordEpisodeStatistics(self.envs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.learning_rate = 3e-4
        self.max_grad_norm = 1.0
        self.total_training_step = 2_000_000
        self.rollout_steps = 1024
        self.ppo_epochs = 10
        self.minibatch_size = 1024
        self.clip_coef = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.0
        self.target_kl = 0.03
        self.normalize_advantages = True

        self.single_observation_space_size = self.envs.single_observation_space.shape[0]
        self.single_action_space_size = self.envs.single_action_space.shape[0]

        self.action_low_np = self.envs.single_action_space.low.astype(np.float32)
        self.action_high_np = self.envs.single_action_space.high.astype(np.float32)

        self.action_low = torch.as_tensor(self.action_low_np, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(self.action_high_np, dtype=torch.float32, device=self.device)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

        self.actor_net = ActorModel(
            self.single_observation_space_size,
            self.single_action_space_size,
        ).to(self.device)

        self.critic_net = CriticModel(
            self.single_observation_space_size,
        ).to(self.device)

        self.optimizer = optim.AdamW(
            list(self.actor_net.parameters()) + list(self.critic_net.parameters()),
            lr=self.learning_rate,
        )

    # ===== ENV =====
    def make_env(self):
        def thunk():
            env = gym.make(self.env_name)
            return env

        return thunk

    def make_env_list(self, n):
        return [self.make_env() for _ in range(n)]

    # ===== ACTION HELPERS =====
    def scale_action(self, action_tanh):
        return action_tanh * self.action_scale + self.action_bias

    def get_real_next_obs(self, next_obs, infos):
        real_next_obs = np.array(next_obs, copy=True)

        if "final_obs" in infos:
            final_obs = infos["final_obs"]
            final_mask = infos.get("_final_obs", np.zeros(self.num_envs, dtype=bool))
        elif "final_observation" in infos:
            final_obs = infos["final_observation"]
            final_mask = infos.get("_final_observation", np.zeros(self.num_envs, dtype=bool))
        else:
            return real_next_obs

        for i in range(self.num_envs):
            if final_mask[i] and final_obs[i] is not None:
                real_next_obs[i] = final_obs[i]

        return real_next_obs

    def get_policy_dist(self, obs_tensor):
        mean, log_std = self.actor_net(obs_tensor)
        std = log_std.exp()
        return Normal(mean, std)

    def sample_action_and_value(self, obs_tensor):
        normal_dist = self.get_policy_dist(obs_tensor)
        pre_tanh_action = normal_dist.rsample()
        action_tanh = torch.tanh(pre_tanh_action)
        action = self.scale_action(action_tanh)

        log_prob = normal_dist.log_prob(pre_tanh_action)
        correction = torch.log(self.action_scale * (1.0 - action_tanh.pow(2)) + EPS)
        log_prob = (log_prob - correction).sum(dim=1)

        entropy = normal_dist.entropy().sum(dim=1)
        value = self.critic_net(obs_tensor).squeeze(-1)
        return action, log_prob, entropy, value

    def evaluate_actions(self, obs_tensor, action_tensor):
        action_tanh = ((action_tensor - self.action_bias) / (self.action_scale + EPS)).clamp(-1.0 + EPS, 1.0 - EPS)
        pre_tanh_action = 0.5 * torch.log((1.0 + action_tanh) / (1.0 - action_tanh))

        normal_dist = self.get_policy_dist(obs_tensor)
        log_prob = normal_dist.log_prob(pre_tanh_action)
        correction = torch.log(self.action_scale * (1.0 - action_tanh.pow(2)) + EPS)
        log_prob = (log_prob - correction).sum(dim=1)
        entropy = normal_dist.entropy().sum(dim=1)
        value = self.critic_net(obs_tensor).squeeze(-1)
        return log_prob, entropy, value

    def select_action(self, obs, deterministic=False):
        obs_tensor = torch.from_numpy(obs).float().to(self.device)

        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor_net(obs_tensor)
                action_tanh = torch.tanh(mean)
                action_tensor = self.scale_action(action_tanh)
            else:
                action_tensor, _, _, _ = self.sample_action_and_value(obs_tensor)

            action_tensor = torch.max(torch.min(action_tensor, self.action_high), self.action_low)

        return action_tensor.cpu().numpy()

    # ===== ROLLOUT =====
    def collect_rollout(self, obs):
        obs_buffer = np.zeros(
            (self.rollout_steps, self.num_envs, self.single_observation_space_size),
            dtype=np.float32,
        )
        action_buffer = np.zeros(
            (self.rollout_steps, self.num_envs, self.single_action_space_size),
            dtype=np.float32,
        )
        log_prob_buffer = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
        reward_buffer = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
        value_buffer = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
        terminated_buffer = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
        next_value_buffer = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)

        episode_returns = []

        for step in range(self.rollout_steps):
            obs_buffer[step] = obs

            obs_tensor = torch.from_numpy(obs).float().to(self.device)
            with torch.no_grad():
                action_tensor, log_prob_tensor, _, value_tensor = self.sample_action_and_value(obs_tensor)

            action = action_tensor.cpu().numpy()
            next_obs, rewards, terminateds, truncateds, infos = self.envs.step(action)
            real_next_obs = self.get_real_next_obs(next_obs, infos)

            action_buffer[step] = action
            log_prob_buffer[step] = log_prob_tensor.cpu().numpy()
            reward_buffer[step] = rewards
            value_buffer[step] = value_tensor.cpu().numpy()
            terminated_buffer[step] = terminateds.astype(np.float32)

            next_obs_tensor = torch.from_numpy(real_next_obs).float().to(self.device)
            with torch.no_grad():
                next_values = self.critic_net(next_obs_tensor).squeeze(-1).cpu().numpy()

            next_values = next_values * (1.0 - terminateds.astype(np.float32))
            next_value_buffer[step] = next_values

            if "episode" in infos and "_episode" in infos:
                for env_idx in range(self.num_envs):
                    if infos["_episode"][env_idx]:
                        episode_returns.append(float(infos["episode"]["r"][env_idx]))

            obs = next_obs

        advantages = np.zeros_like(reward_buffer)
        last_gae = np.zeros(self.num_envs, dtype=np.float32)

        for step in reversed(range(self.rollout_steps)):
            delta = reward_buffer[step] + self.gamma * next_value_buffer[step] - value_buffer[step]
            last_gae = delta + self.gamma * self.gae_lambda * (1.0 - terminated_buffer[step]) * last_gae
            advantages[step] = last_gae

        returns = advantages + value_buffer
        return obs, episode_returns, obs_buffer, action_buffer, log_prob_buffer, value_buffer, advantages, returns

    # ===== OPTIMIZE =====
    def optimize(self, obs_buffer, action_buffer, log_prob_buffer, value_buffer, advantages, returns):
        flat_obs = torch.as_tensor(
            obs_buffer.reshape(-1, self.single_observation_space_size),
            dtype=torch.float32,
            device=self.device,
        )
        flat_actions = torch.as_tensor(
            action_buffer.reshape(-1, self.single_action_space_size),
            dtype=torch.float32,
            device=self.device,
        )
        flat_old_log_probs = torch.as_tensor(
            log_prob_buffer.reshape(-1),
            dtype=torch.float32,
            device=self.device,
        )
        flat_old_values = torch.as_tensor(
            value_buffer.reshape(-1),
            dtype=torch.float32,
            device=self.device,
        )
        flat_advantages = torch.as_tensor(
            advantages.reshape(-1),
            dtype=torch.float32,
            device=self.device,
        )
        flat_returns = torch.as_tensor(
            returns.reshape(-1),
            dtype=torch.float32,
            device=self.device,
        )

        if self.normalize_advantages:
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + EPS)

        batch_size = flat_obs.shape[0]
        batch_indices = np.arange(batch_size)

        actor_loss_value = 0.0
        critic_loss_value = 0.0
        entropy_value = 0.0
        approx_kl_value = 0.0

        for _ in range(self.ppo_epochs):
            np.random.shuffle(batch_indices)

            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                minibatch_indices = batch_indices[start:end]

                mb_obs = flat_obs[minibatch_indices]
                mb_actions = flat_actions[minibatch_indices]
                mb_old_log_probs = flat_old_log_probs[minibatch_indices]
                mb_old_values = flat_old_values[minibatch_indices]
                mb_advantages = flat_advantages[minibatch_indices]
                mb_returns = flat_returns[minibatch_indices]

                new_log_probs, entropy, new_values = self.evaluate_actions(mb_obs, mb_actions)
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = log_ratio.exp()

                unclipped_policy_loss = -mb_advantages * ratio
                clipped_policy_loss = -mb_advantages * torch.clamp(
                    ratio,
                    1.0 - self.clip_coef,
                    1.0 + self.clip_coef,
                )
                actor_loss = torch.max(unclipped_policy_loss, clipped_policy_loss).mean()

                value_delta = new_values - mb_old_values
                clipped_values = mb_old_values + value_delta.clamp(-self.clip_coef, self.clip_coef)
                value_loss_unclipped = (new_values - mb_returns).pow(2)
                value_loss_clipped = (clipped_values - mb_returns).pow(2)
                critic_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                entropy_bonus = entropy.mean()
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor_net.parameters()) + list(self.critic_net.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().abs()

                actor_loss_value = actor_loss.item()
                critic_loss_value = critic_loss.item()
                entropy_value = entropy_bonus.item()
                approx_kl_value = approx_kl.item()

                if approx_kl_value > self.target_kl:
                    return actor_loss_value, critic_loss_value, entropy_value, approx_kl_value

        return actor_loss_value, critic_loss_value, entropy_value, approx_kl_value

    # ===== TRAIN =====
    def train(self):
        self.actor_net.train()
        self.critic_net.train()

        obs, _ = self.envs.reset()
        episode_count = 0
        global_step = 0
        total_updates = self.total_training_step // (self.rollout_steps * self.num_envs)

        for update in range(total_updates):
            (
                obs,
                episode_returns,
                obs_buffer,
                action_buffer,
                log_prob_buffer,
                value_buffer,
                advantages,
                returns,
            ) = self.collect_rollout(obs)

            actor_loss, critic_loss, entropy, approx_kl = self.optimize(
                obs_buffer,
                action_buffer,
                log_prob_buffer,
                value_buffer,
                advantages,
                returns,
            )

            global_step += self.rollout_steps * self.num_envs

            for episode_return in episode_returns:
                episode_count += 1
                print(
                    f"Step={global_step} | Update={update + 1} | Episode={episode_count} | "
                    f"Return={episode_return:.2f}"
                )

            print(
                f"Update={update + 1}/{total_updates} | Step={global_step} | "
                f"ActorLoss={actor_loss:.4f} | CriticLoss={critic_loss:.4f} | "
                f"Entropy={entropy:.4f} | KL={approx_kl:.6f}"
            )

    # ===== TEST =====
    def make_eval_env(self):
        env = gym.make(self.env_name, render_mode="human")
        return env

    def test(self):
        self.actor_net.eval()
        self.critic_net.eval()
        env = self.make_eval_env()
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = self.select_action(obs[np.newaxis, :], deterministic=True)[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Test Reward: {total_reward}")
        env.close()

    # ===== SAVE / LOAD =====
    def save_model(self, path="ppo_humanoid_standup.pth"):
        checkpoint = {
            "actor_net": self.actor_net.state_dict(),
            "critic_net": self.critic_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    def load_model(self, path="ppo_humanoid_standup.pth"):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_net.load_state_dict(checkpoint["actor_net"])
        self.critic_net.load_state_dict(checkpoint["critic_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.actor_net.to(self.device)
        self.critic_net.to(self.device)
        self.actor_net.eval()
        self.critic_net.eval()


def main():
    agent = PPOAgent()
    agent.train()
    agent.save_model()

    # agent.load_model()
    # agent.test()


if __name__ == "__main__":
    main()
