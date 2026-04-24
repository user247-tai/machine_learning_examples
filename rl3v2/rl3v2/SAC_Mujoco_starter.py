import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from stable_baselines3.common.buffers import ReplayBuffer


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
    def __init__(self, observation_shape, action_shape, hidden_dim=256):
        super().__init__()
        self.input_shape = observation_shape + action_shape
        self.network = nn.Sequential(
            nn.Linear(in_features=self.input_shape, out_features=hidden_dim),
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

    def forward(self, state, action):
        critic_input = torch.cat((state, action), dim=1)
        return self.network(critic_input)


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
        self.log_std_head = nn.Linear(in_features=hidden_dim, out_features=action_shape)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            init_linear_layer(
                module,
                final_layer=module is self.mean_head or module is self.log_std_head,
            )

    def forward(self, x):
        features = self.backbone(x)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        return mean, log_std


class SACAgent:
    def __init__(self):
        self.num_envs = 10
        self.env_name = "HumanoidStandup-v5"

        self.envs = gym.vector.SyncVectorEnv(
            self.make_env_list(self.num_envs),
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )
        self.envs = gym.wrappers.vector.RecordEpisodeStatistics(self.envs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.buffer_size = 2_000_000
        self.gamma = 0.99
        self.batch_size = 1024
        self.tau = 0.005
        self.total_training_step = 10_000_000
        self.begin_training_step = 100_000
        self.learning_rate = 3e-4
        self.max_grad_norm = 1.0
        self.updates_per_step = 2
        self.total_it = 0

        self.single_observation_space_size = self.envs.single_observation_space.shape[0]
        self.single_action_space_size = self.envs.single_action_space.shape[0]

        self.action_low_np = self.envs.single_action_space.low.astype(np.float32)
        self.action_high_np = self.envs.single_action_space.high.astype(np.float32)

        self.action_low = torch.as_tensor(self.action_low_np, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(self.action_high_np, dtype=torch.float32, device=self.device)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

        self.memory = ReplayBuffer(
            buffer_size=self.buffer_size,
            observation_space=self.envs.single_observation_space,
            action_space=self.envs.single_action_space,
            n_envs=self.num_envs,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
            device=self.device,
        )

        self.actor_net = ActorModel(
            self.single_observation_space_size,
            self.single_action_space_size,
        ).to(self.device)

        self.critic_net1 = CriticModel(
            self.single_observation_space_size,
            self.single_action_space_size,
        ).to(self.device)

        self.critic_net2 = CriticModel(
            self.single_observation_space_size,
            self.single_action_space_size,
        ).to(self.device)

        self.target_critic_net1 = CriticModel(
            self.single_observation_space_size,
            self.single_action_space_size,
        ).to(self.device)

        self.target_critic_net2 = CriticModel(
            self.single_observation_space_size,
            self.single_action_space_size,
        ).to(self.device)

        self.target_critic_net1.load_state_dict(self.critic_net1.state_dict())
        self.target_critic_net2.load_state_dict(self.critic_net2.state_dict())

        self.actor_optimizer = optim.AdamW(self.actor_net.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.AdamW(
            list(self.critic_net1.parameters()) + list(self.critic_net2.parameters()),
            lr=self.learning_rate,
        )

        self.target_entropy = -float(self.single_action_space_size)
        self.log_alpha = torch.zeros(1, device=self.device, requires_grad=True)
        self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    @property
    def alpha(self):
        return self.log_alpha.exp()

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

    def unscale_action(self, action):
        return (action - self.action_bias) / (self.action_scale + EPS)

    def sample_policy_action(self, obs_tensor):
        mean, log_std = self.actor_net(obs_tensor)
        std = log_std.exp()
        normal_dist = Normal(mean, std)

        pre_tanh_action = normal_dist.rsample()
        action_tanh = torch.tanh(pre_tanh_action)
        action = self.scale_action(action_tanh)

        log_prob = normal_dist.log_prob(pre_tanh_action)
        correction = torch.log(self.action_scale * (1.0 - action_tanh.pow(2)) + EPS)
        log_prob = (log_prob - correction).sum(dim=1, keepdim=True)

        mean_action_tanh = torch.tanh(mean)
        mean_action = self.scale_action(mean_action_tanh)
        return action, log_prob, mean_action

    def select_action(self, obs, deterministic=False):
        obs_tensor = torch.from_numpy(obs).float().to(self.device)

        with torch.no_grad():
            if deterministic:
                _, _, action_tensor = self.sample_policy_action(obs_tensor)
            else:
                action_tensor, _, _ = self.sample_policy_action(obs_tensor)

            action_tensor = torch.max(torch.min(action_tensor, self.action_high), self.action_low)

        return action_tensor.cpu().numpy()

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

    def soft_update(self, target_net, source_net):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    # ===== OPTIMIZE =====
    def optimize(self):
        if self.memory.size() < max(self.batch_size, self.begin_training_step):
            return

        for _ in range(self.updates_per_step):
            batch = self.memory.sample(self.batch_size)

            state_batch = batch.observations.float().to(self.device)
            next_state_batch = batch.next_observations.float().to(self.device)
            action_batch = batch.actions.float().to(self.device)
            reward_batch = batch.rewards.flatten().float().to(self.device)
            done_batch = batch.dones.flatten().float().to(self.device)

            with torch.no_grad():
                next_action_batch, next_log_prob_batch, _ = self.sample_policy_action(next_state_batch)
                next_q1 = self.target_critic_net1(next_state_batch, next_action_batch).squeeze(-1)
                next_q2 = self.target_critic_net2(next_state_batch, next_action_batch).squeeze(-1)
                next_q = torch.min(next_q1, next_q2) - self.alpha.detach() * next_log_prob_batch.squeeze(-1)
                td_target = reward_batch + self.gamma * next_q * (1.0 - done_batch)

            current_q1 = self.critic_net1(state_batch, action_batch).squeeze(-1)
            current_q2 = self.critic_net2(state_batch, action_batch).squeeze(-1)

            critic_loss1 = self.loss_fn(current_q1, td_target)
            critic_loss2 = self.loss_fn(current_q2, td_target)
            critic_loss = critic_loss1 + critic_loss2

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_net1.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic_net2.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            sampled_action_batch, log_prob_batch, _ = self.sample_policy_action(state_batch)
            q1_pi = self.critic_net1(state_batch, sampled_action_batch)
            q2_pi = self.critic_net2(state_batch, sampled_action_batch)
            min_q_pi = torch.min(q1_pi, q2_pi)

            actor_loss = (self.alpha.detach() * log_prob_batch - min_q_pi).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            alpha_loss = -(self.log_alpha * (log_prob_batch + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.soft_update(self.target_critic_net1, self.critic_net1)
            self.soft_update(self.target_critic_net2, self.critic_net2)

            self.total_it += 1

    # ===== TRAIN =====
    def train(self):
        self.actor_net.train()
        self.critic_net1.train()
        self.critic_net2.train()

        obs, _ = self.envs.reset()
        episode_count = 0

        for step in range(self.total_training_step):
            if step < self.begin_training_step:
                action = self.envs.action_space.sample()
            else:
                action = self.select_action(obs, deterministic=False)

            next_obs, rewards, terminateds, truncateds, infos = self.envs.step(action)
            real_next_obs = self.get_real_next_obs(next_obs, infos)

            self.memory.add(
                obs=obs,
                next_obs=real_next_obs,
                action=action,
                reward=rewards,
                done=terminateds,
                infos=infos,
            )

            self.optimize()
            obs = next_obs

            if "episode" in infos and "_episode" in infos:
                for i in range(self.num_envs):
                    if infos["_episode"][i]:
                        episode_count += 1
                        print(
                            f"Step={step} | Update={self.total_it} | Episode={episode_count} | "
                            f"Return={infos['episode']['r'][i]:.2f} | Alpha={self.alpha.item():.4f}"
                        )

    # ===== TEST =====
    def make_eval_env(self):
        env = gym.make(self.env_name, render_mode="human")
        return env

    def test(self):
        self.actor_net.eval()
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
    def save_model(self, path="sac_humanoid_standup.pth"):
        checkpoint = {
            "actor_net": self.actor_net.state_dict(),
            "critic_net1": self.critic_net1.state_dict(),
            "critic_net2": self.critic_net2.state_dict(),
            "target_critic_net1": self.target_critic_net1.state_dict(),
            "target_critic_net2": self.target_critic_net2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }
        torch.save(checkpoint, path)

    def load_model(self, path="sac_humanoid_standup.pth"):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_net.load_state_dict(checkpoint["actor_net"])
        self.critic_net1.load_state_dict(checkpoint["critic_net1"])
        self.critic_net2.load_state_dict(checkpoint["critic_net2"])
        self.target_critic_net1.load_state_dict(checkpoint["target_critic_net1"])
        self.target_critic_net2.load_state_dict(checkpoint["target_critic_net2"])
        self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
        self.actor_net.to(self.device)
        self.critic_net1.to(self.device)
        self.critic_net2.to(self.device)
        self.target_critic_net1.to(self.device)
        self.target_critic_net2.to(self.device)
        self.actor_net.eval()


def main():
    agent = SACAgent()
    agent.train()
    agent.save_model()

    # agent.load_model()
    # agent.test()


if __name__ == "__main__":
    main()
