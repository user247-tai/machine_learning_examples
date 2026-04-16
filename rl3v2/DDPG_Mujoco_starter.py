import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer


class CriticModel(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        self.input_shape = observation_shape + action_shape
        self.network = nn.Sequential(
            nn.Linear(in_features=self.input_shape, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1),
        )

    def forward(self, x):
        return self.network(x)


class ActorModel(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=observation_shape, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=action_shape),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.network(x)


class DDPGAgent:
    def __init__(self):
        self.num_envs = 5
        self.env_name = "Ant-v5"

        self.envs = gym.vector.SyncVectorEnv(
            self.make_env_list(self.num_envs),
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
        )
        self.envs = gym.wrappers.vector.RecordEpisodeStatistics(self.envs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.buffer_size = 1_000_000
        self.gamma = 0.99
        self.batch_size = 128
        self.tau = 0.005
        self.total_training_step = 500_000 #5_000_000
        self.begin_training_step = 5_000
        self.exploration_noise = 0.1
        self.learning_rate = 1e-3
        self.max_grad_norm = 1.0

        self.single_observation_space_size = self.envs.single_observation_space.shape[0]
        self.single_action_space_size = self.envs.single_action_space.shape[0]

        # Real action bounds from env
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
            self.single_action_space_size
        ).to(self.device)

        self.critic_net = CriticModel(
            self.single_observation_space_size,
            self.single_action_space_size
        ).to(self.device)

        self.target_actor_net = ActorModel(
            self.single_observation_space_size,
            self.single_action_space_size
        ).to(self.device)

        self.target_critic_net = CriticModel(
            self.single_observation_space_size,
            self.single_action_space_size
        ).to(self.device)

        self.target_actor_net.load_state_dict(self.actor_net.state_dict())
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())

        self.actor_optimizer = optim.AdamW(self.actor_net.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.AdamW(self.critic_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

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
        # map [-1, 1] -> [low, high]
        return action_tanh * self.action_scale + self.action_bias

    def select_action(self, obs, add_noise=True):
        obs_tensor = torch.from_numpy(obs).float().to(self.device)

        with torch.no_grad():
            action_tensor = self.actor_net(obs_tensor)
            action_tensor = self.scale_action(action_tensor)

            if add_noise:
                noise = torch.randn_like(action_tensor) * (self.exploration_noise * self.action_scale)
                action_tensor = action_tensor + noise

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

        batch = self.memory.sample(self.batch_size)

        state_batch = batch.observations.float().to(self.device)
        next_state_batch = batch.next_observations.float().to(self.device)
        action_batch = batch.actions.float().to(self.device)
        reward_batch = batch.rewards.flatten().float().to(self.device)
        done_batch = batch.dones.flatten().float().to(self.device)

        # ----- Critic target -----
        with torch.no_grad():
            next_actions = self.target_actor_net(next_state_batch)
            next_actions = self.scale_action(next_actions)

            target_critic_input = torch.cat((next_state_batch, next_actions), dim=1)
            next_q = self.target_critic_net(target_critic_input).squeeze(-1)

            td_target = reward_batch + self.gamma * next_q * (1.0 - done_batch)

        # ----- Critic update -----
        critic_input = torch.cat((state_batch, action_batch), dim=1)
        current_q = self.critic_net(critic_input).squeeze(-1)

        critic_loss = self.loss_fn(current_q, td_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # ----- Actor update -----
        actions_pred = self.actor_net(state_batch)
        actions_pred = self.scale_action(actions_pred)

        actor_input = torch.cat((state_batch, actions_pred), dim=1)
        actor_loss = -self.critic_net(actor_input).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # ----- Soft update -----
        self.soft_update(self.target_actor_net, self.actor_net)
        self.soft_update(self.target_critic_net, self.critic_net)

    # ===== TRAIN =====
    def train(self):
        obs, _ = self.envs.reset()
        episode_count = 0

        for step in range(self.total_training_step):
            if step < self.begin_training_step:
                action = self.envs.action_space.sample()
            else:
                action = self.select_action(obs, add_noise=True)

            next_obs, rewards, terminateds, truncateds, infos = self.envs.step(action)

            # Keep true terminal/truncated observation for replay
            real_next_obs = self.get_real_next_obs(next_obs, infos)

            # Use terminated only for bootstrap stopping.
            # Truncated episodes (time limit) can still bootstrap.
            self.memory.add(
                obs=obs,
                next_obs=real_next_obs,
                action=action,
                reward=rewards,
                done=terminateds,
                infos=infos
            )

            self.optimize()
            obs = next_obs

            # Log finished episodes
            if "episode" in infos and "_episode" in infos:
                for i in range(self.num_envs):
                    if infos["_episode"][i]:
                        episode_count += 1
                        print(
                            f"Step={step} | Episode={episode_count} | "
                            f"Return={infos['episode']['r'][i]:.2f}"
                        )

    # ===== TEST =====
    def make_eval_env(self):
        env = gym.make(self.env_name, render_mode="human")
        return env

    def test(self):
        env = self.make_eval_env()
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        self.actor_net.eval()

        with torch.no_grad():
            while not done:
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                action_tensor = self.actor_net(obs_tensor)
                action_tensor = self.scale_action(action_tensor)
                action = action_tensor.squeeze(0).cpu().numpy()

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated

        print(f"Test Reward: {total_reward}")
        env.close()

    # ===== SAVE / LOAD =====
    def save_model(self, path="ant.pth"):
        torch.save(self.actor_net.state_dict(), "actor_" + path)
        torch.save(self.critic_net.state_dict(), "critic_" + path)

    def load_model(self, path="ant.pth"):
        self.actor_net.load_state_dict(torch.load("actor_" + path, map_location=self.device))
        self.actor_net.to(self.device)
        self.target_actor_net.load_state_dict(self.actor_net.state_dict())
        self.actor_net.eval()

        self.critic_net.load_state_dict(torch.load("critic_" + path, map_location=self.device))
        self.critic_net.to(self.device)
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())
        self.critic_net.eval()


def main():
    agent = DDPGAgent()
    agent.train()
    agent.save_model()

    # agent.load_model()
    # agent.test()


if __name__ == "__main__":
    main()