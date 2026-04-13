import random
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


# settings
env_id = "CartPole-v1"
num_envs = 4
total_timesteps = 100_000
learning_rate = 8e-4
gamma = 0.99
seed = None
entropy_weight = 0.01
value_weight = 0.25
max_norm = 0.5
video_path = "videos_a2c_cartpole"


def make_env(env_id, capture_video, seed=None):
    if capture_video:
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env,
            video_path,
            episode_trigger=lambda episode: True,
        )
    else:
        env = gym.make(env_id)

    env = gym.wrappers.RecordEpisodeStatistics(env)

    if seed is not None:
        env.action_space.seed(seed)

    return env


class ActorCritic(nn.Module):
    def __init__(self, envs, n_hidden=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(envs.single_observation_space.shape[0], n_hidden),
            nn.ReLU(),
        )
        self.actor = nn.Linear(n_hidden, envs.single_action_space.n)
        self.critic = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.network(x)
        return self.actor(x), self.critic(x)


def sample_action(logits):
    dist = Categorical(logits=logits)
    return dist.sample()


def compute_entropy_and_log_prob(logits, actions):
    dist = Categorical(logits=logits)
    return dist.entropy().mean(), dist.log_prob(actions)


def np2torch(a, dtype=torch.float32, device=None):
    return torch.as_tensor(a, dtype=dtype, device=device)


def smooth(x, a=0.1):
    y = [x[0]]
    for xi in x[1:]:
        yi = a * xi + (1 - a) * y[-1]
        y.append(yi)
    return y


def train():
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    envs = gym.vector.SyncVectorEnv(
        [
            lambda i=i: make_env(
                env_id,
                False,
                seed if seed is None else seed + i,
            )
            for i in range(num_envs)
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ac_network = ActorCritic(envs).to(device)
    optimizer = optim.Adam(ac_network.parameters(), lr=learning_rate)

    episode_returns = []
    losses = []

    start_time = time.time()
    obs, _ = envs.reset(seed=seed)
    autoreset = np.zeros(num_envs, dtype=bool)

    for global_step in range(total_timesteps):
        action_logits, value = ac_network(np2torch(obs, device=device))

        actions = sample_action(action_logits)
        actions_np = actions.detach().cpu().numpy()

        next_obs, rewards, dones, truncateds, infos = envs.step(actions_np)

        for i, (done, truncated) in enumerate(zip(dones, truncateds)):
            if done or truncated:
                ret = infos["episode"]["r"][i]
                episode_returns.append(ret)
                print(
                    f"global_step={global_step}, "
                    f"episode={len(episode_returns)}, "
                    f"episode_return={ret}"
                )

        obs = next_obs

        mask = np.logical_not(autoreset)
        value_ = value[mask]
        actions_ = actions[mask]
        action_logits_ = action_logits[mask]
        rewards_ = rewards[mask]
        next_obs_ = next_obs[mask]
        dones_ = dones[mask]

        with torch.no_grad():
            _, value_next = ac_network(np2torch(next_obs_, device=device))
            td_target = (
                np2torch(rewards_.flatten(), device=device)
                + gamma * np2torch(1 - dones_.flatten(), device=device) * value_next.flatten()
            )

        pred = value_.flatten()
        value_loss = F.mse_loss(pred, td_target)

        entropy, selected_log_probs = compute_entropy_and_log_prob(action_logits_, actions_)
        advantage = td_target - value_.flatten()
        policy_loss = -torch.mean(selected_log_probs * advantage.detach())

        loss = policy_loss - entropy_weight * entropy + value_weight * value_loss
        losses.append(loss.item())

        if global_step % 100 == 0:
            elapsed = max(time.time() - start_time, 1e-8)
            print("steps per second:", int(global_step / elapsed))

        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(ac_network.parameters(), max_norm)
        optimizer.step()

        autoreset = np.logical_or(dones, truncateds)

        if len(episode_returns) > 10 and np.all(np.equal(episode_returns[-10:], 500)):
            print("max reward achieved!")
            break

    envs.close()
    return ac_network, device, episode_returns, losses


def evaluate(model_path, device, n_episodes_eval=10):
    envs_eval = gym.vector.SyncVectorEnv([lambda: make_env(env_id, True)])
    model = ActorCritic(envs_eval).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    eval_returns = np.zeros(n_episodes_eval)
    obs, _ = envs_eval.reset()

    for i in range(n_episodes_eval):
        episode_done = False
        while not episode_done:
            action_logits, _ = model(np2torch(obs, device=device))
            actions = sample_action(action_logits)
            actions_np = actions.detach().cpu().numpy()

            obs, _, dones, truncateds, infos = envs_eval.step(actions_np)

            if dones[0] or truncateds[0]:
                episode_return = infos["episode"]["r"][0]
                eval_returns[i] = episode_return
                episode_done = True
                print(f"episode={i}, return={episode_return}")

    envs_eval.close()
    return eval_returns


def plot_training(episode_returns, losses):
    if episode_returns:
        plt.plot(episode_returns, alpha=0.2)
        plt.plot(smooth(episode_returns))
        plt.title("episode_returns")
        plt.show()

    if losses:
        plt.plot(losses)
        plt.title("losses")
        plt.show()


def plot_eval(eval_returns):
    plt.hist(eval_returns)
    plt.title("Eval Returns")
    plt.show()


def main():
    model, device, episode_returns, losses = train()
    plot_training(episode_returns, losses)

    model_path = "a2c_cartpole.pth"
    torch.save(model.state_dict(), model_path)

    # eval_returns = evaluate(model_path, device)
    # plot_eval(eval_returns)


if __name__ == "__main__":
    main()
