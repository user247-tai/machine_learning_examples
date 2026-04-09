import gymnasium as gym
import numpy as np


class Agent():
    def __init__(self):
        self.num_envs = 5
        self.env_list = []
        self.envs = gym.vector.SyncVectorEnv(self.make_env_list(self.num_envs))

    def make_env(self, id, render_mode):
        env = gym.make(id, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    def make_env_list(self, num_env):
        env_list = []
        for _ in range(num_env):
            env_list.append(lambda: self.make_env(id = "CartPole-v1", render_mode=None))
        return env_list


def main():
    agent = Agent()
    observation, info = agent.envs.reset()

    episode_over = False
    while not episode_over:
        actions = agent.envs.action_space.sample()

        observations, rewards, terminateds, truncateds, infos = agent.envs.step(actions)
        

        episode_over = terminateds[0] or truncateds[0]
        # if episode_over:
        #     print(actions.shape) 
        #     print(observations.shape)
        #     print(rewards.shape)
        #     print(terminateds.shape)
        #     print(infos['episode']['r'])
        #     print(infos['episode']['l'])
        #     print("---")

        #     (5,)
        #     (5, 4)
        #     (5,)
        #     (5,)
        #     [15.  0.  0.  0.  0.]
        #     [15  0  0  0  0]

        # if episode_over:
        #     print(actions) 
        #     print(observations)
        #     print(rewards)
        #     print(terminateds)
        #     print(infos['episode']['r'])
        #     print(infos['episode']['l'])
        #     print("---")

        #     [0 1 1 1 0]
        #     [[ 0.15823022  0.94215506 -0.22623359 -1.6588615 ]
        #     [-0.06502595  0.1953286   0.09074079 -0.20377554]
        #     [ 0.1704714   0.60995376 -0.2221606  -1.331774  ]
        #     [-0.0818181  -0.6127233   0.17420748  1.0614945 ]
        #     [-0.0518797  -0.58221185 -0.01812001  0.81119007]]
        #     [1. 1. 1. 1. 1.]
        #     [ True False  True False False]
        #     [17.  0. 17.  0.  0.]
        #     [17  0 17  0  0]



if __name__ == "__main__":
    main()