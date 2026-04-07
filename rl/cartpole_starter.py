import numpy as np
import random
import gymnasium as gym
from sklearn.kernel_approximation import RBFSampler

class Agent:
    def __init__(self, env):
        self.env = env
        self.epsilon = 0.1
        self.gamma = 0.99 
        self.alpha = 0.01 
        
        self.featurizer = RBFSampler(n_components=1000)
        
        self.samples = self.getSamples(n_samples=2000)
        self.featurizer.fit(self.samples)
        
        self.dim = self.featurizer.n_components
        self.w = np.zeros(self.dim)
        
    def one_hot_action(self, action: int) -> np.ndarray:
        n = self.env.action_space.n
        encoding = np.zeros(n, dtype=np.float32)
        encoding[action] = 1
        return encoding

    def get_input_vector(self, state, action):
        action_encoded = self.one_hot_action(action)
        return np.concatenate([state, action_encoded])

    def predict(self, state, action):
        data = self.get_input_vector(state, action)
        x = self.featurizer.transform([data])[0]
        return x @ self.w

    def grad(self, state, action):
        data = self.get_input_vector(state, action)
        return self.featurizer.transform([data])[0]

    def epsilonGreedy(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            qs = [self.predict(state, a) for a in range(self.env.action_space.n)]
            return np.argmax(qs)

    def getSamples(self, n_samples=2000):
        samples = []
        while len(samples) < n_samples:
            state, _ = self.env.reset()
            done = False
            while not done and len(samples) < n_samples:
                action = self.env.action_space.sample()
                samples.append(self.get_input_vector(state, action))
                state, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
        return samples

    def q_learning(self, episodes=500):
        for it in range(episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = self.epsilonGreedy(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                if terminated:
                    target = reward
                else:
                    qs_next = [self.predict(next_state, a_n) for a_n in range(self.env.action_space.n)]
                    target = reward + self.gamma * np.max(qs_next)
                
                # w = w + alpha * (target - prediction) * gradient
                current_prediction = self.predict(state, action)
                self.w += self.alpha * (target - current_prediction) * self.grad(state, action)
                
                state = next_state
                total_reward += reward
            
            if (it + 1) % 50 == 0:
                print(f"Episode {it + 1}: Total Reward = {total_reward}")

    def test_agent(self, test_episodes=5):
        for it in range(test_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                qs = [self.predict(state, a) for a in range(self.env.action_space.n)]
                action = np.argmax(qs)
                state, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
            
            print(f"Test Episode {it + 1}: Reward = {total_reward}")

def main():
    env_train = gym.make("CartPole-v1")
    agent = Agent(env_train)
    
    print("\n--- Begin Training ---")
    agent.q_learning(episodes=5000)
    print("\n--- End Training ---")
    print()
    
    print("\n--- Begin Testing ---")
    env_test = gym.make("CartPole-v1", render_mode="human")
    agent.env = env_test
    agent.test_agent()
    env_test.close()
    print("\n--- End Testing ---")

if __name__ == "__main__":
    main()