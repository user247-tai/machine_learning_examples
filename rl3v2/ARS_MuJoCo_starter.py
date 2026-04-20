import multiprocessing
from multiprocessing.pool import Pool
import numpy as np
import gymnasium as gym

def get_return_from_env_worker(args):
    """
    Worker function to evaluate a single set of parameters (one rollout).
    Standardizes observations using running mean/std provided by the main process.
    """
    params, input_size, deep_size, action_size, env_name, mean, std = args

    env = gym.make(env_name)
    net = ANN(input_size, deep_size, action_size)
    net.update_params(params)
    
    action_low = env.action_space.low.astype(np.float32)
    action_high = env.action_space.high.astype(np.float32)

    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    worker_observations = [obs.copy()]
    
    while not done:
        # Standardize observation: (x - mean) / std
        normalized_obs = (obs - mean) / std
        
        # Policy forward pass
        raw_action = net.forward(normalized_obs)
        action = action_low + (raw_action + 1.0) * 0.5 * (action_high - action_low)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        total_reward += reward
        worker_observations.append(obs.copy())

        done = terminated or truncated

    env.close()
    return total_reward, worker_observations

class OnlineStandardScaler():
    """
    Welford's Online Algorithm to track running mean and standard deviation.
    Ensures input features are normalized to Mean 0 and Variance 1.
    """
    def __init__(self, num_inputs):
        self.n = 0
        self.mean = np.zeros(num_inputs)
        self.ssd = np.zeros(num_inputs) 
        self.std = np.ones(num_inputs)

    def partial_fit(self, data_list):
        for X in data_list:
            self.n += 1
            delta = X - self.mean
            self.mean += delta / self.n
            delta2 = X - self.mean
            self.ssd += delta * delta2
        
        if self.n > 1:
            # Clip variance to prevent division by zero or extreme feature scaling
            variance = (self.ssd / self.n).clip(min=1e-2)
            self.std = np.sqrt(variance)

class Optimizer():
    """
    Basic SGD Optimizer. ARS often performs better with simple SGD 
    due to internal reward standardization (sigma_R).
    """
    def __init__(self, params, lr):
        self.lr = lr
        self.params = params

    def update(self, gradient):
        self.params += self.lr * gradient
        return self.params

class ANN():
    """
    Multi-Layer Perceptron (MLP) Policy.
    Structure: Input -> ReLU -> Hidden -> Tanh -> Output.
    """
    def __init__(self, input_shape, deep_shape, output_shape):
        self.input_shape = input_shape
        self.deep_shape = deep_shape
        self.output_shape = output_shape
        self.init_weights()

    def init_weights(self):
        # Xavier/Glorot Initialization
        self.w1 = np.random.randn(self.input_shape, self.deep_shape) / np.sqrt(self.input_shape)
        self.b1 = np.zeros(self.deep_shape)
        self.w2 = np.random.randn(self.deep_shape, self.output_shape) / np.sqrt(self.deep_shape)
        self.b2 = np.zeros(self.output_shape)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        x = self.relu((x @ self.w1) + self.b1) 
        x = np.tanh((x @ self.w2) + self.b2)
        return x

    def update_params(self, params):
        idx = 0
        w1_size = self.input_shape * self.deep_shape
        self.w1 = params[idx : idx + w1_size].reshape(self.input_shape, self.deep_shape)
        idx += w1_size
        self.b1 = params[idx : idx + self.deep_shape]
        idx += self.deep_shape
        w2_size = self.deep_shape * self.output_shape
        self.w2 = params[idx : idx + w2_size].reshape(self.deep_shape, self.output_shape)
        idx += w2_size
        self.b2 = params[idx:]

    def get_params(self):
        return np.concatenate([self.w1.flatten(), self.b1, self.w2.flatten(), self.b2])

class ARSAgent():
    """
    Augmented Random Search (ARS) Agent.
    Implements Mirror Sampling (positive/negative noise) and Elite Selection.
    """
    def __init__(self, env_name="HalfCheetah-v5"):
        self.env_name = env_name
        self.n_directions = 50   # Number of noise vectors (N)
        self.b = 25              # Number of top performing directions to keep (Elites)
        self.sigma = 0.03        # Noise standard deviation (exploration radius)
        self.lr = 0.02           # Learning rate
        self.pool_size = multiprocessing.cpu_count()

        temp_env = gym.make(self.env_name)
        self.obs_size = temp_env.observation_space.shape[0]
        self.act_size = temp_env.action_space.shape[0]
        temp_env.close()

        self.net = ANN(self.obs_size, 128, self.act_size)
        self.scaler = OnlineStandardScaler(self.obs_size)
        self.pool = Pool(self.pool_size)

    def train(self, iterations=1000):
        params = self.net.get_params()
        num_params = len(params)
        optimizer = Optimizer(params, self.lr)

        for t in range(iterations):
            # Generate N random directions
            eps = np.random.randn(self.n_directions, num_params)

            # Create mirror samples: theta + sigma*eps AND theta - sigma*eps
            # Total 2 * n_directions rollouts
            eval_params = []
            for i in range(self.n_directions):
                eval_params.append(params + self.sigma * eps[i])
                eval_params.append(params - self.sigma * eps[i])

            # Prepare arguments for parallel workers
            worker_args = [
                (p, self.obs_size, 128, self.act_size, self.env_name, self.scaler.mean, self.scaler.std)
                for p in eval_params
            ]
            
            results = self.pool.map(get_return_from_env_worker, worker_args)
            
            # Extract rewards and update observation scaler
            rewards = np.array([r[0] for r in results])
            all_obs = [obs for r in results for obs in r[1]]
            self.scaler.partial_fit(all_obs)

            # Pair rewards for Mirror Sampling comparison
            reward_pairs = []
            for i in range(self.n_directions):
                r_pos = rewards[2*i]
                r_neg = rewards[2*i + 1]
                reward_pairs.append((r_pos, r_neg, i))

            # Elite Selection: Sort directions by the maximum reward of either mirrored version
            reward_pairs.sort(key=lambda x: max(x[0], x[1]), reverse=True)
            elites = reward_pairs[: self.b]

            # Calculate Gradient using only the Elite directions
            # Standardizing by reward volatility (sigma_R) ensures stable updates
            all_elite_rewards = [r for pair in elites for r in (pair[0], pair[1])]
            sigma_R = np.std(all_elite_rewards) + 1e-8
            
            grad = np.zeros(num_params)
            for r_pos, r_neg, idx in elites:
                grad += (r_pos - r_neg) * eps[idx]
            
            grad /= (self.b * sigma_R)
            params = optimizer.update(grad)

            if t % 5 == 0:
                print(f"Iter: {t} | Max Reward: {np.max(rewards):.1f} | Avg: {np.mean(rewards):.1f} | Sigma_R: {sigma_R:.3f}")

        self.net.update_params(params)
        return params
    
    def save_model(self, filename="es_cheetah_model.npz"):
        """
        Saves the neural network parameters and scaler statistics to a .npz file.
        """
        np.savez(
            filename,
            params=self.net.get_params(),
            mean=self.scaler.mean,
            std=self.scaler.std,
            ssd=self.scaler.ssd,
            n=self.scaler.n,
        )
        print(f"Model saved to {filename}")

    def load_model(self, filename="es_cheetah_model.npz"):
        """
        Loads the neural network parameters and scaler statistics from a .npz file.
        """
        data = np.load(filename)
        self.net.update_params(data["params"])
        self.scaler.mean = data["mean"]
        self.scaler.std = data["std"]
        self.scaler.ssd = data["ssd"]
        self.scaler.n = int(data["n"])
        print(f"Model loaded from {filename}")

    def test(self, episodes=1, render=True):
        """
        Evaluates the current policy without noise.
        """
        # Create a separate env for testing, potentially with rendering
        test_env = gym.make(self.env_name, render_mode="human" if render else None)
        action_low = test_env.action_space.low.astype(np.float32)
        action_high = test_env.action_space.high.astype(np.float32)

        for e in range(episodes):
            obs, _ = test_env.reset()
            done = False
            total_reward = 0.0

            while not done:
                normalized_obs = (obs - self.scaler.mean) / self.scaler.std

                raw_action = self.net.forward(normalized_obs)
                action = action_low + (raw_action + 1.0) * 0.5 * (action_high - action_low)

                obs, reward, terminated, truncated, _ = test_env.step(action)
                total_reward += reward
                done = terminated or truncated

            print(f"Test Episode {e+1}: Total Reward = {total_reward:.2f}")

        test_env.close()

    def close(self):
        self.pool.close()
        self.pool.join()

def main():
    agent = ARSAgent()

    # train
    print("Starting training...")
    agent.train(iterations=1_000) 
    agent.save_model("cat.npz")

    # Load and test
    try:
        agent.load_model("cat.npz")
        print("Testing the loaded model...")
        agent.test(episodes=3, render=True)
    except FileNotFoundError:
        print("No saved model found. Please train first.")
    finally:
        agent.close()

if __name__ == "__main__":
    main()
    