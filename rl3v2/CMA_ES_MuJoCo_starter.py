import multiprocessing
from multiprocessing.pool import Pool
import numpy as np
import gymnasium as gym
import cma

def get_return_from_env_worker(args):
    """
    Worker function to evaluate a single member of the population (an 'offspring').
    Each worker runs a full episode in its own environment instance.
    """
    # Unpack arguments including global mean and std for consistent observation scaling
    params, input_size, deep_size, action_size, env_name, mean, std = args

    # Initialize environment and neural network for this specific rollout
    env = gym.make(env_name)
    net = ANN(input_size, deep_size, action_size)
    net.update_params(params)
    
    # Retrieve action limits once to avoid repeated overhead
    action_max = env.action_space.high.astype(np.float32)

    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    
    # Collect observations to update the global scaler in the main process
    worker_observations = []

    while not done:
        # Standardize observation using global statistics from the main process
        normalized_obs = (obs - mean) / std
        
        # Forward pass and rescale output to environment's action range
        action = net.forward(normalized_obs) * action_max
        obs, reward, terminated, truncated, _ = env.step(action)
        
        total_reward += reward
        worker_observations.append(obs) 
        done = terminated or truncated

    env.close()
    return total_reward, worker_observations

class OnlineStandardScaler():
    """
    Tracks running mean and standard deviation of observations 
    to ensure input features are on a similar scale (Mean 0, Var 1).
    """
    def __init__(self, num_inputs):
        self.n = 0
        self.mean = np.zeros(num_inputs)
        self.ssd = np.zeros(num_inputs) # Sum of Squared Deviations
        self.std = np.ones(num_inputs)

    def partial_fit(self, data_list):
        """Updates internal statistics using a batch of observations."""
        for X in data_list:
            self.n += 1
            delta = X - self.mean
            self.mean += delta / self.n
            delta2 = X - self.mean
            self.ssd += delta * delta2
        
        if self.n > 1:
            # Clip variance to prevent division by zero or extreme scaling
            variance = (self.ssd / self.n).clip(min=1e-2)
            self.std = np.sqrt(variance)

class ANN():
    """
    Simple MLP Policy: Input -> ReLU(Linear) -> Tanh(Linear) -> Output.
    """
    def __init__(self, input_shape, deep_shape, output_shape):
        self.input_shape = input_shape
        self.deep_shape = deep_shape
        self.output_shape = output_shape
        self.init_weights()

    def init_weights(self):
        """Xavier (Glorot) Initialization to keep signal variance stable."""
        self.w1 = np.random.randn(self.input_shape, self.deep_shape) / np.sqrt(self.input_shape)
        self.b1 = np.zeros(self.deep_shape)
        self.w2 = np.random.randn(self.deep_shape, self.output_shape) / np.sqrt(self.deep_shape)
        self.b2 = np.zeros(self.output_shape)

    def relu(self, x):
        return x * (x > 0)

    def forward(self, x):
        x = self.relu((x @ self.w1) + self.b1) 
        x = np.tanh((x @ self.w2) + self.b2)
        return x

    def update_params(self, params):
        """Unflattens a 1D parameter vector back into matrix weights/biases."""
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
        """Flattens all weights and biases into a single 1D array."""
        return np.concatenate([self.w1.flatten(), self.b1, self.w2.flatten(), self.b2])

class ESAgent():
    """
    Evolution Strategies Agent using Parallel Rollouts and Adam optimization.
    """
    def __init__(self):
        self.env_name = "HalfCheetah-v5"
        self.population_size = 30
        self.sigma = 0.05       # Noise standard deviation (exploration radius)
        self.learning_rate = 0.01 
        self.pool_size = multiprocessing.cpu_count() # Utilize all available CPU cores

        # Briefly initialize env to determine observation and action space sizes
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
        options = {
            # 'popsize': self.population_size,
            'AdaptSigma': True,
            'maxiter': iterations,
            'verb_disp': 1,
            'CMA_diagonal': False, #full cov matrix
            #'CMA_mirrors': True # use f(params + noise) and f(params - noise) to estimate
        }
        reward_per_it = []
        es = cma.CMAEvolutionStrategy(params, self.sigma, options)
        
        while not es.stop():
            # Generate random perturbations (noise)
            solutions = es.ask()

            worker_args = [
                (sol, self.obs_size, 128, self.act_size, self.env_name, self.scaler.mean, self.scaler.std)
                for sol in solutions
            ]
            
            # Perform parallel rollouts
            results = self.pool.map(get_return_from_env_worker, worker_args)
            
            # Separate rewards and new observations collected by workers
            rewards = np.array([r[0] for r in results])
            all_obs = [obs for r in results for obs in r[1]]
            
            # Update global scaler with observations from this generation
            self.scaler.partial_fit(all_obs)

            es.tell(solutions, -rewards)
            es.logger.add()
            reward_per_it.append(rewards.mean())

            # Log actual performance metrics
            print(f"Iter: {es.countiter} | Avg Reward: {rewards.mean():.2f} | Max: {rewards.max():.2f}")

        return es.best.get()[0]
    
    def save_model(self, filename="es_cheetah_model.npz"):
            """
            Saves the neural network parameters and scaler statistics to a .npz file.
            """
            params = self.net.get_params()
            np.savez(filename, 
                    params=params, 
                    mean=self.scaler.mean, 
                    std=self.scaler.std)
            print(f"Model saved to {filename}")

    def load_model(self, filename="es_cheetah_model.npz"):
        """
        Loads the neural network parameters and scaler statistics from a .npz file.
        """
        data = np.load(filename)
        self.net.update_params(data['params'])
        self.scaler.mean = data['mean']
        self.scaler.std = data['std']
        # Crucial: update the 'n' count or set a flag so scaler doesn't overwrite immediately
        self.scaler.n = 1000 
        print(f"Model loaded from {filename}")

    def test(self, episodes=1, render=True):
        """
        Evaluates the current policy without noise.
        """
        # Create a separate env for testing, potentially with rendering
        test_env = gym.make(self.env_name, render_mode="human" if render else None)
        action_max = test_env.action_space.high.astype(np.float32)

        for e in range(episodes):
            obs, _ = test_env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # Use the learned global mean and std to normalize test observations
                normalized_obs = (obs - self.scaler.mean) / self.scaler.std
                
                # Pure forward pass (no noise added)
                action = self.net.forward(normalized_obs) * action_max
                obs, reward, terminated, truncated, _ = test_env.step(action)
                
                total_reward += reward
                done = terminated or truncated
                
            print(f"Test Episode {e+1}: Total Reward = {total_reward:.2f}")
        
        test_env.close()

def main():
    agent = ESAgent()

    # train
    print("Starting training...")
    final_params = agent.train(iterations=300) 
    agent.net.update_params(final_params)
    agent.save_model("cheetah_cma.npz")

    # Load and test
    try:
        agent.load_model("cheetah_cma.npz")
        print("Testing the loaded model...")
        agent.test(episodes=3, render=True)
    except FileNotFoundError:
        print("No saved model found. Please train first.")

if __name__ == "__main__":
    main()
    