import grid_world
import numpy as np
import random
from sklearn.kernel_approximation import RBFSampler

class Agent:
    def __init__(self, world: grid_world.Grid):
        self.world = world
        self.Vs = dict()
        self.Qsa = dict()
        self.policy = dict()
        self.epsilon = 0.1
        self.gamma = 0.9
        self.a = 0.1
        self.featurizer = RBFSampler()
        self.samples = self.getSamples()
        self.featurizer.fit(self.samples)
        self.dim = self.featurizer.n_components
        self.w = np.zeros(self.dim)
        
    def epsilonGreedy(self, state):
        random_num = np.random.random()
        if (random_num < self.epsilon):
            return random.choice(self.world.actions[state])
        else:
            return self.policy[state]

    def initValues(self):
        for i in range(self.world.rows):
            for j in range(self.world.cols):
                self.Vs[(i, j)] = 0.0

        for state, actions in self.world.actions.items():
            self.Qsa[state] = {}
            for action in actions:
                self.Qsa[state][action] = 0.0

            self.policy[state] = actions[0]

    def getSamples(self, n_samples = 10000):
        samples = []
        for _ in range(n_samples):
            s = self.world.reset()
            while not self.world.is_terminal(s):
                a = random.choice(self.world.actions[s])
                r = self.world.move(a)
                samples.append(s)
                s_next = self.world.current_state()
                s = s_next
            samples.append(s)

        return samples

    def printVs(self):
        print("=== Vs ===")
        for i in range(self.world.rows):
            for j in range(self.world.cols):
                print(f"{self.Vs[(i, j)]:6.2f}", end=" ")
            print()

    def printQsa(self):
        print("=== Qsa ===")
        for state, action_values in self.Qsa.items():
            print(f"State {state}:")
            for action, q_value in action_values.items():
                print(f"  {action} -> {q_value:.2f}")

    def printPolicy(self):
        print("=== Policy ===")
        for i in range(self.world.rows):
            for j in range(self.world.cols):
                state = (i, j)
                if state in self.policy:
                    print(f"{self.policy[state]:^6}", end=" ")
                else:
                    print(f"{'':^6}", end=" ")
            print()

    def predict(self, state):
        x = self.featurizer.transform([state])[0]
        return x @ self.w

    def grad(self, state):
        x = self.featurizer.transform([state])[0]
        return x

    def td0Prediction(self, episodes = 10000):
            for _ in range(episodes): 
                s = self.world.reset()
                while not self.world.is_terminal(s):
                    a = self.epsilonGreedy(s)
                    r = self.world.move(a)
                    s_next = self.world.current_state()
                    
                    if self.world.is_terminal(s_next):
                        target = r
                    else:
                        target = r + self.gamma * self.predict(s_next)
                    
                    # w = w + alpha * (target - predict) * gradient
                    self.w += self.a * (target - self.predict(s) ) * self.grad(s)

                    s = s_next

            for i in range(self.world.rows):
                for j in range(self.world.cols):
                    state = (i, j)
                    self.Vs[state] = self.predict(state)
        

def main():
    standard_world = grid_world.standard_grid()
    agent = Agent(world=standard_world)
    agent.initValues()
    
    agent.policy = {
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',

        (1, 0): 'U',
        (1, 2): 'R',

        (2, 0): 'U',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U',
    }
    
    print("---Before---")
    agent.printVs()
    agent.printPolicy()

    agent.td0Prediction()

    print("---After---")
    agent.printVs()
    agent.printPolicy()

if __name__ == "__main__":
    main()