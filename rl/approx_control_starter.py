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
        if np.random.random() < self.epsilon:
            return random.choice(self.world.actions[state])
        else:
            # Tìm action có Q(s,a) cao nhất dựa trên model w hiện tại
            qs = {a: self.predict(state, a) for a in self.world.actions[state]}
            return max(qs, key=qs.get)
        
    def one_hot_action(self, action: str) -> list[int]:
        actions = ['U', 'D', 'L', 'R']
        if action not in actions:
            raise ValueError(f"Invalid action: {action}. Must be one of {actions}")

        encoding = [0] * len(actions)
        encoding[actions.index(action)] = 1
        return encoding

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
                a_encoding = self.one_hot_action(a)
                sample = s + tuple(a_encoding)
                samples.append(sample)
                s_next = self.world.current_state()
                s = s_next

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

    def predict(self, state, action):
        a_encoding = self.one_hot_action(action)
        data = state + tuple(a_encoding)
        x = self.featurizer.transform([data])[0]
        return x @ self.w

    def grad(self, state, action):
        a_encoding = self.one_hot_action(action)
        data = state + tuple(a_encoding)
        x = self.featurizer.transform([data])[0]
        return x

    def q_learning(self, episodes = 10000):
        for _ in range(episodes): 
            s = self.world.reset()
            while not self.world.is_terminal(s):
                a = self.epsilonGreedy(s)
                r = self.world.move(a)
                s_next = self.world.current_state()
                
                if self.world.is_terminal(s_next):
                    target = r
                else:
                    # Q-Learning: target = r + gamma * max_a' Q(s_next, a')
                    qs_next = [self.predict(s_next, a_n) for a_n in self.world.actions[s_next]]
                    target = r + self.gamma * max(qs_next)
                
                # w = w + alpha * (target - predict) * gradient
                self.w += self.a * (target - self.predict(s, a) ) * self.grad(s, a)
                self.Qsa[s][a] = self.predict(s, a)

                s = s_next

        for s in self.world.actions:
            for a in self.world.actions[s]:
                self.Qsa[s][a] = self.predict(s, a) 
            
            best_action = max(self.Qsa[s], key=self.Qsa[s].get)
            self.policy[s] = best_action
            self.Vs[s] = self.Qsa[s][best_action]
        

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

    agent.q_learning()

    print("---After---")
    agent.printVs()
    agent.printPolicy()

if __name__ == "__main__":
    main()