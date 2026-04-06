import grid_world
import random
import numpy as np

import grid_world
import random
import numpy as np

class Agent:
    def __init__(self, world: grid_world.Grid):
        self.world = world
        self.Vs = {}
        self.Qsa = {}
        self.policy = {}
        self.epsilon = 0.1
        self.gamma = 0.9
        self.alpha = 0.1

    def initValues(self):
        for i in range(self.world.rows):
            for j in range(self.world.cols):
                self.Vs[(i, j)] = 0.0

        for state, actions in self.world.actions.items():
            self.Qsa[state] = {}
            for action in actions:
                self.Qsa[state][action] = 0.0

            self.policy[state] = actions[0]

    def epsilonGreedy(self, state):
        if np.random.random() < self.epsilon:
            return random.choice(self.world.actions[state])
        return max(self.Qsa[state], key=self.Qsa[state].get)

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

    def sarsa(self, episodes=2000):
        for _ in range(episodes):
            s = self.world.reset()
            a = self.epsilonGreedy(s)

            while not self.world.is_terminal(s):
                r = self.world.move(a)
                s_next = self.world.current_state()

                if self.world.is_terminal(s_next):
                    target = r
                    self.Qsa[s][a] += self.alpha * (target - self.Qsa[s][a])
                    break

                a_next = self.epsilonGreedy(s_next)
                target = r + self.gamma * self.Qsa[s_next][a_next]

                self.Qsa[s][a] += self.alpha * (target - self.Qsa[s][a])

                s = s_next
                a = a_next

        # build greedy policy and V from learned Q
        for s in self.world.actions:
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

    agent.sarsa()

    print("---After---")
    agent.printVs()
    agent.printPolicy()

if __name__ == "__main__":
    main()