import grid_world
import random
import numpy as np

class Agent:
    def __init__(self, world: grid_world.Grid):
        self.world = world
        self.Vs = dict()
        self.Qsa = dict()
        self.policy = dict()
        self.delta_threshold = 0.001
        self.gamma = 0.9

    def initValues(self):
        for i in range(self.world.rows):
            for j in range(self.world.cols):
                self.Vs[(i, j)] = 0.0

        for state, actions in self.world.actions.items():
            self.Qsa[state] = {}
            for action in actions:
                self.Qsa[state][action] = 0.0

            self.policy[state] = actions[0]

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

    def playEpisode(self, max_steps = 200):
        # Choose random initial state (make sure initial state is not terminal state)
        random_initial_state = None
        while (random_initial_state is None) or (self.world.is_terminal(random_initial_state)):
            random_initial_state = random.choice(list(self.world.all_states()))

        self.world.set_state(random_initial_state) # Reset the game
        step = 1
        reward_list = [0]
        state_list = [random_initial_state]        

        while not self.world.game_over():
            current_state = self.world.current_state()
            reward = self.world.move(self.policy[current_state])
            reward_list.append(reward)
            next_state = self.world.current_state()
            state_list.append(next_state)
            step += 1
            if step >= max_steps:
                break

        return reward_list, state_list
    
    def monteCarloPrediction(self, samples=2000):
        for s in self.world.all_states():
            self.Vs[s] = 0.0

        returns = {s: [] for s in self.world.actions}

        for _ in range(samples):
            rewards, states = self.playEpisode()

            G = 0.0

            for t in range(len(states) - 2, -1, -1):
                s = states[t]
                r = rewards[t + 1]

                G = r + self.gamma * G

                if s not in states[:t]:  
                    returns[s].append(G)
                    self.Vs[s] = np.mean(returns[s])


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

    print("Before prediction: ")
    agent.printVs()
    print("---")
    agent.printPolicy()
    print("After prediction: ")
    agent.monteCarloPrediction()
    agent.printVs()


if __name__ == "__main__":
    main()