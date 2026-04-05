import grid_world
import random
import numpy as np


class Agent:
    def __init__(self, world: grid_world.Grid):
        self.world = world
        self.Vs = {}
        self.Qsa = {}
        self.policy = {}
        self.gamma = 0.9
        self.epsilon = 0.1

    # ----------------------------------
    # INIT
    # ----------------------------------
    def initValues(self):
        for i in range(self.world.rows):
            for j in range(self.world.cols):
                self.Vs[(i, j)] = 0.0

        for s, actions in self.world.actions.items():
            self.Qsa[s] = {a: 0.0 for a in actions}
            self.policy[s] = random.choice(actions)

    # ----------------------------------
    # PRINT
    # ----------------------------------
    def printVs(self):
        print("=== Vs ===")
        for i in range(self.world.rows):
            for j in range(self.world.cols):
                print(f"{self.Vs[(i, j)]:6.2f}", end=" ")
            print()

    def printPolicy(self):
        print("=== Policy ===")
        for i in range(self.world.rows):
            for j in range(self.world.cols):
                s = (i, j)
                if s in self.policy:
                    print(f"{self.policy[s]:^6}", end=" ")
                else:
                    print(f"{'':^6}", end=" ")
            print()

    # ----------------------------------
    # EPSILON-GREEDY (from deterministic policy)
    # ----------------------------------
    def epsilonGreedy(self, state):
        if np.random.random() < self.epsilon:
            return random.choice(self.world.actions[state])
        else:
            return self.policy[state]

    # ----------------------------------
    # PLAY EPISODE (NO EXPLORING STARTS)
    # ----------------------------------
    def playEpisode(self, max_steps=200):
        # reset environment
        s = self.world.reset()

        states = [s]
        actions = []
        rewards = [0]

        steps = 0

        while not self.world.game_over():
            a = self.epsilonGreedy(state=s)

            actions.append(a)
            r = self.world.move(a)

            rewards.append(r)

            s = self.world.current_state()
            states.append(s)

            steps += 1
            if steps >= max_steps:
                break

        return rewards, states, actions

    # ----------------------------------
    # MONTE CARLO CONTROL (ε-greedy)
    # ----------------------------------
    def monteCarloControl(self, episodes=5000):
        # returns for (s,a)
        returns = {
            (s, a): []
            for s, actions in self.world.actions.items()
            for a in actions
        }

        for _ in range(episodes):
            rewards, states, actions = self.playEpisode()

            G = 0.0

            # backward
            for t in range(len(states) - 2, -1, -1):
                s = states[t]
                a = actions[t]
                r = rewards[t + 1]

                G = r + self.gamma * G

                if (s, a) not in zip(states[:t], actions[:t]):
                    returns[(s, a)].append(G)
                    self.Qsa[s][a] = np.mean(returns[(s, a)])

            # policy improvement (greedy)
            for s in self.world.actions:
                self.policy[s] = max(self.Qsa[s], key=self.Qsa[s].get)

        # compute V(s)
        for s in self.world.actions:
            self.Vs[s] = max(self.Qsa[s].values())


# ----------------------------------
# MAIN
# ----------------------------------
def main():
    world = grid_world.standard_grid()

    agent = Agent(world)
    agent.initValues()

    print("Initial Policy:")
    agent.printPolicy()
    print("-----")

    agent.monteCarloControl(episodes=5000)

    print("Final Policy:")
    agent.printPolicy()
    print("-----")

    print("State Values:")
    agent.printVs()


if __name__ == "__main__":
    main()