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

    def initValues(self):
        # init V(s)
        for i in range(self.world.rows):
            for j in range(self.world.cols):
                self.Vs[(i, j)] = 0.0

        # init Q(s,a) + policy
        for s, actions in self.world.actions.items():
            self.Qsa[s] = {a: 0.0 for a in actions}
            self.policy[s] = random.choice(actions)

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
    # PLAY EPISODE (Exploring Starts)
    # ----------------------------------
    def playEpisode(self, max_steps=200):
        # random start state (non-terminal)
        while True:
            s = random.choice(list(self.world.all_states()))
            if s in self.world.actions:
                break

        # random start action
        a = random.choice(self.world.actions[s])

        self.world.set_state(s)

        states = [s]
        actions = [a]
        rewards = [0]

        # first step (exploring start)
        r = self.world.move(a)
        rewards.append(r)
        s = self.world.current_state()
        states.append(s)

        steps = 1

        # follow policy afterwards
        while not self.world.game_over():
            a = self.policy[s]

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
    # MONTE CARLO CONTROL (ES)
    # ----------------------------------
    def monteCarloControl(self, episodes=2000):
        # returns for (s,a)
        returns = {
            (s, a): []
            for s, actions in self.world.actions.items()
            for a in actions
        }

        for _ in range(episodes):
            rewards, states, actions = self.playEpisode()

            G = 0.0
            visited = set()

            # backward
            for t in range(len(states) - 2, -1, -1):
                s = states[t]
                a = actions[t]
                r = rewards[t + 1]

                G = r + self.gamma * G

                # first-visit for (s,a)
                if (s, a) not in zip(states[:t], actions[:t]):
                    visited.add((s, a))

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