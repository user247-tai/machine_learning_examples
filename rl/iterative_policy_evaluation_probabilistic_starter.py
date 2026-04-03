import grid_world

class Agent:
    def __init__(self, world: grid_world.WindyGrid):
        self.world = world
        self.Vs = {}
        self.Qsa = {}
        self.policy = {}
        self.delta_threshold = 0.001
        self.gamma = 0.9

    def initValues(self):
        for i in range(self.world.rows):
            for j in range(self.world.cols):
                self.Vs[(i, j)] = 0.0

        for state, actions in self.world.actions.items():
            self.Qsa[state] = {a: 0.0 for a in actions}

            # default uniform stochastic policy
            p = 1.0 / len(actions)
            self.policy[state] = {a: p for a in actions}

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
        cell_width = 12

        for i in range(self.world.rows):
            for j in range(self.world.cols):
                state = (i, j)

                if state not in self.policy:
                    text = ""
                else:
                    action_probs = self.policy[state]
                    parts = [f"{a}:{p:.1f}" for a, p in action_probs.items() if p > 0]
                    text = ",".join(parts)

                print(f"{text:^{cell_width}}", end=" ")
            print()

    def policyEvaluation(self):
        while True:
            delta = 0.0

            for s in self.world.all_states():
                if s not in self.world.actions:
                    continue

                old_v = self.Vs[s]
                new_v = 0.0

                # sum over actions from stochastic policy
                for a, pi_prob in self.policy[s].items():
                    q_sa = 0.0

                    # sum over next states from stochastic environment
                    for s_next, p_env in self.world.probs[(s, a)].items():
                        r = self.world.rewards.get(s_next, 0)
                        q_sa += p_env * (r + self.gamma * self.Vs[s_next])

                    self.Qsa[s][a] = q_sa
                    new_v += pi_prob * q_sa

                self.Vs[s] = new_v
                delta = max(delta, abs(old_v - new_v))

            self.printVs()
            print()

            if delta < self.delta_threshold:
                break


def main():
    standard_world = grid_world.windy_grid()
    agent = Agent(world=standard_world)
    agent.initValues()

    agent.policy = {
        (2, 0): {'U': 0.5, 'R': 0.5},
        (1, 0): {'U': 1.0},
        (0, 0): {'R': 1.0},
        (0, 1): {'R': 1.0},
        (0, 2): {'R': 1.0},
        (1, 2): {'U': 1.0},
        (2, 1): {'R': 1.0},
        (2, 2): {'U': 1.0},
        (2, 3): {'L': 1.0},
    }

    agent.printPolicy()
    print()
    agent.policyEvaluation()


if __name__ == "__main__":
    main()