import grid_world

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

            # pick first action as default policy
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

import grid_world

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

    def policyEvaluation(self):
        while True:
            delta = 0.0

            for s in self.world.all_states():
                # skip terminal states
                if s not in self.world.actions:
                    continue

                old_v = self.Vs[s]

                # deterministic policy: pick only one action
                a = self.policy[s]
                print(f"Setting state: {s}")
                # simulate one step from s using action a
                self.world.set_state(s)
                r = self.world.move(a)
                s_next = self.world.current_state()

                new_v = r + self.gamma * self.Vs[s_next]
                self.Vs[s] = new_v

                delta = max(delta, abs(old_v - new_v))

            self.printVs()
            if delta < self.delta_threshold:
                break

def main():
    standard_world = grid_world.standard_grid()
    agent = Agent(world=standard_world)
    agent.initValues()
    agent.policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'U',
        (2, 1): 'R',
        (2, 2): 'U',
        (2, 3): 'L',
    }
    agent.policyEvaluation()


if __name__ == "__main__":
    main()