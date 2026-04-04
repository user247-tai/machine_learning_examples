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
        # init V(s)
        for i in range(self.world.rows):
            for j in range(self.world.cols):
                self.Vs[(i, j)] = 0.0

        # init Q(s,a) + random deterministic policy
        for state, actions in self.world.actions.items():
            self.Qsa[state] = {a: 0.0 for a in actions}

            # chọn đại 1 action làm policy ban đầu
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

    def valueIteration(self):
        while True:
            delta = 0.0
            for s in self.world.all_states():
                if s not in self.world.actions:
                    continue
                v_old = self.Vs[s]

                for a in self.world.actions[s]:
                    q = 0.0
                    for s_next, p_env in self.world.probs[(s, a)].items():
                        r = self.world.rewards.get(s_next, 0)
                        q += p_env * (r + self.gamma * self.Vs[s_next])
                    
                    self.Qsa[s][a] = q
                        
                self.Vs[s] = max(self.Qsa[s].values())
                delta = max(delta, abs(v_old - self.Vs[s]))
                
            self.printVs()
            if delta < self.delta_threshold:
                break

        for s in self.world.all_states():
            if s not in self.world.actions:
                continue 
            self.policy[s] = max(self.Qsa[s], key=self.Qsa[s].get)
                    
# -----------------------------
# MAIN
# -----------------------------
def main():
    world = grid_world.windy_grid()

    agent = Agent(world)
    agent.initValues()

    # policy khởi tạo (deterministic)
    agent.policy = {
        (0, 0): 'D',
        (0, 1): 'L',
        (0, 2): 'U',

        (1, 0): 'D',
        (1, 2): 'U',

        (2, 0): 'U',
        (2, 1): 'D',
        (2, 2): 'D',
        (2, 3): 'L',
    }

    print("Initial Policy:")
    agent.printPolicy()
    print("------------")

    agent.valueIteration()

    print("Final Policy:")
    agent.printPolicy()
    print("------------")

    # agent.printVs()
    # print("------------")

    # agent.printQsa()


if __name__ == "__main__":
    main()