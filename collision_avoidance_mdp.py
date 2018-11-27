import time
import copy
import numpy as np

start_time = time.clock()


class Env:
    def __init__(self, size, obstacles, ends):
        self.size = size
        self.obstacles = obstacles
        self.ends = ends
        self.end_reward = -1
        self.obstacle_reward = -1
        self.move_reward = -1

    def set_move_reward(self, val):
        self.move_reward = val

    def set_end_reward(self, val):
        self.end_reward = val

    def set_obstacle_reward(self, val):
        self.obstacle_reward = val

    def get_reward(self, pos):
        [x, y] = pos
        reward = self.move_reward
        if [x, y] in self.obstacles:
            reward = self.obstacle_reward
        elif [x, y] in self.ends:
            reward = self.end_reward

        return reward


def get_next_state(state, move):
    directions = [[-1,  0],
                  [0,  1],
                  [1,  0],
                  [0, -1]]

    next_state = copy.deepcopy(state)
    direction = directions[move]

    next_state[0] = state[0] + direction[0]
    next_state[1] = state[1] + direction[1]

    # if the next state is going out of bounds
    # return the copy of the original state
    if next_state[0] >= s or next_state[0] < 0 or next_state[1] >= s or next_state[1] < 0:
        return copy.deepcopy(state)

    return next_state


def get_utilities(utility_grid, state, move):
    utilities = np.zeros(4)

    if move == 0:
        s = get_next_state(state, 0)
        utilities[0] = utility_grid[s[0], s[1]]

        s = get_next_state(state, 1)
        utilities[1] = utility_grid[s[0], s[1]]

        s = get_next_state(state, 3)
        utilities[2] = utility_grid[s[0], s[1]]

        s = get_next_state(state, 2)
        utilities[3] = utility_grid[s[0], s[1]]

    elif move == 1:
        s = get_next_state(state, 1)
        utilities[0] = utility_grid[s[0], s[1]]

        s = get_next_state(state, 0)
        utilities[1] = utility_grid[s[0], s[1]]

        s = get_next_state(state, 2)
        utilities[2] = utility_grid[s[0], s[1]]

        s = get_next_state(state, 3)
        utilities[3] = utility_grid[s[0], s[1]]

    elif move == 2:
        s = get_next_state(state, 2)
        utilities[0] = utility_grid[s[0], s[1]]

        s = get_next_state(state, 1)
        utilities[1] = utility_grid[s[0], s[1]]

        s = get_next_state(state, 3)
        utilities[2] = utility_grid[s[0], s[1]]

        s = get_next_state(state, 0)
        utilities[3] = utility_grid[s[0], s[1]]

    elif move == 3:
        s = get_next_state(state, 3)
        utilities[0] = utility_grid[s[0], s[1]]

        s = get_next_state(state, 0)
        utilities[1] = utility_grid[s[0], s[1]]

        s = get_next_state(state, 2)
        utilities[2] = utility_grid[s[0], s[1]]

        s = get_next_state(state, 1)
        utilities[3] = utility_grid[s[0], s[1]]

    return utilities


def one_step_lookahead(utility_grid, state):
    # 0 Up
    # 1 Right
    # 2 Down
    # 3 Left

    preference_moves = [0, 2, 3, 1]
    A = np.zeros(4)
    for a in range(4):
        utilities = get_utilities(utility_grid, state, preference_moves[a])
        # A[a] = np.float64(0.7 * utilities[0] + 0.1 * utilities[1] + 0.1 * utilities[2] + 0.1 * utilities[3])
        A[a] = 0.7 * utilities[0] + 0.1 * (utilities[1] + utilities[2] + utilities[3])
    return A


def value_iteration(size, obstacles, end, theta, gamma):
    # delta = 0
    utility_grid = np.full((size, size), 0.0)
    for ob in obstacles:
        utility_grid[ob[0], ob[1]] = -101
    utility_grid[end[0], end[1]] = 99

    while True:
        delta = 0

        # Update each state...
        for x in range(size):
            for y in range(size):
                if [x, y] == end:
                    continue

                # Do a one-step lookahead to find the best action
                A = one_step_lookahead(utility_grid, [x, y])
                best_action_value = np.max(A)

                # Update the value function
                reward = -1
                if [x, y] in obstacles:
                    reward = -101
                elif [x, y] == end:
                    reward = 99

                new_utility = reward + gamma * best_action_value

                # Calculate delta across all states seen so far
                delta = max(delta, np.abs(new_utility - utility_grid[x, y]))

                utility_grid[x, y] = new_utility

        if delta < theta:
            break


    preference_moves = [0, 2, 3, 1]
    policy = np.zeros([size, size], dtype=int)
    for x in range(size):
        for y in range(size):
            # One step lookahead to find the best action for this state
            A = one_step_lookahead(utility_grid, [x, y])

            # best_action = np.argmax(A)
            best_action_index = np.argmax(A)

            # Always take the best action
            policy[x, y] = int(preference_moves[best_action_index])

    # print("After Value Iteration: \n")
    # print(utility_grid)
    return policy


def turn_left(move):
    return (move + 3) % 4


def turn_right(move):
    return (move + 1) % 4


if __name__ == "__main__":
    s = -1
    n = -1
    o = -1

    obstacles = []
    starts = []
    ends = []
    policies = []

    with open('input2.txt') as inputFile:
        s = int(inputFile.readline().rstrip())
        n = int(inputFile.readline().rstrip())
        o = int(inputFile.readline().rstrip())

        reward_grid = np.full((s, s), -1)

        for i in range(o):
            line = inputFile.readline().rstrip()
            (x, y) = line.rstrip().split(",")
            obstacles.append([int(y), int(x)])
            reward_grid[int(y), int(x)] = -101

        for i in range(n):
            line = inputFile.readline().rstrip()
            (x, y) = line.rstrip().split(",")
            starts.append([int(y), int(x)])

        for i in range(n):
            line = inputFile.readline().rstrip()
            (x, y) = line.rstrip().split(",")
            reward_grid[int(y), int(x)] = 99
            ends.append([int(y), int(x)])

    # env = Env(s, obstacles, ends)
    # env.set_move_reward(-1)
    # env.set_end_reward(99)
    # env.set_obstacle_reward(-101)

    # print(reward_grid)
    # DEBUG value_iteration
    # n += 1
    # ends.append([0, 2])
    # starts.append([2, 2])

    for i in range(n):
        policy = value_iteration(s, obstacles, ends[i], 0.1, 0.9)
        # print(policy)
        policies.append(policy)

    # time is running out...
    # if time.clock() - start_time > 170:
    #     break
    p_run_time = time.clock() - start_time
    print("Time taken to create optimal policies: " + str(p_run_time) + " sec")

    # running the simulation
    mean_money = []
    for i in range(n):
        mean = 0
        for j in range(10):
            pos = starts[i]
            np.random.seed(j)
            swerve = np.random.random_sample(10000000)

            k = 0
            money = 0

            while pos != ends[i]:
                move = policies[i][pos[0], pos[1]]

                # deviate with some probability
                if swerve[k] > 0.7:
                    if swerve[k] > 0.8:
                        if swerve[k] > 0.9:
                            move = turn_right(turn_right(move))
                        else:
                            move = turn_right(move)
                    else:
                        move = turn_left(move)

                # else use the move from policy
                pos = get_next_state(pos, move)

                reward = -1
                if [pos[0], pos[1]] in obstacles:
                    reward = -101
                elif [pos[0], pos[1]] == ends[i]:
                    reward = 99

                money += reward

                k += 1

            mean += money
        mean /= 10
        mean_money.append(mean)
    print(mean_money)

    with open('output.txt', 'w') as outputFile:
        for m in mean_money:
            outputFile.write(str(m) + "\n")

    s_run_time = (time.clock() - start_time) - p_run_time
    print("Time taken to run simulation: " + str(s_run_time) + " sec")
