import numpy as np
import matplotlib.pyplot as plt

#TODO:
# 1. Funciton to check current Optimal policy & equivalent loss/reward
# 2. Check optimal reward @ each episode


#NOTE:
# 1. Ex4 - 2 methods & their combination ?

n_episodes = 100
n_states = 63
n_action = 4
actions_set = [0 ,1 ,2 ,3] # Eq to - ['L', 'R', 'U', 'D']
actions_print = [u'\N{BLACK LEFT-POINTING TRIANGLE}' ,
                 u'\N{BLACK RIGHT-POINTING TRIANGLE}' ,
                 u'\N{BLACK UP-POINTING TRIANGLE}' ,
                 u'\N{BLACK DOWN-POINTING TRIANGLE}'] # Eq to - ['L', 'R', 'U', 'D']

actions_move = ['L' ,
                'R' ,
                'U' ,
                'D' ]

# Start Position -
x_start = 7
y_start = 5

# Map Legend ##
# G - Termina/Goal
# S - Start
# O - Obsatcles
# P - Path
###############

STATE_MAP = [['O', 'O', 'O', 'O', 'O', 'O', 'O'],
             ['O', 'P', 'P', 'G', 'P', 'P', 'O'],
             ['O', 'P', 'O', 'O', 'P', 'O', 'O'],
             ['O', 'P', 'P', 'P', 'P', 'P', 'O'],
             ['O', 'P', 'O', 'O', 'O', 'P', 'O'],
             ['O', 'P', 'P', 'P', 'P', 'P', 'O'],
             ['O', 'P', 'O', 'O', 'P', 'O', 'O'],
             ['O', 'P', 'P', 'P', 'P', 'S', 'O'],
             ['O', 'O', 'O', 'O', 'O', 'O', 'O'],]

def visualize_Q_function(Q_s_a, title=""):
    opt_Q, opt_act = calculate_optimal_Q(Q_s_a)
    heatmap = plt.pcolor(opt_Q)
    plt.colorbar(heatmap)
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.show()
    return True

def visualize_Optimal_Policy(Q_s_a, title=""):

    map = np.array(STATE_MAP)

    opt_Q, opt_act = calculate_optimal_Q(Q_s_a)
    print opt_act
    print_array = np.chararray(opt_act.shape)

    for x in range(opt_act.shape[0]):
        for y in range(opt_act.shape[1]):
            if map[x][y] == 'O':
                print_array[x][y] = '*'
            elif map[x][y] == 'G':
                print_array[x][y] = 'G'
            else:
                print_array[x][y] = actions_move[opt_act[x][y]]

    print print_array

    return True

def calculate_optimal_Q(Q_s_a):
    return np.max(Q_s_a, axis=2), np.argmax(Q_s_a, axis=2)

def calculate_optimal_policy_value(Q_s_a):
    map = np.array(STATE_MAP)

    goal_not_reached = True
    value = 0
    opt_Q, opt_act  = calculate_optimal_Q(Q_s_a)

    current_x = x_start
    current_y = y_start

    while(goal_not_reached):
        action = opt_act[current_x][current_y]
        value = value + opt_Q[current_x][current_y]

        current_x, current_y = get_next_state(current_x, current_y, action)
        if map[current_x][current_y] == 'G':
            goal_not_reached = False

    return value

def get_next_state(state_x, state_y, current_action):
    map = np.asarray(STATE_MAP)

    next_state_x = state_x
    next_state_y = state_y

    # Right - x=same , y=+1 -> 1
    # Left  - x=same , y=-1 -> 0
    # Up    - x=-1   , y=same -> 2
    # Down  - x=+1   , y=same -> 3

    if current_action == 0:# Left
        next_state_y = next_state_y - 1
    elif current_action == 1:# Right
        next_state_y = next_state_y + 1
    elif current_action == 2:# Up
        next_state_x = next_state_x - 1
    else:# Down
        next_state_x = next_state_x + 1

    if map[next_state_x][next_state_y] == 'O':
        next_state_x = state_x
        next_state_y = state_y

    return next_state_x, next_state_y


def get_state_reward(state_x, state_y):
    map = np.asarray(STATE_MAP)

    current_state = map[state_x][state_y]
    if current_state == 'G':
        next_reward = 1
    else:
        next_reward = 0

    return next_reward


def iterate_Q_value(Q_s_a, state_x, state_y, current_action, discount):
    map = np.array(STATE_MAP)

    if map[state_x][state_y] != 'G':
        next_state_x, next_state_y = get_next_state(state_x, state_y, current_action)
        max_Q = np.max([Q_s_a[next_state_x][next_state_y][action] for action in actions_set])
        reward = get_state_reward(next_state_x, next_state_y)

        # Bellman Equation
        Q_s_a[state_x][state_y][current_action] = reward + discount * max_Q

    return Q_s_a

# Main- Q Iteration Q_s_a - > Q of state,action pair
def Q_iteration(Q_s_a, discount):
    map = np.asarray(STATE_MAP)
    no_sweep = 0
    for i in range(n_episodes): # For No.of Episodes / Iters
        print "Iteration---- " + str(i) + "----"
        Q_prev = np.copy(Q_s_a)
        for x in range(map.shape[0] ): # For No.of Rows in Map
            for y in range(map.shape[1] ): # For No.of Columns in Map
                for action in actions_set: # No.of Possible Actions
                    if map[x][y] != 'O':
                        Q_s_a = iterate_Q_value(Q_s_a, x, y, action, discount)
                # print "x - " + str(x) + ", y - " + str(y) + ", Q_Val - " + str(Q_s_a[x][y])
        no_sweep = no_sweep + 1
        if np.array_equal(Q_prev, Q_s_a):
            print "SAME - CONVERGENCE : " + str(no_sweep)
            break

    return Q_s_a

def main():
    print("Setup Environment")

    map = np.asarray(STATE_MAP)

    print(str(map))

    print("Discount - 0.9 : Exercise - 1")
    #Init Q values for states & action
    Q_s_a = np.zeros((map.shape[0], map.shape[1], n_action))
    Q_s_a = Q_iteration(Q_s_a=Q_s_a, discount=0.9)

    #Visualize Result
    visualize_Q_function(Q_s_a=Q_s_a, title="Value Function: Discount Rate - " + str(0.9))
    print "Optimal Policy Value - " + str(calculate_optimal_policy_value(Q_s_a=Q_s_a))

    # Visualize Optimal Fn..
    visualize_Optimal_Policy(Q_s_a=Q_s_a, title="Q-iteration Optimal Policy")

    #Optimal Policy - Comp
    optimal_policy, opt_act  = calculate_optimal_Q(Q_s_a=Q_s_a)
    # Save Q_s_a for QLeanirng use
    policy = np.asarray(optimal_policy)
    np.savetxt("Q_s_a.csv", policy, delimiter=",")

    # print("Optimal policy: {}".format(optimal_policy))

    # Varying Discount factor
    for disc_fac in [0, .1, .5, 1.0]:
        print("Exercise - 2")
        print("Discount Factor: {}".format(disc_fac))
        Q_s_a = np.zeros((map.shape[0], map.shape[1], n_action))
        Q_s_a = Q_iteration(Q_s_a=Q_s_a, discount=disc_fac)

        # Visualize Result
        visualize_Q_function(Q_s_a=Q_s_a, title="Value Function: Discount Rate - " + str(disc_fac))


        optimal_policy, opt_act  = calculate_optimal_Q(Q_s_a)
        # print("Optimal policy: {}".format(optimal_policy))


if __name__ == '__main__':
    main()