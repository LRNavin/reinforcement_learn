import numpy as np
import sys


n_states = 63
n_action = 4
actions_set = ['L', 'R', 'U', 'D']


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

def visualize_Q_function(Q_s_a):
    return True

def get_next_state(current_state, current_action):
    next_state = 0

    return next_state

def get_i_j_for_indexmap(map, index):
    row_count = map.shape[0]
    col_count = map.shape[1]

    i = int(index/row_count)

    if i == 0:
        j = index
    else:
        index_temp = index - i
        j = index_temp/col_count

    return i, j

def get_state_reward(current_state):

    if current_state == 'G':
        next_reward = 1
    else:
        next_reward = 0

    return next_reward


def iterate_Q_value(Q_s_a, current_state, current_action, discount):

    next_state = get_next_state(current_state, current_action)

    max_Q = np.max([Q_s_a[action][next_state] for action in actions_set])

    reward = get_state_reward(next_state)

    Q_s_a[current_action][current_state] = reward + discount * max_Q

    return Q_s_a

# Main- Q Iteration Q_s_a - > Q of state,action pair
def Q_iteration(Q_s_a, discount):

    for i in range(n_states - 1):
        for action in actions_set:
            for state in range(n_states - 1):
                Q_s_a = iterate_Q_value(Q_s_a, state, action, discount)

    return Q_s_a


def compute_opt_policy(Q_star):
    pi_star = [0] * n_states

    for i in range(n_states):
        pi_star[i] = -1 if Q_star[-1][i] > Q_star[1][i] else 1

    return pi_star

def main():
    print("Setup Environment")

    map = np.asarray(STATE_MAP)

    print(str(map))

    #Init Q values for states & action
    Q_s_a = np.zeros((map.shape[0], map.shape[1], n_action))
    Q_s_a = Q_iteration(Q_s_a=Q_s_a, discount=0.5)

    #Visualize Result
    print("Discount - 0.5 : Exercise - 1")
    visualize_Q_function(Q_s_a=Q_s_a)

    #Optimal Policy - Comp
    optimal_policy = compute_opt_policy(Q_star=Q_s_a)
    print("Optimal policy: {}".format(optimal_policy))

    # Varying Discount factor
    for disc_fac in [.1, .5, .9]:
        print("Exercise - 1")
        print("Discount Factor: {}".format(disc_fac))
        Q_s_a = np.zeros((map.shape[0], map.shape[1], n_action))
        Q_s_a = Q_iteration(Q_s_a=Q_s_a, discount=disc_fac)

        optimal_policy = compute_opt_policy(Q_s_a)
        print("Optimal policy: {}".format(optimal_policy))


if __name__ == '__main__':
    main()