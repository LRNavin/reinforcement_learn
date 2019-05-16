import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import math

# n_episodes = 100
n_states = 63
n_action = 4
actions_set = [0 ,1 ,2 ,3] # EQ to - ['L', 'R', 'U', 'D']

plot_errors_dict={}

# q-learning improvement flags
# PROGRESS_REWARD = 0
# GOAL_BIAS_INIT  = 1

beta = 10
sigma = 2
Q_i = 100

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

map=np.array(STATE_MAP)

Q_s_a = genfromtxt('Q_s_a.csv', delimiter=',')

def visualize_Q_function(Q_s_a, title=""):
    opt_Q = calculate_optimal_Q(Q_s_a)
    heatmap = plt.pcolor(opt_Q)
    plt.colorbar(heatmap)
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.show()
    return True

def calculate_optimal_Q(Q_s_a):
    return np.max(Q_s_a, axis=2)

def calculate_optimal_Q_action(Q_s_a):
    return np.argmax(Q_s_a, axis=2)


def find_optimal_number_of_steps(Q_learn):
    map = np.asarray(STATE_MAP)
    current_x = 7
    current_y = 5

    optimal_Q_act = calculate_optimal_Q_action(Q_learn)
    total_steps = 0

    while(map[current_x][current_y] != 'G' and total_steps < 200):

        action = optimal_Q_act[current_x][current_y]
        current_x, current_y = get_next_state(current_x, current_y, action)
        total_steps = total_steps + 1

    return total_steps

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


def iterate_episode(Q_learn, current_x, current_y, epsilon, discount, alpha, interaction_error, GOAL_BIAS_INIT, PROGRESS_REWARD):

    map= np.array(STATE_MAP)
    while map[current_x][current_y] != 'G':

        if not GOAL_BIAS_INIT:
            # with prob - epsilon  -> pick rand action
            random_ind = np.random.random()
            if random_ind > epsilon:
                current_action = np.argmax(Q_learn[current_x][current_y])
            else:
                current_action = np.random.randint(0, len(actions_set))
        else:
            optimal_Q_act  = calculate_optimal_Q_action(Q_learn)
            current_action = optimal_Q_act[current_x][current_y]

        next_state_x, next_state_y = get_next_state(current_x, current_y, current_action)


        # if PROGRESS_REWARD:
        #     reward = progress_est_reward_function(state_x=next_state_x, state_y=next_state_y, beta_curr=10, sigma_curr=3.5)
        # else:
        reward = get_state_reward(next_state_x, next_state_y)

        max_Q = np.max([Q_learn[next_state_x][next_state_y][action] for action in actions_set]) # Next State
        temp_diff = discount * max_Q - Q_learn[current_x][current_y][current_action] # Current State Diff

        Q_learn[current_x][current_y][current_action] += (alpha * (reward + temp_diff))

        current_x, current_y = next_state_x, next_state_y

        # E-for every step
        e = np.linalg.norm(Q_s_a - calculate_optimal_Q(Q_learn), ord=2)
        # No.of steps to goal
        # e = find_optimal_number_of_steps(Q_learn=Q_learn)

        interaction_error.append(e)

    return Q_learn, interaction_error


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def Q_learn_learning(Q_learn, epsilon, discount, alpha, GOAL_BIAS_INIT, n_episodes, PROGRESS_REWARD):
    map = np.array(STATE_MAP)

    interaction_error = []

    for t,r in enumerate(range(n_episodes)):
        # print "*********** Episode No: " + str(i) + "***********"
        Q_prev = Q_learn
        not_valid_start = True

        if PROGRESS_REWARD:
            # if t != 0:
            epsilon = epsilon/t

        while(not_valid_start):
            x_start = np.random.randint(0, map.shape[0])
            y_start = np.random.randint(0, map.shape[1])

            if(map[x_start][y_start] != 'O'):
                not_valid_start = False
            else:
                # E-for every step
                e = np.linalg.norm(Q_s_a - calculate_optimal_Q(Q_learn), ord=2)
                interaction_error.append(e)

        Q_learn, interaction_error = iterate_episode(Q_learn, x_start, y_start, epsilon, discount, alpha, interaction_error, GOAL_BIAS_INIT, PROGRESS_REWARD)

        # print Q_s_a
        # print calculate_optimal_Q(Q_prev)

        # if np.array_equal(calculate_optimal_Q(Q_prev), Q_s_a):
        #     print "SAME - CONVERGENCE : " + str(i)
        #     break

    return Q_learn, interaction_error

def visualize_norm_diff_plot(plot_errors_dict, alphas):
    legend_array=[]
    for k, v in plot_errors_dict.items():
        # print(k, v)
        legend_array.append(str(k))

        plt.plot(v)

    legends = [str(alpha) for alpha in legend_array]
    plt.legend(legends)
    plt.xlabel("# of Interactions")
    plt.ylabel(r"$||Q_{learn} - Q_{iter}||_2$")
    plt.title("L2 distance between value function from  $Q_{learn}$" + " and " + "$Q_{iter}$ ")
    plt.grid()
    plt.savefig('figures/q_est_error.eps', dpi=300)
    plt.show()


# Improving Q-Learning - With Q-value Init  and  Progress Based REward function

def progress_est_reward_function(state_x, state_y, beta_curr, sigma_curr):
    # map = np.asarray(STATE_MAP)
    #r(s,a,s') = -d(s`,s_g)^2
    # abs(x1 - x2) + abs(y1 - y2)

    goal_x = 1
    goal_y = 3
    man_dist = (-1 * np.square((np.abs(goal_x - state_x) + np.abs(goal_y - state_y)))) / (2 * np.square(sigma_curr))

    man_dist = beta_curr * np.exp(man_dist)

    return man_dist

def goal_based_Q_init(Q_learn):

    map = np.asarray(STATE_MAP)

    for x in range(Q_learn.shape[0]):
        for y in range(Q_learn.shape[1]):
            if map[x][y] != 'O' and map[x][y] != 'G':
                for a in actions_set:
                    next_state_x, next_state_y = get_next_state(x, y, a)
                    Q_learn[x][y][a] = progress_est_reward_function(next_state_x, next_state_y, 1, 2.5)

    return Q_learn



def main():
    print("Setup Environment")
    map = np.asarray(STATE_MAP)
    print(str(map))

    # Algo params
    epsilons = np.array([0.9,0.9])#np.array([.01, .05, 0.1, 0.2, 0.5, 0.6, 0.8, 1.])
    alphas   = np.array([0.9])#np.array([.01, .05, 0.1, 0.2, 0.5, 0.6, 0.8, 1.])
    disc_fac = 0.9

    legend_algo = ["Q(s,a) = 0  + Static Epsilon" , "Goal Biased + Decaying Epsilon"]

    errors = np.zeros((len(alphas), len(epsilons)))
    for i, alpha in enumerate(alphas):
        for j, epsilon in enumerate(epsilons):
            print "---------- For Epsilon: " + str(epsilon) + "----------"
            print "---------- For Alpha: " + str(alpha) + "----------"

            # Init Q values for states & action
            Q_learn = np.zeros((map.shape[0], map.shape[1], n_action))
            if j ==1:
                PROGRESS_REWARD = 1
                GOAL_BIAS_INIT = 1
                n_episodes = 1000
            else:
                PROGRESS_REWARD = 0
                GOAL_BIAS_INIT = 0

                n_episodes = 100

            if GOAL_BIAS_INIT:
                Q_learn = goal_based_Q_init(Q_learn)
                visualize_Q_function(Q_learn)

            Q_learn, interaction_error = Q_learn_learning(Q_learn, epsilon, disc_fac, alpha, GOAL_BIAS_INIT, n_episodes, PROGRESS_REWARD)

            #Check Validity
            # visualize_Q_function(Q_learn)

            # e = np.linalg.norm(calculate_optimal_Q(Q_s_a) - calculate_optimal_Q(Q_learn), ord=2)

            plot_errors_dict[legend_algo[j]] = interaction_error


    # print "Error Dict :::::::::::;"
    # print str(plot_errors_dict)
        # plt.plot(epsilons, errors[i, :])

    #Viz Plot
    visualize_norm_diff_plot(plot_errors_dict, alphas)






if __name__ == '__main__':
    main()