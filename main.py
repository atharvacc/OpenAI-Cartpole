#CARTPOLE PROBLEM USING Q-LEARNING USING EPSILON GREEDY STRATEGY
#Uncomment the code under plt_avg() to plot the values
#Due to randomness, the agent isn't very stable, and doesn't always get an optimal value soon
#episode for getting avg_reward over last 100 episodes to 150 is in the range of 200-350


import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
tot_state = np.power(10,6)
gamma = 1.     #set discount factor to 1 since we dont want to penalize for future rewards
alpha = 0.02    #Learning Rate
episodes = 1000     # number of episodes to play



#make bins to store continuous values from the observation as discrete values
# the values are position_cart, cart_velocity, pole_angle, pole_velocity
def make_bin():
    bins = np.zeros((4,10))
    bins[0] = np.linspace(-4.8,4.8,10)
    bins[1] = np.linspace(-5,5,10)
    bins[2] = np.linspace(-0.418,0.418,10)
    bins[3] = np.linspace(-5,5,10)
    return bins


#look at observation and assign to bins
def assign_bins(observation,bins):
    state = np.zeros(4)
    for i in range(4):
        state[i] = np.digitize(observation[i],bins[i])
    return state



#get string mapping for all states
def get_all_state_as_string():
    states = []
    for i in range(tot_state):
        states.append(str(i).zfill(4))
    return states

#from dictionary get max_val
def max_dictionary(d):
    max_v = float('-inf')
    for key, val in d.items():
        if val > max_v:
            max_v = val
            max_key = key
    return max_key, max_v



#initialize Q Table
def initialize_Q():
    Q = {}
    all_states = get_all_state_as_string()

    for state in all_states:
        Q[state] = {}
        for action in range(env.action_space.n):
            Q[state][action] = 0
    return Q

#use this for key in Q_table
def get_state_as_string(state):

    string_state = ''.join(str(int(e)) for e in state)
    return string_state



#play one game using greedy action or random action based on epsilon

def play_one_game(bins, Q , eps = 0.5):
    observation = env.reset()
    done = False
    t = 0
    randm = 0
    greedy = 0
    state =get_state_as_string(assign_bins(observation, bins))
    game_reward = 0

    while not done:
        t = t+1
        if (np.random.uniform() < eps):                 # Tried using np.random.randn() but it wasn't being greedy so switched to uniform
            action = env.action_space.sample()         #choose random action
        else:
            action = max_dictionary(Q[state])[0]

        observation,reward,done, _ = env.step(action)
        game_reward = game_reward + reward

        if done and t < 160:
            reward = -300   #penalize for not doing well

        new_state = get_state_as_string(assign_bins(observation,bins))
        a1, max_q_val = max_dictionary(Q[new_state])          #get maximum value from Q-Table
        Q[state][action] += alpha * (reward+gamma * max_q_val - Q[state][action])           #Bellmans Equation

        state,action = new_state, a1

    return game_reward, t


def train(bins, episodes):
    Q = initialize_Q()
    time = []
    rewards = []
    for i in range(episodes):
        eps = 1.0 / np.sqrt(i + 1)
        reward_ep, time_ep = play_one_game(bins , Q,eps)
        #if i % 100 == 0:
            #print(i, '%.4f' % eps, reward_ep)
        time.append(time_ep)
        rewards.append(reward_ep)

    return time, rewards

def plot_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(totalrewards[max(0, t - 100):(t + 1)])

    # plot running average 
    plt.plot(running_avg)
    plt.title("Running_avg")
    plt.show()
    return running_avg


def get_greater_than_150(avg_array):
    for i in range(avg_array.shape[1]):

        if average_array[0,i] > 150:
            return i


#Initialize bins and train
bins = make_bin()
episode_lengths, episode_rewards = train(bins,episodes)
average_array = plot_avg(episode_rewards)

average_array = np.reshape(average_array,[1,1000])

a = get_greater_than_150(average_array)
print("The average value over 100 epsiodes coverges to 150 at ",a)

