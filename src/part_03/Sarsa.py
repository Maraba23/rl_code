import numpy as np
import random
from numpy import savetxt
import sys
import matplotlib.pyplot as plt
import pandas as pd

#
# This class implements the Q-Learning algorithm.
# We can use this implementation to solve Toy text environments from Gym project. 
#

class Sarsa:

    def __init__(self, env, alpha, gamma, epsilon, epsilon_min, epsilon_dec, episodes):
        self.env = env
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes

    def select_action(self, state):
        rv = random.uniform(0, 1)
        if rv < self.epsilon:
            return self.env.action_space.sample() # Explore action space
        return np.argmax(self.q_table[state]) # Exploit learned values

    def train(self, filename, plotFile):
        actions_per_episode = []
        reward_list = []
        for i in range(1, self.episodes+1):
            (state, _) = self.env.reset()
            rewards = 0
            done = False
            actions = 0
            a = self.select_action(state)
            while not done:
                next_state, reward, done, truncated, _ = self.env.step(a) 
                a_prime = self.select_action(next_state)
                new_value = self.q_table[state, a] + self.alpha * (reward + self.gamma * self.q_table[next_state, a_prime] - self.q_table[state, a])
                self.q_table[state, a] = new_value
                state = next_state
                a = a_prime
                actions=actions+1
                if reward == 1:
                    rewards += 100
                elif reward == 0 and not done:
                    rewards -= 1
                elif reward == 0 and done:
                    rewards -= 100
                # save to a reward to a list
            reward_list.append(rewards)


            actions_per_episode.append(actions)
            if i % 100 == 0:
                sys.stdout.write("Episodes: " + str(i) +'\r')
                sys.stdout.flush()
            
            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon * self.epsilon_dec
        df = pd.DataFrame({'episode': [i for i in range(1, self.episodes+1)], 'rewards': reward_list})
        df.to_csv('data/rewards_sarsa.csv', mode='a', index=False, header=False)

        savetxt(filename, self.q_table, delimiter=',')
        if (plotFile is not None): self.plotactions(plotFile, actions_per_episode)
        return self.q_table

    def plotactions(self, plotFile, actions_per_episode):
        plt.plot(actions_per_episode)
        plt.xlabel('Episodes')
        plt.ylabel('# Actions')
        plt.title('# Actions vs Episodes')
        plt.savefig(plotFile+".jpg")     
        plt.close()