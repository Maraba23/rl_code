from IPython.display import clear_output
import gymnasium as gym
import numpy as np
from Sarsa import Sarsa
from numpy import loadtxt
import warnings
warnings.simplefilter("ignore")

# exemplo de ambiente nao determinístico
env = gym.make('FrozenLake-v1', render_mode='ansi').env

# only execute the following lines if you want to create a new q-table
# qlearn = Sarsa(env, alpha=0.6, gamma=0.99, epsilon=1.5, epsilon_min=0.0001, epsilon_dec=0.9999, episodes=50000)
# q_table = qlearn.train('data/q-table-test.csv','results/frozen_test')
q_table = loadtxt('data/q-table-test.csv', delimiter=',')

env = gym.make('FrozenLake-v1', render_mode='human').env
# desc=["SHFF", "FFHH", "HFFH", "FHFG"]
(state, _) = env.reset()
epochs = 0
rewards = 0
done = False
    
while not done:
    action = np.argmax(q_table[state])
    state, reward, done, _, info = env.step(action)
    rewards += reward
    epochs += 1

print("\n")
print("Timesteps taken: {}".format(epochs))
print("Rewards: {}".format(rewards))