import gym
import gym_chess

env = gym.make('Chess-v0')

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))
print('\n\n')

state = env.reset()


