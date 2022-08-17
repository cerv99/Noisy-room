# %%
import random

import pandas as pd
from matplotlib import style

select_env = 3
if select_env == 1:
    from env1 import envCube
elif select_env == 2:
    from env2 import envCube
else:
    from env3 import envCube
style.use('ggplot')

import numpy as np
import matplotlib.pyplot as plt

SIZE = 10  # size of area
EPISODES = 40000
SHOW_EVERY = 3000  # show image every
epsilon = 0.6
EPS_DECAY = 0.9998
DISCOUNT = 0.95
LEARNING_RATE = 0.1
q_table = None

env = envCube()
q_table = env.get_qtable()

# train an agent
episode_rewards = []  # Initialize reward sequence
for episode in range(EPISODES):
    obs = env.reset()
    done = False

    if episode % SHOW_EVERY == 0:
        print('episode ', episode, '  epsilon:', epsilon)
        print('mean_reward:', np.mean(episode_rewards[-SHOW_EVERY:]))
        show = True
    else:
        show = False

    episode_reward = 0
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            # action = np.random.randint(0, env.ACTION_SPACE_VALUES)
            action = random.choice([i for i in range(0, env.ACTION_SPACE_VALUES)])  # randomly make a choice to explore

        new_obs, reward, done = env.step(action)

        # Update q-table
        current_q = q_table[obs][action]
        max_future_q = np.max(q_table[new_obs])
        if reward == env.FOOD_REWARD:
            new_q = env.FOOD_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q
        obs = new_obs

        if show:
            env.render()

        episode_reward += reward

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.xlabel('episode #')
plt.ylabel(f'mean{SHOW_EVERY} reward')
plt.show()

df = pd.DataFrame()
df['reward'] = moving_avg
df.to_csv(f'reward_env{select_env}.csv')