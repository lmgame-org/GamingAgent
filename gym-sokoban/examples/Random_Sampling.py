import gymnasium as gym
import gym_sokoban
import time

# Before you can make a Sokoban Environment you need to call:
# import gym_sokoban
# This import statement registers all Sokoban environments
# provided by this package
env_name = 'Sokoban-v0'
env = gym.make(env_name, render_mode='human')

ACTION_LOOKUP = env.unwrapped.get_action_lookup()
print("Created environment: {}".format(env_name))

for i_episode in range(1):#20
    observation, info = env.reset()

    for t in range(100):#100
        env.render()
        action = env.action_space.sample()

        # Sleep makes the actions visible for users
        time.sleep(1)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        print(ACTION_LOOKUP[action], reward, done, info)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            env.render()
            break

    env.close()

time.sleep(10)
