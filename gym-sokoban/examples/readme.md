## Examples

This readme walks you through all steps to run an external gym environment. 
The example will use this repository [gym-sokoban](https://github.com/mpSchrader/gym-sokoban/) as this is part of this repository, but never the less this works similar with other external gym environments.

### 1 Install the additional package
You need to clone the repository and install the package as follows:
```Bash
git clone git@github.com:mpSchrader/gym-sokoban.git
cd gym-sokoban
pip install -e .
```
### 2 Import packages in your code

To use an external gym environment you allways need to import the corresponding package along with the regular gym package.
```Python
import gym
import gym_sokoban
```

### 3 Load the environment
From now on everything is as you are used to it. You can simply make the environment, render it, perform actions and so on.

```Python
env = gym.make('Sokoban-v0')

env.render(mode='human')

action = env.action_space.sample()
observation, reward, done, info = env.step(action)
```
Now that you are all set with the preparations enjoy the external environment.
