"""
cd /Users/bing/OpenAI/gym    # your gym directory
pip uninstall box2d-py
git clone https://github.com/pybox2d/pybox2d pybox2d_dev
cd pybox2d_dev
python setup.py build    # this will take 10-15 seconds
python setup.py develop

"""


import gym

env = gym.make('BipedalWalker-v2')


observation = env.reset()
for t in range(1000):
    env.render()
    print(observation)
    action = env.action_space.sample()
    print(action)
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
