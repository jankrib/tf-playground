
import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import math

# activate tf && python.py

"""
0: hull angle speed
1: angular velocity
2: horizontal speed
3: vertical speed

4: j0 angle
5: j0 speed

6: j1 angle
7: j1 speed
8: j1 ground


9: j2 angle
10: j2 speed

11: j3 angle
12: j3 speed
13: j3 ground


.. and 10 lidar rangefinder measurements.
"""

sess = tf.Session()

env = gym.make('BipedalWalker-v2')

initW = [[0 for x in range(5)] for y in range(25)]
initW[0][0] = 1
initW[0][1] = 1
initW[0][2] = 1
initW[0][3] = 1

initW[2][4] = 0.35

initW[24][0] = -1
initW[24][1] = 1
initW[24][2] = 1
initW[24][3] = -1

initb = [0 for x in range(5)]
initb[0] = -0.1
initb[2] = -0.1

x = tf.placeholder(tf.float32, [1, 25])
W = tf.Variable(tf.constant(initW, dtype=tf.float32, shape=[25,5]))
b = tf.Variable(tf.constant(initb, dtype=tf.float32, shape=[5]))

initW2 = [[0 for x in range(5)] for y in range(25)]
initb2 = [0 for x in range(5)]

W2 = tf.Variable(tf.constant(initW2, dtype=tf.float32, shape=[25,5]))
b2 = tf.Variable(tf.constant(initb2, dtype=tf.float32, shape=[5]))

initW3 = [[0 for x in range(5)] for y in range(5)]

W3 = tf.Variable(tf.constant(initW3, dtype=tf.float32, shape=[5,5]))

out1 = tf.matmul(x, W) + b
out2 = tf.matmul(tf.nn.relu(tf.matmul(x, W2) + b2), W3)
out = out1 + out2

sess.run(tf.global_variables_initializer())
observation = env.reset()
mem = 0
distance = 0

for i in range(1000):
    #if i % 3 == 0:
    env.render()

    smem = math.sin(mem)

    obm = []
    obm.extend(observation)
    obm.append(smem)
    action = sess.run(out, feed_dict={x: [obm]})[0]
    mem += action[4]#0.3
    action = action[:4]

    #action = [-smem + observation[0], smem + observation[0], smem + observation[0], -smem + observation[0]]

    #action = [observation[0], observation[0], observation[0], observation[0]]
    observation, reward, done, info = env.step(action)
    speed = observation[2]
    distance += speed
    print("{}: {} -- {}".format(i, speed, observation[14]))
    if done:
        break

print("Total: {}".format(distance))
env.close()
