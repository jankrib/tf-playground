
import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import math
import random

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


class ExperienceMemory:
    def __init__(self, capacity, batchSize):
        self.capacity = capacity
        self.batchSize = batchSize
        self.cache = []

    def ready(self):
        return len(self.cache) >= self.batchSize

    def add(self, state, action, reward, newState):
        self.cache.append([state, action, reward, newState])

        if len(self.cache) > self.capacity:
            self.cache.pop(0)

    def getBatch(self):
        return random.sample(self.cache, self.batchSize)


action_space = [
    [1,-1,-1,1],
    [0,0,0,0],
    [-1,1,1,-1],
    [0,1,0,1],
    [0,-1,0,-1],
    [1,0,1,0],
    [-1,0,-1,0]
]

x = tf.placeholder(tf.float32, [1, 24])

#Layer 1
W1 = tf.Variable(tf.truncated_normal(shape=[24, 14], stddev=0.1))
b1 = tf.Variable(tf.truncated_normal(shape=[14], stddev=0.1))

l1 = tf.nn.relu(tf.matmul(x, W1) + b1)

#Layer 2
W2 = tf.Variable(tf.truncated_normal(shape=[14, 7], stddev=0.1))
b2 = tf.Variable(tf.truncated_normal(shape=[7], stddev=0.1))

l2 = tf.nn.relu(tf.matmul(l1, W2) + b2)

#Layer 3
W3 = tf.Variable(tf.truncated_normal(shape=[7, 7], stddev=0.1))
b3 = tf.Variable(tf.truncated_normal(shape=[7], stddev=0.1))

Qout = tf.matmul(l2, W3) + b3
predict = tf.argmax(Qout,1)

maxQout = tf.reduce_max(Qout)

#loss
nextQ = tf.placeholder(shape=[1,7], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

#Trust  [0,0,1,0,0]  Describes how much the Q value can be trusted
trust = tf.placeholder(shape=[1,7],dtype=tf.float32)


#Params
y = .99
e = 0.1
num_episodes = 2000



mem = ExperienceMemory(1000, 100)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

env = gym.make('BipedalWalker-v2')

for i in range(num_episodes):
    observation = env.reset()
    totalReward = 0
    while True:
        #env.render()
        a = sess.run(predict, feed_dict={x: [observation]})
        action = a[0]

        if np.random.rand(1) < e:
            action = np.random.randint(len(action_space))

        newObservation, reward, done, _ = env.step(action_space[action])

        mem.add(observation, action, reward, newObservation)

        if mem.ready() == True:
            for exp in mem.getBatch():
                o = exp[0]
                a = exp[1]
                r = exp[2]
                no = exp[3]

                allQ = sess.run(Qout, feed_dict={x: [o]})
                maxQ1 = sess.run(maxQout,feed_dict={x: [no]})

                targetQ = allQ
                targetQ[0, a] = r + y*maxQ1

                sess.run(updateModel, feed_dict={x: [o], nextQ: targetQ})

        observation = newObservation
        totalReward += reward

        if done:
            e = 1.0/((i/50) + 10)
            print("{}: {}".format(i, totalReward))
            break


env.close()
