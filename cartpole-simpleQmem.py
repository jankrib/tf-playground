
import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import math
import random

def updateQ(q, a, r):
    shape = tf.shape(q)
    width = q.get_shape()[1]
    row_indices = tf.range(shape[0])
    col_indices = a

    linear_indices = row_indices*width + col_indices
    q_flat = tf.reshape(q, [-1])

    unchanged_indices = tf.range(tf.size(q_flat))
    changed_indices = linear_indices
    q_flat = tf.dynamic_stitch([unchanged_indices, changed_indices],
                               [q_flat, r])
    return tf.reshape(q_flat, shape)

class ExperienceMemory:
    def __init__(self, capacity, batchSize):
        self.capacity = capacity
        self.batchSize = batchSize
        self.cache = []

    def ready(self):
        return len(self.cache) >= self.batchSize

    def add(self, state, action, reward, newState):
        self.cache.append([state, int(action), reward, newState])

        if len(self.cache) > self.capacity:
            self.cache.pop(0)

    def getBatch(self):
        return random.sample(self.cache, self.batchSize)

    def getAll(self):
        l = self.cache
        l = list(map(list, zip(*l)))
        return l[0], l[1], l[2], l[3]

    def getReversed(self):
        l = reversed(self.cache)
        l = list(map(list, zip(*l)))
        return l[0], l[1], l[2], l[3]



#Params
y = .99
e = 0.5
num_episodes = 1000

#Model

x = tf.placeholder(tf.float32, [None, 4])

#Layer 1
W1 = tf.Variable(tf.truncated_normal(shape=[4, 2], stddev=0.001))
b1 = tf.Variable(tf.truncated_normal(shape=[2], stddev=0.001))


Qout = tf.matmul(x, W1) + b1
predict = tf.argmax(Qout, 1)

#NextState
xn = tf.placeholder(tf.float32, [None, 4])
actions = tf.placeholder(tf.int32, shape=[None])
rewards = tf.placeholder(tf.float32, shape=[None])

maxQout = tf.reduce_max(tf.matmul(xn, W1) + b1)

nextQ = updateQ(Qout, actions, rewards + y*maxQout)

#loss
#nextQ = tf.placeholder(shape=[None,2], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

#Trust  [0,0,1,0,0]  Describes how much the Q value can be trusted
#trust = tf.placeholder(shape=[1,2],dtype=tf.float32)




mem = ExperienceMemory(1000, 100)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

env = gym.make('CartPole-v0')

for i in range(num_episodes):
    observation = env.reset()
    totalReward = 0
    emem = ExperienceMemory(1000, 50)

    while True:
        #env.render()
        a = sess.run(predict, feed_dict={x: [observation]})
        action = a[0]

        if np.random.rand(1) < e:
            action = env.action_space.sample()

        newObservation, reward, done, _ = env.step(action)

        if done:
            reward = 0

        emem.add(observation, action, reward, newObservation)

        observation = newObservation
        totalReward += reward

        #print("Reward {}".format(reward))

        if done:
            e = 1.0/((i/50) + 10)
            print("{}: {}".format(i, totalReward))
            break

    #Training
    o, a, r, no = emem.getReversed()

    sess.run(updateModel, feed_dict={x: o, xn: no, actions: a, rewards: r})

env.close()
