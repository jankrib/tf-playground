
import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import math
import random

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


x = tf.placeholder(tf.float32, [1, 4])

#Layer 1
W1 = tf.Variable(tf.truncated_normal(shape=[4, 2], stddev=0.001))
b1 = tf.Variable(tf.truncated_normal(shape=[2], stddev=0.001))


Qout = tf.matmul(x, W1) + b1
predict = tf.argmax(Qout, 1)

maxQout = tf.reduce_max(Qout)

#loss
nextQ = tf.placeholder(shape=[1,2], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

#Trust  [0,0,1,0,0]  Describes how much the Q value can be trusted
#trust = tf.placeholder(shape=[1,2],dtype=tf.float32)


#Params
y = .99
e = 0.1
num_episodes = 1000



mem = ExperienceMemory(1000, 100)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

env = gym.make('CartPole-v0')

for i in range(num_episodes):
    observation = env.reset()
    totalReward = 0
    while True:
        #env.render()
        a = sess.run(predict, feed_dict={x: [observation]})
        action = a[0]

        if np.random.rand(1) < e:
            action = env.action_space.sample()

        newObservation, reward, done, _ = env.step(action)

        mem.add(observation, action, reward, newObservation)

        allQ = sess.run(Qout, feed_dict={x: [observation]})
        maxQ1 = 0 if done else sess.run(maxQout, feed_dict={x: [newObservation]})

        targetQ = allQ
        targetQ[0, action] = reward + y*maxQ1

        sess.run(updateModel, feed_dict={x: [observation], nextQ: targetQ})

        observation = newObservation
        totalReward += reward

        #print("Reward {}".format(reward))

        if done:
            e = 1.0/((i/50) + 10)
            print("{}: {}".format(i, totalReward))
            break


env.close()
