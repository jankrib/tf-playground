
"""
cd /Users/bing/OpenAI/gym    # your gym directory
pip uninstall box2d-py
git clone https://github.com/pybox2d/pybox2d pybox2d_dev
cd pybox2d_dev
python setup.py build    # this will take 10-15 seconds
python setup.py develop

"""

import tensorflow as tf
import numpy as np
import gym
from gym import wrappers

import operator
import math
import random
import string


class World:
    def __init__(self, env, session):
        self.env = env
        self.session = session

    def evaluate(self, net):
        observation = self.env.reset()
        totalReward = 0
        net.reset()

        while True:
            action = net.getAction(self.session, observation)
            newObservation, reward, done, info = self.env.step(action)

            totalReward += reward

            if done:
                break

            observation = newObservation

        return totalReward

class Net:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 25])
        self.W = tf.Variable(tf.truncated_normal(shape=[25, 12], stddev=0.1))
        self.W2 = tf.Variable(tf.truncated_normal(shape=[12, 5], stddev=0.1))
        self.b = tf.Variable(tf.truncated_normal(shape=[12], stddev=1))
        self.c2 = tf.nn.relu(tf.matmul(self.x, self.W) + self.b)
        self.action = tf.matmul(self.c2, self.W2)
        self.generateId()

    def getAction(self, session, observation):
        obm = []
        obm.extend(observation)
        obm.append(math.sin(self.mem))
        a = session.run(self.action, feed_dict={self.x: [obm]})
        self.mem += a[0][4]
        return a[0][:4]

    def reset(self):
        self.mem = 0

    def generateId(self):
        self.id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(4))


class Tribe:
    def __init__(self, pop, world):
        self.pop = pop
        self.nets = []
        self.world = world

        

        self.W = tf.Variable(tf.truncated_normal(shape=[25, 12], stddev=0.1))
        self.W2 = tf.Variable(tf.truncated_normal(shape=[12, 5], stddev=0.1))
        self.b = tf.Variable(tf.truncated_normal(shape=[12], stddev=0.1))

        self.popRewards = tf.placeholder(tf.float32, shape=[pop])
        self.popWeights = tf.nn.softmax(self.popRewards)

        for i in range(pop):
            net = Net()
            self.nets.append(net)
            netWeight = tf.gather(self.popWeights, [i])
            nW = net.W * netWeight if i == 0 else tf.add(nW, net.W * netWeight)
            nW2 = net.W2 * netWeight if i == 0 else tf.add(nW2, net.W2 * netWeight)
            nb = net.b * netWeight if i == 0 else tf.add(nb, net.b * netWeight)

        self.assignments = [
            tf.assign(self.W, nW),
            tf.assign(self.W2, nW2),
            tf.assign(self.b, nb)
        ]

        dev = 0.01

        for net in self.nets:
            self.assignments.append(tf.assign(net.W, self.W + tf.random_normal([25, 12], stddev=dev)))
            self.assignments.append(tf.assign(net.W2, self.W2 + tf.random_normal([12, 5], stddev=dev)))
            self.assignments.append(tf.assign(net.b, self.b + tf.random_normal([12], stddev=dev)))

    def step(self, session):
        rewards = []
        for net in self.nets:
            reward = self.world.evaluate(net)
            rewards.append(reward)

        total = sum(rewards)

        session.run(self.assignments, feed_dict={self.popRewards: rewards})

        return total/self.pop

env = gym.make('BipedalWalker-v2')
env = wrappers.Monitor(env, './exp/bipedal-experiment-9')
sess = tf.Session()

world = World(env, sess)

tribe = Tribe(10, world)

sess.run(tf.global_variables_initializer())

for i in range(10000):

    r = tribe.step(sess)

    print("{}: {}".format(i, r))


env.close()
