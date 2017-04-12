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
import random
import string


class World:
    def __init__(self, env, session):
        self.env = env
        self.session = session

    def evaluate(self, net):
        observation = self.env.reset()
        totalReward = 0
        for _ in range(1000):
            action = net.getAction(self.session, observation)
            newObservation, reward, done, info = self.env.step(action)

            totalReward += reward

            if done:
                break

            observation = newObservation

        return totalReward

class Net:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 4])
        self.W = tf.Variable(tf.truncated_normal(shape=[4, 2], stddev=0.1))
        self.action = tf.argmax(tf.nn.softmax(tf.matmul(self.x, self.W)), 1)

        self.generateId()

    def getAction(self, session, observation):
        a = session.run(self.action, feed_dict={self.x: [observation]})
        return a[0]

    def generateId(self):
        self.id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(4))


class Tribe:
    def __init__(self, pop, world):
        self.pop = pop
        self.nets = []
        self.world = world

        self.W = tf.Variable(tf.truncated_normal(shape=[4, 2], stddev=0.1))
        #self.Wm = tf.Variable(tf.truncated_normal(shape=[4, 2], stddev=0.1))

        self.popRewards = tf.placeholder(tf.float32, shape=[(pop + 1)])
        self.rWeights = tf.nn.softmax(self.popRewards)

        for i in range(pop):
            net = Net()
            self.nets.append(net)
            netWeight = tf.gather(self.rWeights, [i])
            nW = net.W * netWeight if i == 0 else tf.add(nW, net.W * netWeight)

        self.assignments = [
            #tf.assign(self.Wm, nW-self.W),
            tf.assign(self.W, nW + self.W * tf.gather(self.rWeights, [pop]))
        ]

        dev = 0.01

        for net in self.nets:
            self.assignments.append(tf.assign(net.W, self.W + tf.truncated_normal([4, 2], stddev=dev)))

    def step(self, session, lastReward):
        rewards = []
        for net in self.nets:
            reward = self.world.evaluate(net)
            rewards.append(reward)

        total = sum(rewards)

        rewards.append(lastReward)
        session.run(self.assignments, feed_dict={self.popRewards: rewards})

        return total/self.pop

env = gym.make('CartPole-v0')
sess = tf.Session()

world = World(env, sess)

tribe = Tribe(10, world)

sess.run(tf.global_variables_initializer())
lastReward = 0

for i in range(1000):

    lastReward = tribe.step(sess, lastReward)

    print("{}: {}".format(i, lastReward))

    #min_index, min_value = min(enumerate(rewards), key=lambda p: p[1])
    #max_index, max_value = max(enumerate(rewards), key=lambda p: p[1])

    #f = 1 - ((100+max_value) / 1000)*0.99

    #nets[min_index].adopt(sess, nets[max_index])


env.close()
