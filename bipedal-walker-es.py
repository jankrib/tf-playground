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


class Environment:
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
        self.x = tf.placeholder(tf.float32, [None, 24])
        self.W = tf.Variable(tf.truncated_normal(shape=[24, 4], stddev=0.1))
        self.b = tf.Variable(tf.truncated_normal(shape=[4], stddev=0.1))
        self.action = tf.matmul(self.x, self.W) + self.b
        self.generateId()

    def adopt(self, session, net, factor = 1):
        dev = 0.01 * factor
        session.run(tf.assign(self.W, net.W + tf.truncated_normal([24, 4], stddev=dev)))
        session.run(tf.assign(self.b, net.b + tf.truncated_normal([4], stddev=dev)))
        self.generateId()

    def getAction(self, session, observation):
        a = session.run(self.action, feed_dict={self.x: [observation]})
        return a[0]

    def generateId(self):
        self.id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(4))




env = gym.make('BipedalWalker-v2')
sess = tf.InteractiveSession()

environment = Environment(env, sess)

nets = []
pop = 3

for i in range(pop):
    nets.append(Net())

sess.run(tf.global_variables_initializer())

#print(nets[0].getAction(sess, [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))


while True:
    rewards = []
    print("---------Run----------")
    for i in range(pop):
        reward = environment.evaluate(nets[i])
        print("{}:Reward {}".format(nets[i].id, reward))
        rewards.append(reward)

    min_index, min_value = min(enumerate(rewards), key=lambda p: p[1])
    max_index, max_value = max(enumerate(rewards), key=lambda p: p[1])

    #f = 1 - ((100+max_value) / 1000)*0.99

    nets[min_index].adopt(sess, nets[max_index])


env.close()
