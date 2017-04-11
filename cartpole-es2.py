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
        self.x = tf.placeholder(tf.float32, [None, 4])
        self.W = tf.Variable(tf.truncated_normal(shape=[4, 2], stddev=0.1))
        #self.b = tf.Variable(tf.zeros([2]))
        self.randomVector = tf.truncated_normal([4, 2], stddev=0.1)
        self.y = tf.nn.softmax(tf.matmul(self.x, self.W))
        self.action = tf.argmax(self.y, 1)
        self.generateId()

    def adopt(self, session, net, factor):
        dev = 0.1 * factor
        session.run(tf.assign(self.W, net.W + tf.truncated_normal([4, 2], stddev=dev)))
        self.generateId()

    def getAction(self, session, observation):
        a = session.run(self.action, feed_dict={self.x: [observation]})
        return a[0]

    def generateId(self):
        self.id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(4))

sess = tf.InteractiveSession()
env = Environment(gym.make('CartPole-v0'), sess)
#env = wrappers.Monitor(env, './cartpole-experiment-1')

nets = []

for i in range(10):
    nets.append(Net())

sess.run(tf.global_variables_initializer())

for _ in range(100):
    rewards = []
    print("---------Run----------")
    for i in range(10):
        reward = env.evaluate(nets[i])
        print("{}:Reward {}".format(nets[i].id, reward))
        rewards.append(reward)

    min_index, min_value = min(enumerate(rewards), key=lambda p: p[1])
    max_index, max_value = max(enumerate(rewards), key=lambda p: p[1])

    f = 1 - (max_value / 200)*0.99

    nets[min_index].adopt(sess, nets[max_index], f)
