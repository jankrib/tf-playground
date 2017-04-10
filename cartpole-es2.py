import tensorflow as tf
import numpy as np
import gym

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

    def assignWeights(session, weights):
        session.run(tf.assign(self.W, weights + tf.truncated_normal([4, 2], stddev=0.1)))

    def getAction(self, session, observation):
        a = session.run(self.action, feed_dict={self.x: [observation]})
        return a[0]

sess = tf.InteractiveSession()
env = Environment(gym.make('CartPole-v0'), sess)

net = Net()

sess.run(tf.global_variables_initializer())

print("Reward {}".format(env.evaluate(net)))
