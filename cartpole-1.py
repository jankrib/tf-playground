import gym
import tensorflow as tf

env = gym.make('CartPole-v0')
observations = []
actions = []
rewards = []

observation = env.reset()
for _ in range(1000):
    #env.render()
    action = env.action_space.sample()
    newObservation, reward, done, info = env.step(action)
    observations.append(observation)
    actions.append(action)
    rewards.append(reward)
    observation = newObservation

#Model
x = tf.placeholder(tf.float32, [None, 4])
W = tf.Variable(tf.zeros([4, 2]))
b = tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

#Training
y_ = tf.placeholder(tf.float32, [None, 2])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))



print(observations)
