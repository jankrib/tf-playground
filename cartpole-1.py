import gym
import tensorflow as tf
import numpy as np

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  print(initial)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def map(fn, arrays, dtype=tf.float32):
    # assumes all arrays have same leading dim
    indices = tf.range(tf.shape(arrays[0])[0])
    out = tf.map_fn(lambda ii: fn(*[array[ii] for array in arrays]), indices, dtype=dtype)
    return out

def evaluate(env, sess, index):
    observation = env.reset()
    totalReward = 0
    for _ in range(1000):
        #env.render()
        obm = [observation]
        action = sess.run(a, feed_dict={x: obm, i: [index]})[0]
        newObservation, reward, done, info = env.step(action)
        observations.append(observation)
        actions.append(action)
        rewards.append(reward)

        totalReward += reward

        if done:
            break;

        observation = newObservation

    return totalReward

def train(env, sess):
    results = []
    for index in range(10):
        result = evaluate(env, sess, index)
        results.append(result)
        #print("Total reward["+str(index)+"]: " + str(result))

    #outW = sess.run(resultW, feed_dict={r: results})
    sess.run(tf.assign(startWeight, resultW), feed_dict={r: results})
    return np.mean(results)

env = gym.make('CartPole-v0')
observations = []
actions = []
rewards = []

sess = tf.InteractiveSession()

#Model
i = tf.placeholder(tf.int32, [1])
x = tf.placeholder(tf.float32, [None, 4])
startWeight = weight_variable([4, 2])
randomVectors = tf.truncated_normal([10, 4, 2], stddev=0.1)
wtable = tf.map_fn(lambda a:a + startWeight, randomVectors)
W = tf.gather_nd(wtable, i)
#b = bias_variable([2])
y = tf.nn.softmax(tf.matmul(x, W))
a = tf.argmax(y, 1)

#Adjust model
r = tf.placeholder(tf.float32, [10])
rm = tf.nn.softmax(r)
weighted = map(lambda x, y: tf.multiply(x, y), [wtable, rm])
resultW = tf.reduce_sum(weighted, 0)

sess.run(tf.global_variables_initializer())

for episode in range(100):
    print(train(env, sess))
