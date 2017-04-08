import gym
import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  print(initial)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def evaluate(env, sess, index):
    observation = env.reset()
    totalReward = 0
    for _ in range(100):
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

env = gym.make('CartPole-v0')
observations = []
actions = []
rewards = []

sess = tf.InteractiveSession()

#Model
i = tf.placeholder(tf.int32, [1])

x = tf.placeholder(tf.float32, [None, 4])
wtable = weight_variable([10, 4, 2]) # tf.Variable(tf.zeros([4, 2]))
W = tf.gather_nd(wtable, i)
b = bias_variable([2]) #tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
a = tf.argmax(y, 1)

#Adjust model
r = tf.placeholder(tf.float32, [10])
rm = tf.nn.softmax(r)

sess.run(tf.global_variables_initializer())

print(sess.run(wtable, feed_dict={i: [0]}))
print(sess.run(W, feed_dict={i: [0]}))

results = []

for index in range(10):
    result = evaluate(env, sess, index)
    results.append(result)
    print("Total reward["+str(index)+"]: " + str(result))

print(sess.run(rm, feed_dict={r: results}))

#W.assign

#Training
y_ = tf.placeholder(tf.float32, [None, 2])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
