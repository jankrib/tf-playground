import tensorflow as tf


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

sess = tf.InteractiveSession()

#model
W = weight_variable([2, 2])#tf.placeholder(tf.float32, [4, 2])
wtable = weight_variable([3, 2, 2])

combined = tf.map_fn(lambda a:a + W, wtable)

sess.run(tf.global_variables_initializer())

print(sess.run(W));
print(sess.run(wtable));
print(sess.run(combined));
