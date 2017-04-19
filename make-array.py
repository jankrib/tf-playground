import tensorflow as tf

sess = tf.InteractiveSession()

x = tf.constant([
    [1, 2, 3],
    [4, 5, 6],
], dtype=tf.float32)

actions = tf.constant([0,1], dtype=tf.int32)
rewards = tf.constant([7, 8], dtype=tf.float32)

def updateQ(q, a, r):
    row_indices = tf.range(q.get_shape()[0])
    col_indices = a

    linear_indices = row_indices*q.get_shape()[1] + col_indices
    q_flat = tf.reshape(q, [-1])

    unchanged_indices = tf.range(tf.size(q_flat))
    changed_indices = linear_indices
    q_flat = tf.dynamic_stitch([unchanged_indices, changed_indices],
                               [q_flat, r])
    return tf.reshape(q_flat, q.get_shape())

result = updateQ(x, actions, rewards)

print(sess.run(result))
