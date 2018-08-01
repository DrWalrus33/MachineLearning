import tensorflow as tf

tf.reset_default_graph()

input_data = tf.placeHolder(dtype=tf.float32, shape=None)

output_data = tf.placeHolder(dtype=tf.float32, shape=None)

slope= tf.Variable(0.5, dtype=tf.float32)

intercept=tf.Variable(0.1, dtype=tf.float32)

model_operation = slope * input_data + intercept

error = model_operation - output_data

squared_error = tf.square(error)

loss = tf.reduce_mean(squared_error)
