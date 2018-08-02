# Load cifar data from file

import tensorflow as tf

import numpy as np

class ConvNet:
    def __init__(self, image_height, image_width, channels, num_classes):
        print("test1")
        self.input_layer = tf.placeholder(dtype=tf.float32, shape = [None, image_height, image_width, channels])
        print (self.input_layer.shape)

        #First layer defining

        conv_layer_1 = tf.layers.conv2d(self.input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        print(conv_layer_1.shape)

        pooling_layer_1 = tf.layers.max_pooling2d(conv_layer_1, pool_size = [2,2], strides=2)
        print(pooling_layer_1.shape)

        #Second layer defining

        """conv_layer_2 = tf.layers.conv2d(pooling_layer_1, filters=64, kernel_size=[5, 5], padding="same",activation=tf.nn.relu)
        print(conv_layer_2.shape)

        pooling_layer_2 = tf.layers.max_pooling2d(conv_layer_2, pool_size=[2, 2], strides=2)
        print(pooling_layer_2.shape)"""

        flatten_pooling = tf.layers.flatten(pooling_layer_1)
#       Adding more neurons

        dense_layer = tf.layer.dense(flatten_pooling, 1024, activation=tf.nn.relu)
        print(dense_layer.shape)

        #Getting rid of some neurons to stop dependebility
        dropout = tf.layers.dropout(dense_layer, rate=0.4, training=True)

        outputs = tf.layers.dense(dropout, num_classes)
        print(outputs.shape)

        self.choice = tf.argmax(outputs, axis=1)
        self.probability = tf.nn.softmax(outputs)

        self.labels = tf.placeholder(dtype=tf.float32, name="labels")
        self.accuracy, self.accuracy_op = tf.metrics.accuracy(self.labels, self.choice)

        one_hot_labels = tf.one_hot(indices=tf.cast(self.labels, dtype=tf.int32), depth=num_classes)
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=outputs)

        optimizer = tf.train.GradientDescenentOptimizer(learning_rate=1e-2)
        self.train_operation = optimizer.minimize(loss=self.loss, global_step=tf.train.get_global_step())

import matplotlib.pyplot as plt

image_height = 32

image_width = 32

training_steps = 5000

batch_size = 64

model_name = "cifar"

path = "./" + model_name + "-cnn/"

load_checkpoint = False

color_channels = 3

print("test2")


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

print("test3")
cifar_path = '/home/student/Desktop/MachineLearningmUrr/cifar-10-data/'

train_data = np.array([])
train_labels = np.array([])

# Load all the data batches.
for i in range(1, 6):
    print("test4")
    data_batch = unpickle(cifar_path + 'data_batch_' + str(i))
    train_data = np.append(train_data, data_batch[b'data'])
    train_labels = np.append(train_labels, data_batch[b'labels'])

# Load the eval batch.
eval_batch = unpickle(cifar_path + 'test_batch')
print("test5")
eval_data = eval_batch[b'data']
eval_labels = eval_batch[b'labels']

# Load the english category names.
category_names_bytes = unpickle(cifar_path + 'batches.meta')[b'label_names']
category_names = list(map(lambda x: x.decode("utf-8"), category_names_bytes))


# TODO: Process Cifar data

print("test6")

def process_data(data):
    float_data = np.array(data, dtype=float) / 255.0
    print("test7")
    reshaped_data = np.reshape(float_data, (-1, image_height, image_width, color_channels))
    transposed_data = np.transpose(reshaped_data, [0,2,3,1])
    return transposed_data

train_data = process_data(train_data)
eval_data = process_data(eval_data)
print("test8")
tf.reset_default_graph()
print("test9")
dataset= tf.data.Dataset.from_tensor_slices((train_data, train_labels))
print("testa")
dataset = dataset.shuffle(buffer_size=train_labels.shape[0])
print("testb")
dataset = dataset.batch(batch_size)
print("testc")
dataset = dataset.repeat()
print("testd")
print("testtt")
dataset_iterator = dataset.make_initializable_iterator()
print("test10")
next_element = dataset_iterator.get_next()
print("test11")
cnn = ConvNet(image_height, image_width, color_channels, 10)