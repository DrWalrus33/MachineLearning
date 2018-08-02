import gym
import tensorflow as tf
import numpy as np



num_actions = 2

state_size = 4

path = "./cartpole-pg/"

training_episodes = 1000

max_steps_per_episode = 10000

episode_batch_size = 5

discount_rate = 0.95

class Agent:
    def __init__(self, num_actions, state_size):

        initializer = tf.contrib.layers.xavier_initializer()
        self.input_layer = tf.placeholder(dtype=tf.float32, shape=[None, state_size])

        hidden_layer = tf.layers.dense(self.input_layer, 8, activation=tf.nn.relu, kernel_initializer=initializer)
        hidden_layer_2 = tf.layers.dense(hidden_layer, 8, activation=tf.nn.relu, kernel_initializer=initializer)
        out = tf.layers.dense(hidden_layer_2, num_actions, activation=None)
        self.outputs = tf.nn.softmax(out)
        self.choice = tf.argmax(self.outputs, axis=1)
        self.rewards = tf.placeholder(shape=[None, ], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None, ], dtype=tf.float32)
        one_hot_actions = tf.one_hot(self.actions, num_actions)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=one_hot_actions)
        self.loss = tf.reduce_mean(cross_entropy * self.rewards)
        self.gradients = tf.gradients(self.loss, tf.trainable_variables())
        self.gradients_to_apply = []
        for _, _ in enumerate(tf.trainable_variables()):
            gradient_placeholder = tf.placeholder(tf.float32)
            self.gradients_to_apply.append(gradient_placeholder)
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
            self.update_gradients = optimizer.apply_graidents(zip(self.gradients_to_apply, tf.trainable_variables()))


def discount_normalize_rewards(rewards):
    discounted_rewards = np.zeros_like(rewards)
    total_rewards = 0

    for i in reversed(range(len(rewards))):
        total_rewards = total_rewards * discount_rate + rewards[i]
        discount__rewards[i] = total_rewards

    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)
    return discounted_rewards
agent = Agent( num_actions, state_size)

init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep = 2)

if not os.path.exists(path):
    os.makedirs(path)
tf.reset_default_graph()

env = gym.make("CartPole-v1")

print(env.observation_space)

print(env.action_space)

games_to_play  = 10

with tf.Session() as sess:

    sess.run(init)

    total_episode_rewards = []

    gradient_buffer = sess.run(tf.trainable_variables())

    for index, gradient in enumerate(graident_buffer):

        gradient_buffer[index] = gradient * 0

for i in range(games_to_play):
    obs = env.reset()
    episode_rewards = 0
    done = False
    while not done:
        env.render()
        action = env.action_space.sample()
        obs, reward, done, info =  env.step(action)
        episode_rewards += reward

    print(episode_rewards)
env.close()