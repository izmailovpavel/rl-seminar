import tensorflow as tf
import numpy as np
import itertools
import os

import actor_critic as ac
import gym
from gym import wrappers

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('alpha', 0.01, 'Policy learning rate')
flags.DEFINE_float('beta', 0.1, 'Value function learning rate')
flags.DEFINE_integer('n_episodes', 500, 'Number of training episodes')
flags.DEFINE_string('logdir', 'data', 'Directory for progress logging')
flags.DEFINE_bool('refresh_stats', False, 'Deletes old events from logdir if True')
flags.DEFINE_float('gamma', 1., 'Discount rate')
flags.DEFINE_integer('n_warming', 0, 'Number of warming runs to train the value function prediciton')
flags.DEFINE_integer('n_render', 0, 'Number of episodes to render after training is finished')
flags.DEFINE_float('lambda_', 0., 'TD(lambda_) approximation is used')

if FLAGS.refresh_stats:
    print('Deleting old stats')
    os.system('rm -rf '+FLAGS.logdir)

env = gym.make('CartPole-v0')
n_actions = env.action_space.n
n_features = env.observation_space.shape[0]
lambda_ = FLAGS.lambda_
gamma = FLAGS.gamma

np.random.seed(417)
tf.set_random_seed(417)
with tf.Graph().as_default():
     
    pl_probabilities, pl_train, pl_states, pl_advantages, pl_actions = ac.policy(n_actions, n_features, FLAGS.alpha)
    vf_values, vf_train, vf_states, vf_observed_returns = ac.value_function(n_features, FLAGS.beta)
    n_timesteps_ph = tf.placeholder(tf.int32, [])
    tf.scalar_summary('timesteps', n_timesteps_ph)
    merged = tf.merge_all_summaries()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.train.SummaryWriter(FLAGS.logdir, sess.graph)
        init = tf.initialize_all_variables()
        sess.run(init)
        
        
        for i in range(FLAGS.n_episodes):
            actions = []
            states = []
            transitions = []
            rewards = []
            total_reward = 0
            observation = env.reset()
            print('Episode', i, ': playing')
            for t in range(200):
                obs_vector = np.expand_dims(observation, axis=0)
                probs = sess.run(pl_probabilities, feed_dict={pl_states: obs_vector})
                action = np.random.choice(np.arange(n_actions), p=probs.reshape(-1))
                states.append(observation)
                action_blank = np.zeros(n_actions)
                action_blank[action] = 1
                actions.append(action_blank)
                old_observation = observation
                observation, reward, done, info = env.step(action)
                rewards.append(reward)
                transitions.append((old_observation, action, reward))
                total_reward += reward
                if done: break
            print('Episodde terminated in', t ,'timesteps')
            summary = sess.run(merged, feed_dict={n_timesteps_ph: t}) 
            writer.add_summary(summary, i)
            print('Episode', i, ': training')
            advantages = []
            
            values = sess.run(vf_values, feed_dict={vf_states: states})[:, 0]
            returns = list(itertools.accumulate(rewards[::-1], lambda x, y: gamma * x + y))[::-1]

            for idx, trans in enumerate(transitions):
                obs, action, reward = trans
                future_return = 0
                future_transitions = len(transitions) - idx
                decrease = 1.
                state = np.expand_dims(obs, axis=0)
                advantages.append(returns[idx] - values[idx])
            returns = np.expand_dims(returns, axis=1)
            advantages = np.expand_dims(advantages, axis=1)
            sess.run(vf_train, feed_dict={vf_states: states, vf_observed_returns: returns})
            if i >= FLAGS.n_warming:
                sess.run(pl_train, feed_dict={pl_states: states, pl_advantages: advantages, pl_actions: actions})  
            print()
   
   # Render some episodes
        print('Redering')
        env = wrappers.Monitor(env, 'videos/', force=True)
        for episode in range(FLAGS.n_render):
            observation = env.reset()
            for t in range(200):
                env.render()
                obs_vector = np.expand_dims(observation, axis=0)
                probs = sess.run(pl_probabilities, feed_dict={pl_states: obs_vector})
                action = np.random.choice(np.arange(n_actions), p=probs.reshape(-1))
                observation, reward, done, info = env.step(action)
                if done: break
            print('Rendering episode', episode)
            print('Terminated in', t, 'timesteps')
            print()
        env.close()
print('Done')
