import tensorflow as tf
import numpy as np


def policy(n_actions, n_features, alpha=0.01):
    """
    Builds the part of the computational graph, related to policy
    :param alpha: optimizer step-size
    """
#    with tf.variable_scope('Policy'):
    params = tf.get_variable('float', [n_features, n_actions])
    state_ph = tf.placeholder('float', [None, n_features], 'States')
    preferences = tf.matmul(state_ph, params)
    probabilities = tf.nn.softmax(preferences)
    
    # training loop
    actions_ph = tf.placeholder('float', [None, n_actions], 'ObservedActions')
    advantages_ph = tf.placeholder('float', [None, 1], 'ObservedAdvantages')
    log_likelihood = tf.log(tf.reduce_sum(tf.mul(probabilities, actions_ph)))
    loss = - tf.reduce_sum(log_likelihood * advantages_ph)
    optimizer = tf.train.AdamOptimizer(alpha).minimize(loss)

    return probabilities, optimizer, state_ph, advantages_ph, actions_ph 

def value_function(n_features, beta=0.1):
    """
    Builds the part of the computational graph, related to value function approximation
    :param beta: optimizer step-size 
    """
#    with tf.variable_scope('Value Function'):
    state_ph = tf.placeholder('float', [None, n_features], 'States')
    
    # 1 hidden layer nn
    n_hidden = 10
    w1 = tf.get_variable('w1', [n_features, n_hidden])
    b1 = tf.get_variable('b1', [n_hidden])
    w2 = tf.get_variable('w2', [n_hidden, 1])
    b2 = tf.get_variable('b2', [1])
    h1 = tf.nn.relu(tf.matmul(state_ph, w1) + b1, 'hidden')
    value = tf.add(tf.matmul(h1, w2), b2, 'output')

    # training loop
    observed_returns_ph = tf.placeholder('float', [None, 1], 'ObservedReturns')
    diffs = value - observed_returns_ph
    loss = tf.nn.l2_loss(diffs)
    optimizer = tf.train.AdamOptimizer(beta).minimize(loss)
    return value, optimizer, state_ph, observed_returns_ph
