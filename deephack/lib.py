import gym
import numpy as np
import matplotlib.pyplot as plt
import random
A_CNT = 3
gamma = 1.0

class Model(object):
    def __init__(self, features_num=0):
        self.w = np.random.normal(0.0, 0.01, size=(A_CNT, features_num))

    def copy(self):
        res = Model()
        res.w = self.w.copy()
        return res

    def getQ(self, states_f, a=None):
        if a is not None:
            return np.dot(states_f, self.w[a, :].T)
        else:
            return np.dot(states_f, self.w.T)

    def getPolicy(self, state_f, eps=0.0): #_q_eps_greedy
        q = self.getQ(state_f)
        policy = np.ones((A_CNT), dtype=np.float32) * eps / A_CNT
        policy[np.argmax(q)] = 1.0 - eps + eps / A_CNT
        return policy

    def update(self, states_f, actions, targets, alpha):
        N = actions.size
        q = self.getQ(states_f)
        self.w[actions, :] += alpha * (targets - q[np.arange(N), actions])[:, None] * states_f

def process_state(state,
                  state_transform,
                  last_reprs):
    s_repr = state_transform(state)
    last_reprs.pop(0)
    last_reprs.append(s_repr)
    state_f = np.concatenate(([1], np.array(last_reprs).ravel()))
    return state_f, last_reprs

def run(env,
        state_transform,
        n_frames=4,
        num_episodes=20000,
        alpha_init=0.00001):

    res = []
    np.set_printoptions(precision=3, suppress=True)

    features_num = 1 + n_frames * 64
    model = Model(features_num)
    eps = 0.75

    batch_size = 32
    replay_max_size = 20000
    replay_init_size = 1000

    replay_memory = []

    last_reprs = [np.zeros(64, dtype=np.float32) for i in range(n_frames)]
    state = env.reset()
    state_f, last_reprs = process_state(state, state_transform, last_reprs)
    for i in range(replay_init_size):
        policy = model.getPolicy(state_f, eps)
        action = np.random.choice(A_CNT, p=policy)
        n_state, reward, done, info = env.step(action)
        n_state_f, last_reps = process_state(n_state, state_transform, last_reprs)
        replay_memory.append((state_f, action, n_state_f, reward, done))
        state_f = n_state_f

        if ((i + 1) % 100 == 0):
            print('Initial replay filled with (%d) records' % (i + 1))

    alpha = alpha_init
    scores = []

    k = 1
    episodes = 0
    step_count = 0
    sum_reward = 0
    last_reprs = [np.zeros(64, dtype=np.float32) for i in range(n_frames)]
    state = env.reset()
    state_f, last_reprs = process_state(state, state_transform, last_reprs)
    while episodes < num_episodes:
        policy = model.getPolicy(state_f, eps)
        action = np.random.choice(A_CNT, p=policy)
        n_state, reward, done, info = env.step(action)
        n_state_f, last_reps = process_state(n_state, state_transform, last_reprs)
        replay_memory.append((state_f, action, n_state_f, reward, done))

        if (len(replay_memory) > replay_max_size):
            replay_memory.pop(0)

        samples = random.sample(replay_memory, batch_size)
        state_f_batch, action_batch, n_state_f_batch, reward_batch, done_batch = map(np.array, zip(*samples))
        n_q = model.getQ(n_state_f_batch)
        target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                  gamma * np.max(n_q, axis=1)
        model.update(state_f_batch, action_batch, target_batch, alpha)

        if episodes % 20 == 0:
            env.render()
        sum_reward += reward
        step_count += 1
        if step_count % 5000 == 0:
            print('Episodes: %d. Steps: %d. Score: %.2f Alpha: %.5g Eps: %.5g' % (episodes, step_count, np.mean(scores), alpha, eps))
            print('reward = ', sum_reward)
            print('q = ', model.getQ(state_f))
            print('w norm = ', np.sqrt(np.sum(np.square(model.w[:, :]), axis=1)))

        if done:
            print('Episode #%d ended! Reward: %d' % (episodes + 1, sum_reward))
            eps *= 0.995
            print('Eps reduced to %.4f' % (eps))
            last_reprs = [np.zeros(64, dtype=np.float32) for i in range(n_frames)]
            state = env.reset()
            state_f, last_reprs = process_state(state, state_transform, last_reprs)
            scores.append(sum_reward)
            if len(scores) > 100:
                scores = scores[1:]
            episodes += 1
            if episodes % 100 == 0:
                res.append(np.mean(scores))
            if episodes % 100 == 0:
                k += 1
                alpha = alpha_init / (k ** 0.6)
                print('Alpha reduced to %.4f' % (alpha))
            sum_reward = 0
        else:
            state_f = n_state_f
    return res, model
