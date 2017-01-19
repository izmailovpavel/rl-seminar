import gym
import numpy as np
import matplotlib.pyplot as plt
import random
A_CNT = 3
gamma = 1.0

class Featurizer(object):
    def __init__(self, env=None, n_samples=[400], gamma=[0.007]):
        if env is not None:
            self.samples = np.array([env.observation_space.sample() for x in range(sum(n_samples))])
            self.n_samples = n_samples
            self.gamma = gamma
            self.fd = 3 + sum(n_samples)
            self.mean = np.zeros(self.fd, dtype=np.float32)
            self.std = np.ones(self.fd, dtype=np.float32)
            X = self.get_features([env.observation_space.sample() for x in range(10000)])
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)

    def get_features(self, states):
        s = np.array(states).reshape([-1, 2])
        N = s.shape[0]
        f = np.hstack((np.ones((N, 1), dtype=np.float32), s))
        p = 0
        for i in range(len(self.n_samples)):
            scale = np.array([1.0, 4.0])
            x = scale[None, :] * s
            y = scale[None, :] * self.samples[p: p + self.n_samples[i]]
            d = np.sum(np.square(x), axis=1)[:, None] - 2.0 * np.dot(x, y.T) + np.sum(np.square(y), axis=1)[None, :]
            rbf = np.exp(-d / self.gamma[i])
            f = np.hstack((f, rbf))
            p += self.n_samples[i]

        f[:, 1:] = (f[:, 1:] - self.mean[None, 1:]) / self.std[None, 1:]
        return f.squeeze()

    def copy(self):
        res = Featurizer()
        res.samples = self.samples.copy()
        res.n_samples = self.n_samples
        res.gamma = self.gamma
        res.fd = self.fd
        res.mean = self.mean
        res.std = self.std
        return res

class Model(object):
    def __init__(self, env=None):
        if env is not None:
            self.Featurizer = Featurizer(env)
            self.w = np.random.normal(0.0, 0.01, size=(A_CNT, self.Featurizer.fd))

    def copy(self):
        res = Model()
        res.Featurizer = self.Featurizer.copy()
        res.w = self.w.copy()
        return res

    def getQ(self, states, a=None):
        f = self.Featurizer.get_features(states)
        if a is not None:
            return np.dot(f, self.w[a, :])
        else:
            return np.dot(f, self.w.T)

    def getPolicy(self, state, eps=0.0): #_q_eps_greedy
        q = self.getQ(state)
        policy = np.ones((A_CNT), dtype=np.float32) * eps / A_CNT
        policy[np.argmax(q)] = 1.0 - eps + eps / A_CNT
        return policy

    def plot(self):
        min_x, max_x = -1.2, 0.6
        min_v, max_v = -0.07, 0.07
        x = np.linspace(min_x, max_x, 40)
        v = np.linspace(min_v, max_v, 60)
        phi = np.array([(xx, vv) for vv in v for xx in x]).reshape([-1, 2])
        phi = self.getQ(phi).reshape([v.size, x.size, A_CNT])
        d = np.argmax(phi, axis=2)
        mx, mv = np.meshgrid(x, v)
        CS = plt.contourf(mx, mv, d, levels=[-1e-4, 1.0 - 1e-4, 2.0 - 1e-4, 3.0])
        sx = self.Featurizer.samples[:, 0]
        sy = self.Featurizer.samples[:, 1]
        plt.scatter(sx, sy, c='k', marker='x')
        plt.colorbar(CS)
        plt.grid()
        plt.show()

    def update(self, states, actions, targets, alpha):
        actions = np.array(actions, dtype=np.int32)
        targets = np.array(targets, dtype=np.float32)
        f = self.Featurizer.get_features(states)
        q = self.getQ(states).reshape([-1, A_CNT])
        N = actions.shape[0]
        self.w[actions, :] += alpha * (targets - q[np.arange(N), actions])[:, None] * f


def sarsa(env, win, num_episodes=50000, alpha_init=0.001):
    res = []

    np.set_printoptions(precision=3, suppress=True)

    model = Model(env)
    eps = 0.75

    batch_size = 32
    replay_max_size = 50000
    replay_init_size = 10000

    replay_memory = []
    state = env.reset()
    for i in range(replay_init_size):
        policy = model.getPolicy(state, eps)
        action = np.random.choice(A_CNT, p=policy)
        n_state, reward, done, info = env.step(action)
        replay_memory.append((state, action, n_state, reward, done))
        state = n_state

    alpha = alpha_init
    scores = []

    k = 1
    episodes = 0
    step_count = 0
    sum_reward = 0
    state = env.reset()

    while episodes < num_episodes:
        policy = model.getPolicy(state, eps)
        action = np.random.choice(A_CNT, p=policy)
        n_state, reward, done, info = env.step(action)

        replay_memory.append((state, action, n_state, reward, done))

        if (len(replay_memory) > replay_max_size):
            replay_memory.pop(0)

        samples = random.sample(replay_memory, batch_size)
        state_batch, action_batch, n_state_batch, reward_batch, done_batch = map(np.array, zip(*samples))
        n_q = model.getQ(n_state_batch)
        target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                  gamma * np.max(n_q, axis=1)
        model.update(state_batch, action_batch, target_batch, alpha)

        if episodes % 100 == 99:
            env.render()
        sum_reward += reward
        step_count += 1
        if step_count % 40000 == 0:
            print('Episodes: %d. Steps: %d. Score: %.2f Alpha: %.5g Eps: %.5g' % (episodes, step_count, np.mean(scores), alpha, eps))
            print('reward = ', sum_reward)
            print('q = ', model.getQ(state))
            print('w norm = ', np.sqrt(np.sum(np.square(model.w[:, 1:]), axis=1)))
            model.plot()

        if step_count % 100000 == 0:
            eps *= 0.5
            print('Eps reduced to %.4f' % (eps))
            k += 1
            alpha = alpha_init / (k ** 0.6)

        if done:
            print('Win #%d!' % (episodes + 1))
            state = env.reset()
            scores.append(sum_reward)
            if len(scores) > 100:
                scores = scores[1:]
            episodes += 1
            if episodes % 100 == 0:
                res.append(np.mean(scores))
            sum_reward = 0
        else:
            state = n_state
        if episodes >= 0 and np.mean(scores) >= win:
            res.append(np.mean(scores))
            break
    return res, model

def run(env, win, alpha_init=0.0002):
    scores, model = sarsa(env, win, alpha_init=alpha_init)
    return scores, model
