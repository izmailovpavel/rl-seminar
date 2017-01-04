import gym
import numpy as np

gamma = 1.0

def policy_q_eps_greedy(q, eps):
    S_CNT, A_CNT = q.shape
    policy = np.ones((S_CNT, A_CNT), dtype=np.float32) * eps / A_CNT
    policy[np.arange(S_CNT), np.argmax(q, axis=1)] = 1.0 - eps + eps / A_CNT
    return policy

def sarsa_lambda(env, win, num_episodes=150000, alpha_init=0.1):
    S_CNT, A_CNT = env.nS, env.nA
    res = []
    q = np.zeros((S_CNT, A_CNT), dtype=np.float32)
    state = env.reset()
    np.set_printoptions(precision=3, suppress=True)
    _lambda = 0.2
    alpha = alpha_init
    scores = []
    eps = 1.0
    k = 1
    episodes = 0
    E = np.zeros((S_CNT, A_CNT), dtype=np.float32)
    sum_reward = 0
    len_episod = 0
    while episodes < num_episodes:
        policy = policy_q_eps_greedy(q, eps)
        action = np.random.choice(A_CNT, p=policy[state, :])
        n_state, reward, done, info = env.step(action)
        n_action = np.random.choice(A_CNT, p=policy[n_state, :])
        delta = reward + gamma * q[n_state, n_action] - q[state, action]
        E[state, action] += 1.0
        q += alpha * delta * E
        E *= gamma * _lambda  
        sum_reward += reward
        len_episod += 1
        if done or len_episod > 640:
            state = env.reset()
            scores.append(sum_reward)
            if len(scores) > 100:
                scores = scores[1:]
            episodes += 1
            if episodes % 100 == 0:
                res.append(np.mean(scores))
            if episodes % 2000 == 0:
                print('Episodes: %d. Score: %.2f Alpha: %.5g Eps: %.5g' % (episodes, np.mean(scores), alpha, eps))
                res.append(np.mean(scores))
                k += 1
                eps *= 0.3
                alpha = alpha_init / k
                _lambda = 1.0 - 1.0 / k
            E = np.zeros((S_CNT, A_CNT), dtype=np.float32)
            sum_reward = 0
            len_episod = 0
        else:
            state = n_state
        if len(res)!=0 and res[-1] >= win:
            break
            
    return res, policy

def run(env, win, alpha_init=0.1):
    scores, policy = sarsa_lambda(env, win, alpha_init=alpha_init)
    return scores, policy
