import numpy as np
import matplotlib.pyplot as plt

class nBandit:
    def __init__(self, n_arm, n_trial):
        """
        n本腕バンディット問題
        n_arm : アームの本数
        n_trial : アームを引く回数
        """
        self.n_arm = n_arm
        self.n_trial = n_trial

    def arm_set(self):
        self.reward_mean_ = list(range(self.n_arm))

    def eps_greedy(self, epsilon):
        """
        ε-Greedy方策
        epsilon : ランダムにアームを選ぶ確率
        """
        got_reward = []
        got_reward_sum = []
        got_reward_mean = []
        score_sum_for_arms = np.zeros(self.n_arm)
        score_mean_for_arms = np.zeros(self.n_arm)
        selected_num_for_arms = np.zeros(self.n_arm)
        for _iter in range(self.n_trial):
            if np.random.rand() < epsilon:
                selected_arm = np.random.randint(self.n_arm)
            else:
                selected_arm = np.argmax(score_mean_for_arms)
            reward = self.reward_mean_[selected_arm] + 3 * np.random.randn()
            score_sum_for_arms[selected_arm] += reward
            selected_num_for_arms[selected_arm] += 1
            score_mean_for_arms[selected_arm] = score_sum_for_arms[selected_arm]/selected_num_for_arms[selected_arm]
            got_reward.append(reward)
            if _iter:
                got_reward_sum.append(got_reward_sum[-1] + reward)
            else:
                got_reward_sum = [reward]
            got_reward_mean.append(got_reward_sum[-1]/(_iter+1))
        return got_reward_mean

    def soft_max(self, tau):
        """
        SofMax方策
        tau : 温度
        """
        got_reward = []
        got_reward_sum = []
        got_reward_mean = []
        score_sum_for_arms = np.zeros(self.n_arm)
        score_mean_for_arms = np.zeros(self.n_arm)
        selected_num_for_arms = np.zeros(self.n_arm)
        for _iter in range(self.n_trial):
            exp_Q_tau = np.exp(score_mean_for_arms/tau)
            prob_func = exp_Q_tau/sum(exp_Q_tau)
            dens_func = [prob_func[0]]
            for i in range(self.n_arm):
                dens_func.append(dens_func[-1] + prob_func[i])
            u = np.random.rand()
            for i in range(self.n_arm):
                if u <= dens_func[i]:
                    selected_arm = i
                    break
            reward = self.reward_mean_[selected_arm] + 3 * np.random.randn()
            score_sum_for_arms[selected_arm] += reward
            selected_num_for_arms[selected_arm] += 1
            score_mean_for_arms[selected_arm] = score_sum_for_arms[selected_arm]/selected_num_for_arms[selected_arm]
            got_reward.append(reward)
            if _iter:
                got_reward_sum.append(got_reward_sum[-1] + reward)
            else:
                got_reward_sum = [reward]
            got_reward_mean.append(got_reward_sum[-1]/(_iter+1))
        return got_reward_mean

b = nBandit(n_arm=10, n_trial=1000)

b.arm_set()

p0 = b.eps_greedy(epsilon=0)
p1 = b.eps_greedy(epsilon=0.1)
p2 = b.eps_greedy(epsilon=0.01)
p3 = b.soft_max(tau=0.1)
p4 = b.soft_max(tau=0.01)

plt.plot(p0, label="greedy")
plt.plot(p1, label="eps-greedy 0.1")
plt.plot(p2, label="eps-greedy 0.01")
plt.plot(p3, label="softmax 0.1")
plt.plot(p4, label="softmax 0.01")
plt.legend()
plt.show()
