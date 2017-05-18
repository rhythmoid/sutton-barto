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
        self.reward_mean_ = np.ones(self.n_arm)


    def arm_set(self):
        """
        ランダムウォークによって報酬の真値を更新
        """
        self.reward_mean_ += np.random.randint(0,2,self.n_arm) * 2 - 1

    def eps_greedy(self, sample_averaging):
        """
        ε-Greedy方策
        epsilon = 0.1
        sample_averaging : 標本平均手法をとるかどうか
        """
        got_reward = []
        got_reward_sum = []
        got_reward_mean = []
        Q = np.zeros(self.n_arm)

        self.arm_set()
        for _iter in range(self.n_trial):
            if np.random.rand() < 0.1:
                selected_arm = np.random.randint(self.n_arm)
            else:
                selected_arm = np.argmax(Q)
            reward = self.reward_mean_[selected_arm] + 3 * np.random.randn()

            if sample_averaging:
                alpha = 1 / (_iter + 1)
            else:
                alpha = 0.1
            Q[selected_arm] += alpha * (reward - Q[selected_arm])

            got_reward.append(reward)
            if _iter:
                got_reward_sum.append(got_reward_sum[-1] + reward)
            else:
                got_reward_sum = [reward]
            got_reward_mean.append(got_reward_sum[-1]/(_iter+1))
            self.arm_set()
        return got_reward_mean

b1 = nBandit(n_arm=10, n_trial=10000)

p1 = b1.eps_greedy(sample_averaging=True)
p2 = b1.eps_greedy(sample_averaging=False)

plt.plot(p1, label="alpha = 1/k")
plt.plot(p2, label="alpha = Const.")
plt.legend()
plt.show()
