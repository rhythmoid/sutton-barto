import numpy as np
import matplotlib.pyplot as plt

class nBandit:
    def __init__(self, n_arm=10, n_trial=1000):
        """
        n本腕バンディット,
        n_arm : 腕の本数
        n_trial : 腕を引く回数
        """
        self.n_arm = n_arm
        self.n_trial = n_trial
        self.reward_mean_ = range(1, self.n_arm+1)

    def add_reward(self, r_list, r):
        if len(r_list) == 0:
            r_list.append(r)
        else:
            r_list.append(r_list[-1] + 1/(len(r_list)+1) * (r - r_list[-1]))

    def update_Q(self, Q, k, a, r):
        Q[a] += 1/(k[a] + 1) * (r - Q[a])
        k[a] += 1

class epsilon_Greedy(nBandit):
    def __init__(self, n_arm=10, n_trial=1000, epsilon=0.1):
        """
        ε-Greedy
        epsilon : ε
        """
        super().__init__(n_arm, n_trial)
        self.epsilon = epsilon

    def play(self):
        self.reward_mean_list_ = []
        self.Q_ = np.zeros(self.n_arm)
        self.k_ = np.zeros(self.n_arm)
        for i_iter in range(self.n_trial):
            if np.random.rand() < self.epsilon:
                arm = np.random.randint(self.n_arm)
            else:
                arm = np.argmax(self.Q_)
            rew = self.reward_mean_[arm] + np.random.randn()
            self.update_Q(self.Q_, self.k_, arm, rew)
            self.add_reward(self.reward_mean_list_, rew)
        return self.reward_mean_list_

class ReinforcementComparison(nBandit):
    def __init__(self, n_arm=10, n_trial=1000, alpha=0.1, beta=0.01):
        """
        強化比較手法
        alpha : リファレンス報酬更新におけるステップサイズ・パラメータ
        beta : 優先度更新におけるステップサイズ・パラメータ
        """
        super().__init__(n_arm, n_trial)
        self.alpha = alpha
        self.beta = beta

    def play(self):
        """
        ref_ : リファレンス報酬
        p : 優先度
        """
        self.ref_ = 0
        self.p_ = np.zeros(self.n_arm)
        self.reward_mean_list_ = []
        for i_iter in range(self.n_trial):
            self.arm_ = self.act(self.p_)
            self.reward_ = self.reward_mean_[self.arm_] + np.random.randn()
            self.p_[self.arm_] += self.beta * (self.reward_ - self.ref_)
            self.ref_ += self.alpha * (self.reward_ - self.ref_)
            self.add_reward(self.reward_mean_list_, self.reward_)
        return self.reward_mean_list_

    def pi(self, p):
        return np.exp(p) / sum(np.exp(p))

    def act(self, p):
        """soft-max手法による行動の選択"""
        self.pi_ = self.pi(p)
        self.F_ = np.zeros(len(self.pi_))
        self.F_[0] = self.pi_[0]
        for i in range(len(p)-1):
            self.F_[i+1] = self.F_[i] + self.pi_[i+1]
        x_rand = np.random.rand()
        for i in range(len(p)):
            if x_rand < self.F_[i]:
                return i

class Pursuit(nBandit):
    def __init__(self, n_arm=10, n_trial=1000, beta=0.01):
        """
        追跡手法
        beta : 優先度更新におけるステップサイズ・パラメータ
        """
        super().__init__(n_arm, n_trial)
        self.beta = beta

    def play(self):
        """
        p : 優先度
        """
        self.p_ = np.zeros(self.n_arm)
        self.reward_mean_list_ = []
        self.Q_ = np.zeros(self.n_arm)
        self.k_ = np.zeros(self.n_arm)
        for i_iter in range(self.n_trial):
            self.arm_ = self.act(self.p_)
            self.reward_ = self.reward_mean_[self.arm_] + np.random.randn()
            if self.arm_ == np.argmax(self.Q_):
                self.target_ = self.reward_
            else:
                self.target_ = 0
            self.p_[self.arm_] += self.beta * (self.target_ - self.p_[self.arm_])
            self.update_Q(self.Q_, self.k_, self.arm_, self.reward_)
            self.add_reward(self.reward_mean_list_, self.reward_)
        return self.reward_mean_list_

    def pi(self, p):
        return np.exp(p) / sum(np.exp(p))

    def act(self, p):
        """soft-max手法による行動の選択"""
        self.pi_ = self.pi(p)
        self.F_ = np.zeros(len(self.pi_))
        self.F_[0] = self.pi_[0]
        for i in range(len(p)-1):
            self.F_[i+1] = self.F_[i] + self.pi_[i+1]
        x_rand = np.random.rand()
        for i in range(len(p)):
            if x_rand < self.F_[i]:
                return i

eg = epsilon_Greedy(n_arm=10, n_trial=10000, epsilon=0.1)
rc = ReinforcementComparison(n_arm=10, n_trial=10000, alpha=0.1, beta=0.01)
pu = Pursuit(n_arm=10, n_trial=10000, beta=0.01)

rew1 = eg.play()
rew2 = rc.play()
rew3 = pu.play()

plt.plot(rew1, label="$\epsilon$ -Greedy")
plt.plot(rew2, label="Reinforcement Comparison")
plt.plot(rew3, label="Pursuit")
plt.legend()
plt.show()
