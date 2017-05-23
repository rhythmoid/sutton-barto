import numpy as np
import matplotlib.pyplot as plt

class nBandit_ReinforcementComparison:
    def __init__(self, n_arm=10, n_trial=1000, alpha=0.1, beta=0.01):
        """
        n本腕バンディット, 強化比較手法
        n_arm : 腕の本数
        n_trial : 腕を引く回数
        alpha : リファレンス報酬更新におけるステップサイズ・パラメータ
        beta : 優先度更新におけるステップサイズ・パラメータ
        """
        self.n_arm = n_arm
        self.n_trial = n_trial
        self.alpha = alpha
        self.beta = beta

    def arm_set(self):
        for _ in range(self.n_arm):
            self.reward_mean_ = range(1, self.n_arm+1)

    def play(self, adjust):
        """
        ref_ : リファレンス報酬
        p : 優先度
        adjust : 補正項(1-π(a))をつけるかどうか
        """
        self.ref_ = 0
        self.p_ = np.zeros(self.n_arm)
        self.reward_mean_list_ = []
        for i_iter in range(self.n_trial):
            self.arm_ = self.act(self.p_)
            self.reward_ = self.reward_mean_[self.arm_] + np.random.randn()
            if adjust:
                self.p_[self.arm_] += self.alpha * ( 1 - self.pi(self.p_)[self.arm_] )
            self.p_[self.arm_] += self.beta * (self.reward_ - self.ref_)
            self.ref_ += self.alpha * (self.reward_ - self.ref_)
            if i_iter == 0:
                self.reward_mean_list_.append(self.reward_)
            else:
                self.m_ = self.reward_mean_list_[-1]
                self.reward_mean_list_.append(self.m_ + 1/(i_iter+1) * (self.reward_ - self.m_))
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

b = nBandit_ReinforcementComparison(n_arm=10, n_trial=10000, alpha=0.1, beta=0.01)

b.arm_set()
rew1 = b.play(adjust=True)
rew2 = b.play(adjust=False)

#調整しないほうがいい結果が出ているような……
plt.plot(rew1, label='adjusted')
plt.plot(rew2, label='not adjusted')
plt.xlabel('iteration')
plt.ylabel('reward')
plt.legend()
plt.show()
