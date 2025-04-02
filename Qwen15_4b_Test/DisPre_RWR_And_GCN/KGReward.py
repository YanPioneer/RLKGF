class KGReward:
    def __init__(self, gcnreward, rwrreward, mu):
        self.gcnreward = gcnreward
        self.rwrreward = rwrreward
        self.mu = mu

    def score(self, goal):
        gcn_dis, _ = self.gcnreward.score(goal)
        rwr_dis, _ = self.rwrreward.score(goal)
        # mu = 0.01
        mu = self.mu
        dis_concat = mu * gcn_dis + (1 - mu) * rwr_dis

        return dis_concat, None


