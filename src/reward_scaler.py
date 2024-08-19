class RewardScaler:
    def __init__(self, scale=0.01, use_scaling=True):
        self.scale = scale
        self.use_scaling = use_scaling

    def scale_reward(self, reward):
        if self.use_scaling:
            return reward * self.scale
        return reward

    def unscale_reward(self, scaled_reward):
        if self.use_scaling:
            return scaled_reward / self.scale
        return scaled_reward