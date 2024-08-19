import os
import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.wrappers import RecordVideo
from gymnasium.wrappers.monitoring import video_recorder
import numpy as np
from src.atari_util import PreprocessAtari
from collections import deque
from src.reward_scaler import RewardScaler

class CompatibilityWrapper(Wrapper):
    def step(self, action):
        step_result = self.env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            return obs, reward, done, False, info
        return step_result

    def reset(self, **kwargs):
        try:
            return self.env.reset(**kwargs)
        except TypeError:
            return self.env.reset()

class FlexibleRecordVideo(RecordVideo):
    def __init__(self, env, video_folder, name_prefix="rl-video"):
        super().__init__(env, video_folder, name_prefix=name_prefix)
        self.episode_count = 0
        self.recording = False
        self.video_folder = video_folder
        self.name_prefix = name_prefix

    def reset(self, **kwargs):
        self.episode_count += 1
        print(f"Reset called, preparing to record video for episode {self.episode_count}...")
        
        if self.recording:
            self.close_video_recorder()
        
        obs = super().reset(**kwargs)
        
        self.start_video_recorder()
        self.recording = True
        
        return obs

    def start_video_recorder(self):
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env,
            base_path=os.path.join(self.video_folder, f"{self.name_prefix}-episode-{self.episode_count}"),
            metadata={"episode_id": self.episode_count}
        )

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        
        if terminated or truncated:
            self.close_video_recorder()
            self.recording = False
        
        return observation, reward, terminated, truncated, info

    def close_video_recorder(self):
        if self.video_recorder:
            self.video_recorder.close()
        self.video_recorder = None
        self.recording = False

    def close(self):
        super().close()
        print(f"Total episodes recorded: {self.episode_count}")

class ImprovedEnvWrapper(Wrapper):
    def __init__(self, env, window_size=10):
        super().__init__(env)
        self.window_size = window_size
        self.action_history = deque(maxlen=5)
        self.consecutive_same_action_count = 0
        
        self.noop_action = [0]
        self.movement_actions = [2, 3, 5, 6]
        self.defensive_actions = [1, 4]
        self.fire_actions = [7, 8, 9, 10, 11, 12, 13]
        
        self.steps_in_current_episode = 0
        self.current_score = 0
        self.highest_score = 0
        self.score_threshold_1 = 10 # Represents score of 1000
        self.score_threshold_2 = 50 # Represents score of 5000
        self.score_threshold_3 = 100 # Represents score of 10000

        self.reward_scaler = RewardScaler(scale=0.01, use_scaling=True)
        self.lives = 4  # Initialize lives
        self.previous_lives = 4
        self.steps_since_last_kill = 0
        self.last_kill_step = 0
        self.combo_count = 0

    def step(self, action):
            step_result = self.env.step(action)
            
            # Handle both 4-value and 5-value returns
            if len(step_result) == 4:
                observation, reward, done, info = step_result
                terminated, truncated = done, False
            else:
                observation, reward, terminated, truncated, info = step_result

            self.update_action_history(action)
            self.steps_in_current_episode += 1
            self.current_score += reward
            
            modified_reward = self._modify_reward(reward, action, info)
            
            scaled_reward = self.reward_scaler.scale_reward(modified_reward)
            
            # Update lives based on the info
            self.previous_lives = self.lives
            self.lives = info.get('lives', self.lives)
            
            # Create our own info dictionary
            custom_info = {
                'lives': self.lives,
                'episode_frame_number': self.steps_in_current_episode * 4,  # Assuming frame skip of 4
                'frame_number': self.steps_in_current_episode * 4,  # This might need adjustment if you're tracking total frames across episodes
                'score': self.current_score
            }
            
            if terminated or truncated:
                self.reset_episode_specific_data()
            
            return observation, scaled_reward, terminated, truncated, custom_info

    def reset(self, **kwargs):
        self.reset_episode_specific_data()
        reset_result = self.env.reset(**kwargs)
        
        # Handle both single value and tuple returns
        if isinstance(reset_result, tuple):
            observation, info = reset_result
        else:
            observation, info = reset_result, {}
        
        self.lives = 4  # Reset lives at the start of each episode
        self.previous_lives = 4
        return observation, info

    def reset_episode_specific_data(self):
        self.steps_in_current_episode = 0
        self.current_score = 0
        self.action_history.clear()
        self.consecutive_same_action_count = 0
        self.steps_since_last_kill = 0
        self.last_kill_step = 0
        self.combo_count = 0

    def update_action_history(self, action):
        if len(self.action_history) > 0 and action == self.action_history[-1]:
            self.consecutive_same_action_count += 1
        else:
            self.consecutive_same_action_count = 1
        self.action_history.append(action)

    def _modify_reward(self, original_reward, action, info):
        modified_reward = original_reward

        # Enemy Defeat Bonus
        if original_reward > 0:
            modified_reward += min(original_reward * 0.1, 1.0)  # Bonus scaled to enemy strength
            self.steps_since_last_kill = 0
            self.last_kill_step = self.steps_in_current_episode
            self.combo_count += 1
        else:
            self.steps_since_last_kill += 1

        # Life Preservation
        if self.lives < self.previous_lives:
            modified_reward -= 5.0

        # Active Gameplay Encouragement
        if 0 < self.steps_in_current_episode - self.last_kill_step <= 50:
            modified_reward += 0.1

        # Exploration Incentive
        if self.steps_since_last_kill >= 500:
            modified_reward -= 0.05

        # Combo Reward
        if self.combo_count > 1:
            modified_reward += 0.5

        # Additional reward shaping from the original implementation
        if action in self.fire_actions:
            if self.current_score < self.score_threshold_1:
                if original_reward > 0:
                    modified_reward += 0.05
            elif self.score_threshold_1 <= self.current_score < self.score_threshold_2:
                if original_reward <= 0:
                    modified_reward -= 0.01
            elif self.score_threshold_2 <= self.current_score < self.score_threshold_3:
                if original_reward <= 0:
                    modified_reward -= 0.05
            else:
                if original_reward <= 0:
                    modified_reward -= 0.1

        if original_reward == 0 and action not in self.noop_action:
            if self.current_score < self.score_threshold_1:
                modified_reward -= 0.005
            elif self.score_threshold_1 <= self.current_score < self.score_threshold_2:
                modified_reward -= 0.01
            elif self.score_threshold_2 <= self.current_score < self.score_threshold_3:
                modified_reward -= 0.05
            else:
                modified_reward -= 0.1

        if self.current_score > self.highest_score:
            modified_reward += 0.1
            self.highest_score = self.current_score

        if self.current_score >= self.score_threshold_2:
            if action in self.defensive_actions:
                modified_reward += 0.02

        if self.current_score >= self.score_threshold_2:
            if self.consecutive_same_action_count > 3:
                if action in self.fire_actions:
                    modified_reward -= 0.005
                elif action in self.movement_actions:
                    modified_reward -= 0.01
                elif action in self.defensive_actions:
                    modified_reward -= 0.02

        return modified_reward

def make_improved_env(video_folder=None, video_name=None):
    env = gym.make("KungFuMasterDeterministic-v0", render_mode="rgb_array")
    env = PreprocessAtari(
        env, height=42, width=42,
        crop=lambda img: img[60:-30, 5:],
        dim_order='pytorch',
        color=False, n_frames=4)
    env = ImprovedEnvWrapper(env)
    
    if video_folder:
        name_prefix = video_name.rsplit('.', 1)[0] if video_name else "rl-video"
        env = FlexibleRecordVideo(env, video_folder=video_folder, name_prefix=name_prefix)
        
    return env

def make_env(video_folder=None, video_name=None):
    env = gym.make("KungFuMasterDeterministic-v0", render_mode="rgb_array")
    env = PreprocessAtari(
        env, height=42, width=42,
        crop=lambda img: img[60:-30, 5:],
        dim_order='pytorch',
        color=False, n_frames=4)
    env = CompatibilityWrapper(env)
    
    if video_folder:
        name_prefix = video_name.rsplit('.', 1)[0] if video_name else "rl-video"
        env = FlexibleRecordVideo(env, video_folder=video_folder, name_prefix=name_prefix)
    
    return env

class EnvBatch:
    def __init__(self, n_envs=10):
        self.envs = [make_env() for _ in range(n_envs)]
        
    def reset(self):
        results = [env.reset() for env in self.envs]
        if isinstance(results[0], tuple):  # New format (observation, info)
            observations, _ = zip(*results)
        else:  # Old format (just observation)
            observations = results
        return np.array(observations)
    
    def step(self, actions):
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        if len(results[0]) == 5:  # New Gymnasium version
            new_obs, rewards, terminated, truncated, infos = map(list, zip(*results))
            done = [t or tr for t, tr in zip(terminated, truncated)]
        else:  # Older version
            new_obs, rewards, done, infos = map(list, zip(*results))
        
        for i in range(len(self.envs)):
            if done[i]:
                reset_result = self.envs[i].reset()
                if isinstance(reset_result, tuple):
                    new_obs[i] = reset_result[0]
                else:
                    new_obs[i] = reset_result
        
        return np.array(new_obs), np.array(rewards), np.array(done), infos

class ImprovedEnvBatch:
    def __init__(self, n_envs = 10):
        self.envs = [make_improved_env() for _ in range(n_envs)]
        
    def reset(self):
        results = [env.reset() for env in self.envs]
        observations, infos = zip(*results)
        return np.array(observations)
    
    def step(self, actions):
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        new_obs, rewards, terminated, truncated, infos = map(np.array, zip(*results))
        
        done = np.logical_or(terminated, truncated)
        
        for i in range(len(self.envs)):
            if done[i]:
                new_obs[i] = self.envs[i].reset()[0]  # Only take the observation, not the info
        
        return new_obs, rewards, done, infos