import pandas as pd
import numpy as np
import imageio
from pathlib import Path
from IPython.display import HTML
from src.environment import ImprovedEnvWrapper
from src.reward_scaler import RewardScaler
import matplotlib.pyplot as plt

def ewma(x, span=100):
    return pd.DataFrame({'x': np.asarray(x)}).x.ewm(span=span).mean().values

def generate_video(agent, env, filename='kungfu_video.mp4'):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    done = False
    total_reward = 0

    while not done:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state
        total_reward += reward
        done = terminated or truncated

    env.close()
    print(f"Total reward: {total_reward}")
    print(f"Video saved as {filename}")

def display_video(video_path):
    video_path = Path(video_path)
    if video_path.exists():
        return HTML("""
        <video width="640" height="480" controls>
          <source src="{}" type="video/mp4">
        </video>
        """.format(video_path))
    else:
        print("Video file not found. Please check if the video was generated successfully.")

def evaluate_agent(agent, env, n_games=3):
    game_rewards = []
    for _ in range(n_games):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            step_result = env.step(action)
            if len(step_result) == 5:  # New Gymnasium version
                state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:  # Older version
                state, reward, done, _ = step_result
            total_reward += reward
        game_rewards.append(total_reward)
    return game_rewards
        
def compare_agent_results(rewards_history, rewards_history_exploding, improved_rewards,
                          entropy_history, entropy_history_exploding, improved_entropy,
                          iterations_history, iterations_exploding, improved_iterations,
                          save_path=None):
    
    reward_scaler = RewardScaler(scale=0.01, use_scaling=True)
    
    plt.figure(figsize=(20, 10))
    
    # Rewards Comparison
    plt.subplot(2, 2, 1)
    plt.plot(np.arange(len(rewards_history)) * 2500, rewards_history, label='Initial')
    plt.plot(np.arange(len(rewards_history_exploding)) * 2500, rewards_history_exploding, label='Exploding')
    plt.plot(np.arange(len(improved_rewards)) * 2500, improved_rewards, label='Improved')
    plt.title('Rewards Comparison (Unscaled)')
    plt.xlabel('Iterations')
    plt.ylabel('Rewards')
    plt.legend()

    # Scaled Rewards Comparison
    plt.subplot(2, 2, 2)
    plt.plot(np.arange(len(rewards_history)) * 2500, reward_scaler.scale_reward(np.array(rewards_history)), label='Initial (Scaled)')
    plt.plot(np.arange(len(rewards_history_exploding)) * 2500, reward_scaler.scale_reward(np.array(rewards_history_exploding)), label='Exploding (Scaled)')
    plt.plot(np.arange(len(improved_rewards)) * 2500, reward_scaler.scale_reward(np.array(improved_rewards)), label='Improved (Scaled)')
    plt.title('Rewards Comparison (Scaled)')
    plt.xlabel('Iterations')
    plt.ylabel('Scaled Rewards')
    plt.legend()

    # Entropy Comparison
    plt.subplot(2, 2, 3)
    plt.plot(iterations_history, entropy_history, label='Initial')
    plt.plot(iterations_exploding, entropy_history_exploding, label='Exploding')
    plt.plot(improved_iterations, improved_entropy, label='Improved')
    plt.title('Entropy Comparison')
    plt.xlabel('Iterations')
    plt.ylabel('Entropy')
    plt.legend()

    # Adjust layout and display the plot
    plt.tight_layout()
    
    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    # Display the plot
    plt.show()

    # Additional Analysis
    print("Performance Analysis:")
    print(f"Initial Agent - Final Reward: {rewards_history[-1]:.2f}, Final Entropy: {entropy_history[-1]:.4f}")
    print(f"Exploding Agent - Final Reward: {rewards_history_exploding[-1]:.2f}, Final Entropy: {entropy_history_exploding[-1]:.4f}")
    print(f"Improved Agent - Final Reward: {improved_rewards[-1]:.2f}, Final Entropy: {improved_entropy[-1]:.4f}")

    print("\nLearning Speed Analysis:")
    initial_learning_speed = (rewards_history[-1] - rewards_history[0]) / len(rewards_history)
    exploding_learning_speed = (rewards_history_exploding[-1] - rewards_history_exploding[0]) / len(rewards_history_exploding)
    improved_learning_speed = (improved_rewards[-1] - improved_rewards[0]) / len(improved_rewards)

    print(f"Initial Agent - Learning Speed: {initial_learning_speed:.4f} reward/iteration")
    print(f"Exploding Agent - Learning Speed: {exploding_learning_speed:.4f} reward/iteration")
    print(f"Improved Agent - Learning Speed: {improved_learning_speed:.4f} reward/iteration")