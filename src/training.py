import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange
from IPython.display import clear_output
import matplotlib.pyplot as plt
from src.utils import ewma
from src.environment import make_env, EnvBatch, make_improved_env, ImprovedEnvBatch, ImprovedEnvWrapper
from src.agent import Agent
from src.reward_scaler import RewardScaler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def compute_actor_critic_loss(agent, state, action, reward, next_state, done, gamma=0.99):
    batch_size = state.shape[0]

    state      = torch.tensor(state, dtype=torch.float32, device=device)
    next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
    reward     = torch.tensor(reward, dtype=torch.float32, device=device)
    done       = torch.tensor(done, dtype=torch.bool, device=device)

    logits, state_value = agent(state)
    next_logits, next_state_value = agent(next_state)

    probs    = F.softmax(logits, dim=-1)
    logprobs = F.log_softmax(logits, dim=-1)

    target_state_value = reward + gamma * next_state_value * (~done)
    advantage = target_state_value - state_value

    batch_idx = np.arange(batch_size)
    logp_actions = logprobs[batch_idx, action]

    entropy = -torch.sum(probs * logprobs, axis=-1)

    actor_loss = -(logp_actions * advantage.detach()).mean() - 0.001 * entropy.mean()
    critic_loss = F.mse_loss(target_state_value.detach(), state_value)

    total_loss = actor_loss + critic_loss

    agent.optimizer.zero_grad()
    total_loss.backward()
    agent.optimizer.step()

    return actor_loss.cpu().detach().numpy(), critic_loss.cpu().detach().numpy(), entropy.cpu().detach().numpy()

def improved_compute_actor_critic_loss(agent, state, action, reward, next_state, done, gamma=0.99):
    batch_size = state.shape[0]
    state = torch.tensor(state, dtype=torch.float32, device=device)
    next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
    reward = torch.tensor(reward, dtype=torch.float32, device=device)
    done = torch.tensor(done, dtype=torch.bool, device=device)

    logits, state_value = agent(state)
    next_logits, next_state_value = agent(next_state)

    probs = F.softmax(logits, dim=-1)
    logprobs = F.log_softmax(logits, dim=-1)

    target_state_value = reward + gamma * next_state_value * (~done)
    advantage = target_state_value.detach() - state_value

    batch_idx = np.arange(batch_size)
    logp_actions = logprobs[batch_idx, action]

    entropy = -torch.sum(probs * logprobs, axis=-1)

    actor_loss = -(logp_actions * advantage.detach()).mean() - 0.01 * entropy.mean()
    critic_loss = F.mse_loss(target_state_value.detach(), state_value)

    total_loss = actor_loss + critic_loss

    agent.optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
    agent.optimizer.step()

    return actor_loss.cpu().detach().numpy(), critic_loss.cpu().detach().numpy(), entropy.cpu().detach().numpy()

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
                state, reward, done, truncated, _ = step_result
                done = done or truncated
            else:  # Older version
                state, reward, done, _ = step_result
            total_reward += reward
            
        if isinstance(env, ImprovedEnvWrapper):
            reward_scaler = RewardScaler(scale=0.01, use_scaling=True)
            total_reward = reward_scaler.unscale_reward(total_reward)    
            
        game_rewards.append(total_reward)
    return game_rewards

def train_agent(n_iterations=20000, eval_interval=2500, entropy_interval=500):
    env_batch = EnvBatch(10)
    agent = Agent(input_dims=env_batch.envs[0].observation_space.shape, n_actions=env_batch.envs[0].action_space.n, lr=1e-4)
    batch_states = env_batch.reset()
    reward_scaler = RewardScaler(scale=0.01, use_scaling=True)

    rewards_history = []
    entropy_history = []
    iterations = []

    with trange(n_iterations) as progress_bar:
        for i in progress_bar:
            batch_actions = agent.choose_action(batch_states)
            batch_next_states, batch_rewards, batch_done, _ = env_batch.step(batch_actions)

            batch_rewards = reward_scaler.scale_reward(batch_rewards)

            agent_loss, critic_loss, entropy = compute_actor_critic_loss(
                agent, batch_states, batch_actions, batch_rewards, batch_next_states, batch_done)
            
            if i % entropy_interval == 0:
                entropy_history.append(np.mean(entropy))
                iterations.append(i)

            batch_states = batch_next_states

            if i % eval_interval == 0:
                eval_rewards = evaluate_agent(agent, make_env(), n_games=3)
                rewards_history.append(np.mean(eval_rewards))
                if rewards_history[-1] >= 5000:
                    print("Your agent has earned the yellow belt")

                clear_output(True)
                plot_training_progress(rewards_history, entropy_history, iterations, eval_interval, entropy_interval, reward_scaler)

    return agent, rewards_history, entropy_history, iterations

def train_agent_exploding_gradient(n_iterations=20000, eval_interval=2500, entropy_interval=500):
    env_batch = EnvBatch(10)
    agent = Agent(input_dims=env_batch.envs[0].observation_space.shape, n_actions=env_batch.envs[0].action_space.n, lr=1e-4)
    batch_states = env_batch.reset()
    reward_scaler = RewardScaler(scale=0.01, use_scaling=False)

    rewards_history = []
    entropy_history = []
    iterations = []

    with trange(n_iterations) as progress_bar:
        for i in progress_bar:
            batch_actions = agent.choose_action(batch_states)
            batch_next_states, batch_rewards, batch_done, _ = env_batch.step(batch_actions)

            batch_rewards = reward_scaler.scale_reward(batch_rewards)

            agent_loss, critic_loss, entropy = compute_actor_critic_loss(
                agent, batch_states, batch_actions, batch_rewards, batch_next_states, batch_done)
            
            if i % entropy_interval == 0:
                entropy_history.append(np.mean(entropy))
                iterations.append(i)

            batch_states = batch_next_states

            if i % eval_interval == 0:
                eval_rewards = evaluate_agent(agent, make_env(), n_games=3)
                rewards_history.append(np.mean(eval_rewards))
                if rewards_history[-1] >= 5000:
                    print("Your agent has earned the yellow belt")

                clear_output(True)
                plot_training_progress(rewards_history, entropy_history, iterations, eval_interval, entropy_interval)

    return agent, rewards_history, entropy_history, iterations

def train_improved_agent(n_iterations=400000, eval_interval=2500, entropy_interval=500):
    env_batch = ImprovedEnvBatch(10)
    agent = Agent(input_dims=env_batch.envs[0].observation_space.shape, n_actions=env_batch.envs[0].action_space.n, lr=1e-4)
    batch_states = env_batch.reset()

    rewards_history = []
    entropy_history = []
    iterations = []

    with trange(n_iterations) as progress_bar:
        for i in progress_bar:
            batch_actions = agent.choose_action(batch_states)
            batch_next_states, batch_rewards, batch_done, _ = env_batch.step(batch_actions)

            agent_loss, critic_loss, entropy = improved_compute_actor_critic_loss(
                agent, batch_states, batch_actions, batch_rewards, batch_next_states, batch_done)
            
            if i % entropy_interval == 0:
                entropy_history.append(np.mean(entropy))
                iterations.append(i)

            batch_states = batch_next_states

            if i % eval_interval == 0:
                eval_rewards = evaluate_agent(agent, make_improved_env(), n_games=3)
                rewards_history.append(np.mean(eval_rewards))
                if rewards_history[-1] >= 5000:
                    print("Your agent has earned the yellow belt")

                clear_output(True)
                plot_training_progress(rewards_history, entropy_history, iterations, eval_interval, entropy_interval)

    return agent, rewards_history, entropy_history, iterations

def plot_training_progress(rewards_history, entropy_history, iterations, eval_interval, entropy_interval, reward_scaler=None):
    plt.figure(figsize=[16, 6])
    plt.subplot(1, 2, 1)
    
    plot_rewards = np.array(rewards_history)
    
    plt.plot(np.arange(len(rewards_history)) * eval_interval, plot_rewards, label='rewards')
    plt.plot(np.arange(len(rewards_history)) * eval_interval, 
             ewma(plot_rewards, span=10), marker='.', label='rewards ewma@10')
    plt.title("Session rewards")
    plt.xlabel("Iterations")
    plt.ylabel("Rewards")
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(iterations, entropy_history, label='entropy')
    plt.plot(iterations, 
             ewma(np.array(entropy_history), span=1000), marker='.', label='entropy ewma@1000')
    plt.title("Policy entropy")
    plt.xlabel("Iterations")
    plt.ylabel("Entropy")
    plt.grid()
    plt.legend()
    plt.show()