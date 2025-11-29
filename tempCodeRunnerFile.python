"""
Train DQN/Double DQN using official Gymnasium LunarLander-v3
Uses the official implementation instead of custom environment
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import gymnasium as gym
import os
import cv2
from datetime import datetime


class DQN(nn.Module):
    """Deep Q-Network"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network

    Separates Q-value into Value and Advantage streams:
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))

    Key advantages:
    - Better state value estimation
    - Learns which states are valuable independent of actions
    - More stable learning for many similar-valued actions

    Reference: Wang et al. (2016) "Dueling Network Architectures for Deep RL"
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DuelingDQN, self).__init__()

        # Shared feature layers
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Advantage stream: A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(self, x):
        # Extract features
        features = self.feature(x)

        # Get value and advantages
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine using dueling architecture formula
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values


class ReplayBuffer:
    """Experience Replay Buffer"""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent (Vanilla DQN)"""

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.algorithm = "DQN"

        # Q-networks
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax(1).item()

    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Decay epsilon"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Save model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filepath)

    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']


class DoubleDQNAgent(DQNAgent):
    """
    Double DQN Agent

    Key difference from vanilla DQN:
    - Uses online network to SELECT the best action
    - Uses target network to EVALUATE that action
    - This reduces overestimation bias in Q-value estimation
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64,
    ):
        super().__init__(
            state_dim,
            action_dim,
            lr,
            gamma,
            epsilon_start,
            epsilon_end,
            epsilon_decay,
            buffer_capacity,
            batch_size,
        )
        self.algorithm = "Double DQN"

    def train_step(self):
        """Perform one training step using Double DQN algorithm"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Double DQN: Use online network to select actions, target network to evaluate
        with torch.no_grad():
            # Select best actions using online network
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)

            # Evaluate selected actions using target network
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)

            # Compute target Q values
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class DuelingDQNAgent(DQNAgent):
    """
    Dueling DQN Agent

    Combines Dueling architecture with standard DQN training.
    Can also be combined with Double DQN for best results.

    Key differences from vanilla DQN:
    - Uses DuelingDQN network (separate Value and Advantage streams)
    - Better state value estimation
    - More stable learning

    Reference: Wang et al. (2016) "Dueling Network Architectures for Deep RL"
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64,
    ):
        # Initialize parent (but we'll replace networks)
        super().__init__(
            state_dim,
            action_dim,
            lr,
            gamma,
            epsilon_start,
            epsilon_end,
            epsilon_decay,
            buffer_capacity,
            batch_size,
        )

        # Replace standard DQN networks with Dueling DQN networks
        self.q_network = DuelingDQN(state_dim, action_dim)
        self.target_network = DuelingDQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Recreate optimizer with new network
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.algorithm = "Dueling DQN"


class DuelingDoubleDQNAgent(DQNAgent):
    """
    Dueling Double DQN Agent

    Combines the best of both worlds:
    - Dueling architecture: Better state value estimation
    - Double DQN: Reduced Q-value overestimation

    This is expected to be the most stable and performant variant.

    References:
    - Wang et al. (2016) "Dueling Network Architectures for Deep RL"
    - van Hasselt et al. (2016) "Deep RL with Double Q-learning"
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64,
    ):
        # Initialize parent (but we'll replace networks)
        super().__init__(
            state_dim,
            action_dim,
            lr,
            gamma,
            epsilon_start,
            epsilon_end,
            epsilon_decay,
            buffer_capacity,
            batch_size,
        )

        # Replace standard DQN networks with Dueling DQN networks
        self.q_network = DuelingDQN(state_dim, action_dim)
        self.target_network = DuelingDQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Recreate optimizer with new network
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.algorithm = "Dueling Double DQN"

    def train_step(self):
        """Perform one training step using Dueling Double DQN algorithm"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Double DQN with Dueling architecture
        with torch.no_grad():
            # Select best actions using online network (Dueling)
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)

            # Evaluate selected actions using target network (Dueling)
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)

            # Compute target Q values
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


def record_episode_video(env, agent, episode_num, output_dir="trained_videos"):
    """Record a video of trained agent"""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = os.path.join(output_dir, f"trained_ep_{episode_num}_{timestamp}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    obs, info = env.reset()
    total_reward = 0
    steps = 0

    while True:
        action = agent.select_action(obs, training=False)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        # Try to get frame for video
        try:
            frame = env.render()
            if frame is not None and isinstance(frame, np.ndarray):
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if out is None:
                    height, width = frame_bgr.shape[:2]
                    out = cv2.VideoWriter(video_filename, fourcc, 50.0, (width, height))
                out.write(frame_bgr)
        except:
            pass

        if terminated or truncated:
            break

    if out is not None:
        out.release()

    return total_reward, steps, video_filename


def train_dqn(
    num_episodes=500,
    max_steps=1000,
    target_update_freq=10,
    save_freq=50,
    test_freq=100,
    algorithm="dqn",
    show_test_gui=False,
):
    """Train DQN or Double DQN agent on official Gymnasium LunarLander-v3"""
    print("="*60)
    print(f"Training {algorithm.upper()} on Official LunarLander-v3")
    print("="*60)

    # Create official environment (no rendering during training)
    env = gym.make("LunarLander-v3")

    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    algo = algorithm.lower()
    if algo in ["double_dqn", "ddqn"]:
        agent = DoubleDQNAgent(state_dim, action_dim)
        print("Using Double DQN algorithm")
    elif algo in ["dueling_dqn", "dueling", "duel"]:
        agent = DuelingDQNAgent(state_dim, action_dim)
        print("Using Dueling DQN algorithm")
    elif algo in ["dueling_double_dqn", "dueling_ddqn", "d3qn"]:
        agent = DuelingDoubleDQNAgent(state_dim, action_dim)
        print("Using Dueling Double DQN algorithm (D3QN)")
    else:
        agent = DQNAgent(state_dim, action_dim)
        print("Using vanilla DQN algorithm")

    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("trained_videos", exist_ok=True)

    # Training loop
    episode_rewards = []
    best_reward = -float('inf')

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        episode_reward = 0
        episode_loss = []

        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Store transition
            agent.replay_buffer.push(obs, action, reward, next_obs, terminated or truncated)

            # Train
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)

            episode_reward += reward
            obs = next_obs

            if terminated or truncated:
                break

        # Update target network
        if episode % target_update_freq == 0:
            agent.update_target_network()

        # Decay epsilon
        agent.decay_epsilon()

        # Record metrics
        episode_rewards.append(episode_reward)
        avg_loss = np.mean(episode_loss) if episode_loss else 0

        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode}/{num_episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg Reward (10): {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f}")

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save("models/best_model.pt")
            print(f"New best model saved! Reward: {best_reward:.2f}")

        # Save checkpoint
        if episode % save_freq == 0:
            agent.save(f"models/checkpoint_ep_{episode}.pt")

        # Test and record video
        if episode % test_freq == 0:
            print(f"\nTesting at episode {episode}...")
            test_render_mode = "human" if show_test_gui else "rgb_array"
            test_env = gym.make("LunarLander-v3", render_mode=test_render_mode)
            reward, steps, video_path = record_episode_video(test_env, agent, episode)
            test_env.close()
            print(f"Test - Reward: {reward:.2f}, Steps: {steps}")
            print(f"Video saved: {video_path}\n")

    env.close()

    # Final test
    print("\n" + "="*60)
    print("Training completed! Running final test...")
    print("="*60)

    agent.load("models/best_model.pt")
    # Final test always shows GUI to see the trained agent
    test_env = gym.make("LunarLander-v3", render_mode="human")

    for i in range(3):
        reward, steps, video_path = record_episode_video(test_env, agent, f"final_{i}")
        print(f"Final Test {i+1} - Reward: {reward:.2f}, Steps: {steps}")
        print(f"Video: {video_path}")

    test_env.close()

    print("\n" + "="*60)
    print(f"Best reward achieved: {best_reward:.2f}")
    print("All models saved in 'models/' directory")
    print("All videos saved in 'trained_videos/' directory")
    print("="*60)


if __name__ == "__main__":
    import sys

    # Default values
    num_episodes = 500
    algorithm = "ddqn"  # Default to Double DQN (better performance)
    show_test_gui = False

    # Parse command line arguments
    args = sys.argv[1:]
    for arg in args:
        arg_lower = arg.lower()
        if arg_lower in ["dqn", "double_dqn", "ddqn", "dueling_dqn", "dueling", "duel",
                         "dueling_double_dqn", "dueling_ddqn", "d3qn"]:
            algorithm = arg_lower
        elif arg_lower in ["--show-gui", "--gui"]:
            show_test_gui = True
        else:
            try:
                num_episodes = int(arg)
            except ValueError:
                print("Usage: python train.py [num_episodes] [algorithm] [--show-gui]")
                print("\nArguments:")
                print("  num_episodes : Number of episodes (default: 500)")
                print("  algorithm    : Algorithm to use (default: ddqn)")
                print("  --show-gui   : Show pygame window during test episodes")
                print("\nAvailable Algorithms:")
                print("  dqn              : Vanilla DQN")
                print("  ddqn, double_dqn : Double DQN (recommended)")
                print("  dueling, duel    : Dueling DQN")
                print("  d3qn, dueling_ddqn : Dueling Double DQN (best)")
                print("\nExamples:")
                print("  python train.py")
                print("  python train.py 1000 ddqn")
                print("  python train.py 1000 dueling")
                print("  python train.py 1000 d3qn")
                print("  python train.py ddqn --show-gui")
                sys.exit(1)

    print("\n" + "="*60)
    print("Configuration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Algorithm: {algorithm.upper()}")
    print(f"  Environment: Official Gymnasium LunarLander-v3")
    print(f"  Show test GUI: {'Yes' if show_test_gui else 'No (final test only)'}")
    print("="*60 + "\n")

    train_dqn(num_episodes=num_episodes, algorithm=algorithm, show_test_gui=show_test_gui)