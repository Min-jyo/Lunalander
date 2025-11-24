import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import collections
import random
from tqdm import tqdm
import numpy as np
import os

# --- 1. 하이퍼파라미터 설정 ---
learning_rate = 0.0005
gamma = 0.99
buffer_limit = 50000
batch_size = 32
# --- 수정: 에피소드 수 10000으로 조정 ---
num_episodes = 10000 
print_interval = 20

# --- 2. 리플레이 버퍼 클래스 (변경 없음) ---
class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
        
        return torch.tensor(np.array(s_lst), dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(np.array(s_prime_lst), dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)

# --- 3. Q-네트워크 클래스 (변경 없음) ---
class Qnet(nn.Module):
    def __init__(self, obs_space_n, action_space_n):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(obs_space_n, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_space_n)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, out.shape[-1] - 1)
        else:
            return out.argmax().item()

# --- 4. DQN 훈련 함수 (변경 없음) ---
def train_dqn(q, q_target, memory, optimizer):
    for i in range(1): 
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        
        q_out = q(s)
        q_a = q_out.gather(1, a)
        
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask 
        
        loss = F.mse_loss(q_a, target.float())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# --- 5. 메인 실행 함수 (지표 수집 기능 추가) ---
def main():
    env_name = "LunarLander-v3"
    env = gym.make(env_name, render_mode="none")
    obs_space_n = env.observation_space.shape[0]
    action_space_n = env.action_space.n
    
    q = Qnet(obs_space_n, action_space_n)
    q_target = Qnet(obs_space_n, action_space_n)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    
    # --- 추가: 결과 저장을 위한 리스트 ---
    episode_returns = []
    episode_lengths = []
    episode_successes = []
    
    epsilon_start = 0.08
    epsilon_min = 0.01
    epsilon_decay_rate = 5000 
    
    print(f"--- DQN Training for {env_name} Started ---")

    for n_epi in tqdm(range(num_episodes)):
        epsilon = max(epsilon_min, epsilon_start - (epsilon_start - epsilon_min) * (n_epi / epsilon_decay_rate)) 
        
        s, info = env.reset()
        done = False
        
        # --- 추가: 에피소드별 지표 초기화 ---
        current_episode_return = 0.0
        current_episode_length = 0
        
        while not done:
            s_tensor = torch.from_numpy(s).float()
            action = q.sample_action(s_tensor, epsilon)
            
            s_prime, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            done_mask = 0.0 if terminated else 1.0 
            
            memory.put((s, action, r, s_prime, done_mask))
            
            s = s_prime
            current_episode_return += r
            current_episode_length += 1
            
            if done:
                break
        
        # --- 추가: 에피소드 종료 후 지표 저장 ---
        episode_returns.append(current_episode_return)
        episode_lengths.append(current_episode_length)
        # 성공 기준: 점수 200점 이상
        episode_successes.append(1 if current_episode_return >= 200 else 0)

        if memory.size() > 2000:
            train_dqn(q, q_target, memory, optimizer)
        
        # --- 수정: Target Network 업데이트 로직 (기존과 동일하게 print_interval에 맞춰 업데이트) ---
        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            
            # --- 수정: 출력 형식 변경 ---
            avg_return = np.mean(episode_returns[-print_interval:])
            print(f"n_episode: {n_epi}, avg_return: {avg_return:.1f}, n_buffer: {memory.size()}, eps: {epsilon*100:.1f}%")

    env.close()

    # --- 추가: 학습 종료 후 SageMaker 출력 디렉터리에 결과 파일로 저장 ---
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", ".")
    np.savez(os.path.join(output_dir, 'dqn_results_v1.npz'),
             returns=np.array(episode_returns),
             lengths=np.array(episode_lengths),
             successes=np.array(episode_successes))
    print(f"\n--- DQN Training Finished. Results saved to {output_dir} ---")


if __name__ == '__main__':
    main()