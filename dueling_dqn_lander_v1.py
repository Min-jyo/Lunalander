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
# --- 추가: Target Network 업데이트 주기 조정 ---
target_update_interval = 100 

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

# --- 3. Dueling Q-네트워크 클래스 (변경 없음) ---
class DuelingQnet(nn.Module):
    def __init__(self, obs_space_n, action_space_n):
        super(DuelingQnet, self).__init__()
        # 공통 특징 추출 레이어
        self.feature_layer = nn.Sequential(
            nn.Linear(obs_space_n, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # 상태 가치(V) 스트림
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # 행동 우위(A) 스트림
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_space_n)
        )

    def forward(self, x):
        # 특징 추출
        features = self.feature_layer(x)
        
        # 각 스트림의 출력 계산
        v = self.value_stream(features)
        adv = self.advantage_stream(features)
        
        # Q-값 결합: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_vals = v + (adv - adv.mean(dim=-1, keepdim=True))
        return q_vals

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, out.shape[-1] - 1)
        else:
            return out.argmax().item()

# --- 4. 훈련 함수 (DQN과 동일) ---
def train_dueling_dqn(q, q_target, memory, optimizer):
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

# --- 5. 메인 실행 함수 (DuelingQnet 사용 및 결과 파일명 변경) ---
def main():
    env_name = "LunarLander-v3"
    env = gym.make(env_name, render_mode="none")
    obs_space_n = env.observation_space.shape[0]
    action_space_n = env.action_space.n
    
    # --- 수정: DuelingQnet 인스턴스 생성 ---
    q = DuelingQnet(obs_space_n, action_space_n)
    q_target = DuelingQnet(obs_space_n, action_space_n)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    
    episode_returns = []
    episode_lengths = []
    episode_successes = []
    
    epsilon_start = 0.08
    epsilon_min = 0.01
    # --- 수정: 엡실론 감쇠 속도 10000으로 조정 ---
    # epsilon_decay_rate = 5000 # 기존
    epsilon_decay_rate = 10000 
    
    print(f"--- Dueling DQN Training for {env_name} Started ---")

    for n_epi in tqdm(range(num_episodes)):
        epsilon = max(epsilon_min, epsilon_start - (epsilon_start - epsilon_min) * (n_epi / epsilon_decay_rate)) 
        
        s, info = env.reset()
        done = False
        
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
        
        episode_returns.append(current_episode_return)
        episode_lengths.append(current_episode_length)
        episode_successes.append(1 if current_episode_return >= 200 else 0)

        if memory.size() > 2000:
            train_dueling_dqn(q, q_target, memory, optimizer)
        
        # --- 수정: Target Network 업데이트 로직을 target_update_interval로 분리 ---
        if n_epi % target_update_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())

        # 로그 출력은 print_interval에 맞춰 진행
        if n_epi % print_interval == 0 and n_epi != 0:
            avg_return = np.mean(episode_returns[-print_interval:])
            print(f"n_episode: {n_epi}, avg_return: {avg_return:.1f}, n_buffer: {memory.size()}, eps: {epsilon*100:.1f}%")

    env.close()

    # --- 추가: 학습 종료 후 SageMaker 출력 디렉터리에 결과 파일로 저장 ---
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", ".")
    np.savez(os.path.join(output_dir, 'dueling_dqn_results_v1.npz'),
             returns=np.array(episode_returns),
             lengths=np.array(episode_lengths),
             successes=np.array(episode_successes))
    print(f"\n--- Dueling DQN Training Finished. Results saved to {output_dir} ---")


if __name__ == '__main__':
    main()