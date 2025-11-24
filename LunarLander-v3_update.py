import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import collections
import random
from tqdm import tqdm
import numpy as np

# --- 1. 하이퍼파라미터 설정 ---
learning_rate = 0.0005
gamma = 0.99            # 할인 계수 [cite: 4677]
buffer_limit = 50000    # 리플레이 버퍼 크기 [cite: 4679]
batch_size = 32         # 미니 배치 크기 [cite: 4678]
print_interval = 20     # 결과 출력 및 타겟 네트워크 업데이트 주기

# --- 2. 리플레이 버퍼 클래스 [cite: 4454] ---
class ReplayBuffer:
    def __init__(self):
        # double-ended queue (양쪽 끝에서 데이터 삽입/삭제 가능한 큐) 사용 [cite: 4713]
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        # 경험 (s, a, r, s', done_mask) 저장 [cite: 4453]
        self.buffer.append(transition)

    def sample(self, n):
        # 버퍼에서 무작위로 n개의 샘플(미니 배치) 추출 [cite: 4712]
        mini_batch = random.sample(self.buffer, n)
        
        # 각 요소를 리스트로 분리
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            # done_mask는 다음 상태가 종료 상태인 경우 (Q-값 0)를 처리하기 위함
            done_mask_lst.append([done_mask])
        
        # PyTorch Tensor로 변환하여 반환
        return torch.tensor(np.array(s_lst), dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(np.array(s_prime_lst), dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)

# --- 3. Q-네트워크 클래스 (심층 신경망) ---
class Qnet(nn.Module):
    def __init__(self, obs_space_n, action_space_n):
        super(Qnet, self).__init__()
        # LunarLander: obs_space_n = 8, action_space_n = 4
        self.fc1 = nn.Linear(obs_space_n, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_space_n) # 최종 출력: 각 행동에 대한 Q-값 [cite: 4417]

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        # 입실론-탐욕 정책 [cite: 4621]
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, out.shape[-1] - 1) # 탐험(Exploration): 무작위 행동 [cite: 4621]
        else:
            return out.argmax().item() # 활용(Exploitation): 최대 Q-값을 가진 행동 선택 [cite: 4621]

# --- 4. DQN 훈련 함수 ---
def train_dqn(q, q_target, memory, optimizer):
    """표준 DQN 훈련 로직"""
    # Cart-Pole 예시에서는 10번 반복하지만, LunarLander에서는 1번만 해도 무방
    for i in range(1): 
        # 미니 배치 샘플링
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        
        # 1. 예측 Q-값 (Q_theta(s, a)) 계산
        q_out = q(s)
        q_a = q_out.gather(1, a) # 실제로 선택한 행동 a에 대한 Q-값만 추출
        
        # 2. 목표 Q-값 (y) 계산: r + gamma * max_a' Q_theta'(s', a') [cite: 4557, 4574]
        # 타겟 네트워크 q_target 사용 [cite: 4604]
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        
        # done_mask 처리: s'가 종료 상태이면 max_q_prime 부분을 0으로 만듦 [cite: 4577, 4779]
        target = r + gamma * max_q_prime * done_mask 
        
        # 3. 손실 (MSE) 계산 및 역전파
        loss = F.mse_loss(q_a, target.float())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# --- 5. 메인 실행 함수 ---
def main():
    # 5-1. 환경 초기화 및 준비
    env_name = "LunarLander-v3"
    env = gym.make(env_name, render_mode="none")
    obs_space_n = env.observation_space.shape[0] # 상태 공간 크기 (8)
    action_space_n = env.action_space.n          # 행동 공간 크기 (4)
    
    q = Qnet(obs_space_n, action_space_n)
    q_target = Qnet(obs_space_n, action_space_n)
    q_target.load_state_dict(q.state_dict()) # 타겟 네트워크 초기화 [cite: 4825]
    memory = ReplayBuffer()
    
    optimizer = optim.Adam(q.parameters(), lr=learning_rate) # [cite: 4841]
    
    # 5-2. 학습 루프 변수 설정
    num_episodes = 50000 
    score = 0.0
    
    # epsilon 선형 감소 (8%에서 시작하여 1%까지 감소) [cite: 4856]
    epsilon_start = 0.08
    epsilon_min = 0.01
    epsilon_decay_rate = 5000 
    
    print(f"--- DQN Training for {env_name} Started ---")

    # 5-3. 메인 학습 루프
    for n_epi in tqdm(range(num_episodes)):
        # Epsilon 계산
        epsilon = max(epsilon_min, epsilon_start - (epsilon_start - epsilon_min) * (n_epi / epsilon_decay_rate)) 
        
        # 에피소드 시작
        s, info = env.reset()
        done = False
        
        while not done:
            # 1. 행동 선택 (Epsilon-Greedy Policy)
            # 상태 's'를 Tensor로 변환
            s_tensor = torch.from_numpy(s).float()
            action = q.sample_action(s_tensor, epsilon)
            
            # 2. 환경 상호작용 (Step)
            s_prime, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 3. 경험 저장 (Done Mask 처리)
            # LunarLander는 보상이 크므로 정규화하지 않아도 되지만, CartPole 예시와 동일하게 0/1로 처리함
            done_mask = 0.0 if terminated else 1.0 
            
            # 다음 상태 s_prime은 numpy 배열로 저장
            memory.put((s, action, r, s_prime, done_mask))
            
            s = s_prime
            score += r
            
            if done:
                break
        
        # 4. 학습 수행 및 타겟 네트워크 업데이트
        # 리플레이 버퍼에 충분한 데이터가 쌓이면 학습 시작 [cite: 4895]
        if memory.size() > 2000:
            train_dqn(q, q_target, memory, optimizer)
        
        # 5. 결과 출력 및 타겟 네트워크 동기화
        if n_epi % print_interval == 0 and n_epi != 0:
            # 타겟 네트워크 동기화: q의 가중치를 q_target에 복사 [cite: 4608, 4899]
            q_target.load_state_dict(q.state_dict())
            print("n_episode : {}, score: {:.1f}, n_buffer: {}, eps: {:.1f}%".format(
                  n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0

    # 5-4. 종료 처리
    env.close()

if __name__ == '__main__':
    # 학습 시간이 오래 걸릴 수 있으므로, 실행 시 유의해 주세요.
    main()