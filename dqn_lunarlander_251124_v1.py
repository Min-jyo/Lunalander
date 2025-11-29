# libraries
import gymnasium as gym
import collections
import random
import numpy as np

# pytorch library is used for deep learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# hyperparameters
learning_rate = 0.001
gamma = 0.98
buffer_limit = 50000        # size of replay buffer
batch_size = 32
tau = 1e-3                  # for soft update of target network

class ReplayBuffer():
    """
    경험 리플레이 버퍼를 구현하는 클래스. 
    이중 종료 큐(deque)를 사용하여 경험을 저장하고 무작위로 샘플링합니다.
    """
    def __init__(self):
        # 최대 크기(buffer_limit)를 가진 이중 종료 큐 초기화
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        # 새로운 경험(transition: s, a, r, s', done_mask)을 버퍼에 추가
        self.buffer.append(transition)

    def sample(self, n):
        # 버퍼에서 n개(batch_size)의 경험을 무작위로 샘플링
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        # PyTorch 텐서로 변환 (float 형식을 사용)
        return torch.tensor(np.array(s_lst), dtype=torch.float), \
               torch.tensor(a_lst), \
               torch.tensor(r_lst, dtype=torch.float), \
               torch.tensor(np.array(s_prime_lst), dtype=torch.float), \
               torch.tensor(done_mask_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    """표준 DQN에 사용되는 Q-네트워크 구조"""
    def __init__(self):
        super(Qnet, self).__init__()
        # LunarLander 상태 8개 입력, 행동 4개 출력
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        # 엡실론-탐욕 정책(epsilon-greedy policy)에 따라 행동 선택
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,3) # 탐험 (Exploration)
        else:
            return out.argmax().item() # 활용 (Exploitation)

class DuelingQnet(nn.Module):
    """Dueling DQN에 사용되는 Q-네트워크 구조 (가치와 이점 분리)"""
    def __init__(self):
        super(DuelingQnet, self).__init__()
        # 상태 입력 (8)
        self.fc1 = nn.Linear(8, 128)
        
        # 가치(Value) 스트림
        self.fc_value = nn.Linear(128, 128)
        self.value = nn.Linear(128, 1) # 상태 가치 V(s) 출력 (1개)

        # 이점(Advantage) 스트림
        self.fc_adv = nn.Linear(128, 128)
        self.adv = nn.Linear(128, 4) # 행동 이점 A(s, a) 출력 (4개)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        
        # 가치 스트림 계산
        v = F.relu(self.fc_value(x))
        v = self.value(v)
        
        # 이점 스트림 계산
        a = F.relu(self.fc_adv(x))
        a = self.adv(a)
        
        # 최종 Q 계산: Q = V + (A - mean(A))
        # 평균 이점(A_avg)을 빼주어 가치와 이점 추정의 분리성을 확보
        a_avg = torch.mean(a, dim=1, keepdim=True)
        q = v + a - a_avg
        return q

    def sample_action(self, obs, epsilon):
        # 엡실론-탐욕 정책에 따라 행동 선택
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,3)
        else:
            return out.argmax().item()

def train_dqn(q, q_target, memory, optimizer, tau):
    """
    Standard DQN training with Soft Update.
    타겟 값 계산: Q_target(s')에서 max를 취함.
    """
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1,a)

        # DQN Target Calculation
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)

        target = r + gamma * max_q_prime * done_mask
        loss = F.mse_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Soft update of target network parameters
        for target_param, local_param in zip(q_target.parameters(), q.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

def train_double_dqn(q, q_target, memory, optimizer, tau):
    """
    Double DQN training with Soft Update.
    타겟 값 계산: Q_main(s')에서 max 행동 선택, Q_target(s')로 값 평가.
    """
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1,a)

        # Double DQN Target Calculation
        # 1. 메인 Q(q)로 최적 행동(argmax_Q)을 선택
        argmax_Q = q(s_prime).max(1)[1].unsqueeze(1)
        # 2. 타겟 Q(q_target)로 선택된 행동의 Q 값(max_q_prime)을 평가
        max_q_prime = q_target(s_prime).gather(1, argmax_Q)

        target = r + gamma * max_q_prime * done_mask
        loss = F.mse_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Soft update of target network parameters
        for target_param, local_param in zip(q_target.parameters(), q.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

def run_experiment(algorithm_type="DQN", render=False):
    """
    지정된 알고리즘 타입으로 LunarLander 환경에서 강화 학습을 실행합니다.
    (Dueling DQN은 Double DQN 훈련 함수를 사용합니다.)
    """
    print(f"\n=== Running {algorithm_type} Experiment ===")

    # LunarLander-v3 환경 생성
    if render:
        env = gym.make('LunarLander-v3', render_mode='human')
        print("Rendering enabled - GUI window will show LunarLander visualization")
    else:
        env = gym.make('LunarLander-v3')

    # 알고리즘 타입에 따라 신경망과 훈련 함수 선택
    if algorithm_type == "Dueling_DQN":
        q = DuelingQnet()
        q_target = DuelingQnet()
        train_fn = train_double_dqn
    else:
        q = Qnet()
        q_target = Qnet()
        train_fn = train_dqn if algorithm_type == "DQN" else train_double_dqn

    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    
    epsilon = 1.0 # 초기 엡실론 설정

    # 총 10,000 에피소드 진행
    for n_epi in range(10000):
        s, _ = env.reset()
        done = False

        while not done:
            # Epsilon-greedy를 통한 행동 선택
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, terminated, truncated, info = env.step(a)
            done = (terminated or truncated)
            done_mask = 0.0 if done else 1.0
            
            # [수정 #1] 보상 클리핑 적용: r을 [-1.0, 1.0]으로 제한하여 학습 안정성 확보
            r_clipped = np.clip(r, -1.0, 1.0)
            memory.put((s,a,r_clipped,s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break

        # 버퍼에 충분한 경험이 쌓이면 훈련 시작
        if memory.size()>2000:
            train_fn(q, q_target, memory, optimizer, tau)

        if n_epi%print_interval==0 and n_epi!=0:
            # [수정 #2] Soft Update 사용으로 인해 Hard Copy 코드 제거
            # q_target.load_state_dict(q.state_dict()) # 이 코드를 제거함

            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
        
        # [수정 #3] Epsilon decay factor를 0.998로 조정 (탐험 속도 증가)
        epsilon = max(0.01, epsilon * 0.998) 

    env.close()
    return q, q_target

def main():
    """Run experiments for all three algorithms"""
    algorithms = ["DQN", "Double_DQN", "Dueling_DQN"]

    print("Choose algorithm to run:")
    print("1. DQN")
    print("2. Double DQN")
    print("3. Dueling DQN")
    print("4. Run all algorithms")

    choice = input("Enter your choice (1-4): ")

    # Ask about rendering
    render_choice = input("Enable GUI visualization? (y/n): ").lower()
    render = render_choice in ['y', 'yes']

    if choice == "1":
        run_experiment("DQN", render)
    elif choice == "2":
        run_experiment("Double_DQN", render)
    elif choice == "3":
        run_experiment("Dueling_DQN", render)
    elif choice == "4":
        for alg in algorithms:
            run_experiment(alg, render)
    else:
        print("Invalid choice, running DQN by default")
        run_experiment("DQN", render)

if __name__ == '__main__':
    main()