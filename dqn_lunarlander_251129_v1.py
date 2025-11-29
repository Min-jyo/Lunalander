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

# 💡 파일 시스템 관리를 위해 os 라이브러리 추가
import os 
# 💡 그래프 출력을 위해 matplotlib 라이브러리 추가
import matplotlib.pyplot as plt

# hyperparameters
learning_rate = 0.005
gamma = 0.98
buffer_limit = 50000        # size of replay buffer
batch_size = 32
tau = 1e-3                  # for soft update
epsilon = 1.0               # Epsilon을 전역 변수로 관리
PRINT_INTERVAL = 20         # 💡 평균 점수 계산 및 출력 간격 (Global로 정의)

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)    # double-ended queue

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
               torch.tensor(r_lst, dtype=torch.float), torch.tensor(np.array(s_prime_lst), dtype=torch.float), \
               torch.tensor(done_mask_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x):
        # 🐛 오류 해결: 단일 샘플일 경우 배치 차원(dim=0)이 없으므로,
        # 배치 입력이 아닌 경우에만 unsqueeze(0)를 적용하도록 처리
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        # 🐛 오류 해결: forward 함수 내부에서 unsqueeze를 처리하도록 변경했으므로 여기서는 obs만 전달
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,3)
        else:
            # 단일 입력 시 out은 [1, 4] 크기이므로 dim=1에서 argmax를 취합니다.
            return out.argmax(dim=1).item() 

class DuelingQnet(nn.Module):
    def __init__(self):
        super(DuelingQnet, self).__init__()
        self.fc1 = nn.Linear(8, 128)
        self.fc_value = nn.Linear(128, 128)
        self.fc_adv = nn.Linear(128, 128)
        self.value = nn.Linear(128, 1)
        self.adv = nn.Linear(128, 4)

    def forward(self, x):
        # 🐛 오류 해결: 단일 샘플일 경우 배치 차원(dim=0)이 없으므로,
        # 배치 입력이 아닌 경우에만 unsqueeze(0)를 적용하도록 처리
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        x = F.relu(self.fc1(x))
        v = F.relu(self.fc_value(x))
        a = F.relu(self.fc_adv(x))
        v = self.value(v)
        a = self.adv(a)
        
        # 🐛 오류 해결: 이제 x가 최소 [1, 8] 형태이므로 a는 [배치크기, 4] 형태를 가짐
        # 따라서 dim=1에서 평균을 취하는 것이 가능해집니다.
        a_avg = torch.mean(a, dim=1, keepdim=True)
        q = v + a - a_avg
        return q

    def sample_action(self, obs, epsilon):
        # 🐛 오류 해결: forward 함수 내부에서 unsqueeze를 처리하도록 변경했으므로 여기서는 obs만 전달
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,3)
        else:
            # 단일 입력 시 out은 [1, 4] 크기이므로 dim=1에서 argmax를 취합니다.
            return out.argmax(dim=1).item()

def train_dqn(q, q_target, memory, optimizer, tau):
    """Standard DQN training"""
    s,a,r,s_prime,done_mask = memory.sample(batch_size)

    q_out = q(s)
    q_a = q_out.gather(1,a)

    # DQN
    max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)

    target = r + gamma * max_q_prime * done_mask
    loss = F.mse_loss(q_a, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Soft update
    for target_param, local_param in zip(q_target.parameters(), q.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

def train_double_dqn(q, q_target, memory, optimizer, tau):
    """Double DQN training"""
    s,a,r,s_prime,done_mask = memory.sample(batch_size)

    q_out = q(s)
    q_a = q_out.gather(1,a)

    # Double DQN
    argmax_Q = q(s_prime).max(1)[1].unsqueeze(1)
    max_q_prime = q_target(s_prime).gather(1, argmax_Q)

    target = r + gamma * max_q_prime * done_mask
    loss = F.mse_loss(q_a, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Soft update
    for target_param, local_param in zip(q_target.parameters(), q.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

# 💡 run_experiment 함수를 원래의 인자 구성으로 되돌림 (lr, decay_rate 인자 제거)
def run_experiment(algorithm_type="DQN", render=False, load_model=False):
    """지정된 알고리즘으로 실험을 실행하고 학습된 모델을 반환합니다."""
    global epsilon
    
    # 📌 고정된 전역 변수 하이퍼파라미터 사용을 명시
    print(f"\n=== Running {algorithm_type} Experiment (LR={learning_rate}, Decay=0.995) ===")

    if render:
        env = gym.make('LunarLander-v3', render_mode='human')
        print("Rendering enabled - GUI window will show LunarLander visualization")
    else:
        env = gym.make('LunarLander-v3')

    # 네트워크 선택
    if algorithm_type == "Dueling_DQN":
        q = DuelingQnet()
        q_target = DuelingQnet()
        train_fn = train_double_dqn
    else:
        q = Qnet()
        q_target = Qnet()
        train_fn = train_dqn if algorithm_type == "DQN" else train_double_dqn

    model_path = f'./{algorithm_type}_q_net.pth'
    
    # 📌 모델 불러오기 (Load Model)
    if load_model and os.path.exists(model_path):
        print(f"Loading previous model from {model_path} to continue training.")
        try:
            q.load_state_dict(torch.load(model_path))
            q_target.load_state_dict(q.state_dict())
            epsilon = 0.1 
        except Exception as e:
            print(f"Error loading model state: {e}. Starting new training.")
            q_target.load_state_dict(q.state_dict())
            epsilon = 1.0
    else:
        q_target.load_state_dict(q.state_dict())
        epsilon = 1.0 # 새로 학습 시작 시 epsilon 초기화

    memory = ReplayBuffer()

    score = 0.0
    # 💡 고정된 전역 변수 learning_rate 사용
    optimizer = optim.Adam(q.parameters(), lr=learning_rate) 
    
    score_history = []
    
    for n_epi in range(1000): 
        s, _ = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, terminated, truncated, info = env.step(a)
            done = (terminated or truncated)
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r,s_prime, done_mask))
            s = s_prime

            score += r

            if memory.size()>2000:
                train_fn(q, q_target, memory, optimizer, tau)

            if done:
                break
        
        # 💡 고정된 decay rate 0.999 사용
        epsilon = max(0.01, epsilon * 0.995) 

        if n_epi % PRINT_INTERVAL == 0 and n_epi != 0: 
            avg_score = score / PRINT_INTERVAL
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, avg_score, memory.size(), epsilon*100))
            score_history.append((n_epi, avg_score))
            score = 0.0

    env.close()
    
    # 📌 모델 저장 (Save Model)
    torch.save(q.state_dict(), model_path)
    print(f"\nModel for {algorithm_type} saved to {model_path}")
    
    return q, q_target, score_history 

def evaluate_model(q_net, env_name='LunarLander-v3', num_episodes=100, render=False):
    """
    최종 학습된 모델을 사용하여 성능(평균 리턴값, 성공률, 평균 길이)을 평가합니다.
    성공 기준은 스코어 200점 이상입니다.
    """
    # 📌 튜닝 모드 관련 출력 제거 (단일 실험 모드)
    print(f"\n=== Evaluating Model over {num_episodes} episodes ===")
    
    if render:
        env = gym.make(env_name, render_mode='human')
    else:
        env = gym.make(env_name)
    
    total_score = 0.0
    success_count = 0  # 성공 횟수 카운터
    total_length = 0.0 # 총 에피소드 길이 누적 변수
    
    epsilon_eval = 0.0 # 평가 시 탐험 없음 (최적 행동만)

    for n_epi in range(num_episodes):
        s, _ = env.reset()
        done = False
        episode_score = 0.0
        episode_length = 0 

        with torch.no_grad(): # 평가 시에는 그래디언트 계산 불필요
            while not done:
                a = q_net.sample_action(torch.from_numpy(s).float(), epsilon_eval)
                s_prime, r, terminated, truncated, info = env.step(a)
                done = (terminated or truncated)
                s = s_prime
                episode_score += r
                episode_length += 1
                
                if done:
                    break
        
        total_score += episode_score
        
        if episode_score >= 200:
            success_count += 1
            
        total_length += episode_length

    env.close()
    
    avg_return = total_score / num_episodes
    success_rate = (success_count / num_episodes) * 100
    avg_length = total_length / num_episodes

    # 📌 튜닝 모드 관련 출력 제거 (단일 실험 모드)
    print(f"\n[Evaluation Results - {num_episodes} Episodes]")
    print(f"✅ Average Return: {avg_return:.2f}")
    print(f"🎉 Success Rate (Score >= 200): {success_rate:.2f}%")
    print(f"⏱️ Average Episode Length: {avg_length:.1f} steps")
    
    return avg_return, success_rate, avg_length

def plot_results(all_history, algorithms):
    """학습 곡선(평균 리턴값 vs 에피소드)을 시각화합니다."""
    print("\n=== Generating Learning Curve Plot ===")
    
    plt.figure(figsize=(10, 6))
    
    for history, alg_name in zip(all_history, algorithms):
        episodes = [item[0] for item in history]
        scores = [item[1] for item in history]
        
        # 각 알고리즘별로 다른 색상으로 라인 그래프를 그립니다.
        plt.plot(episodes, scores, label=alg_name)

    plt.title('DQN Algorithms Performance Comparison (Average Return)')
    plt.xlabel(f'Episode (Average of {PRINT_INTERVAL} episodes)') 
    plt.ylabel('Average Return (Score)')
    plt.grid(True, linestyle='--')
    plt.legend(loc='lower right')
    
    # 성능 비교 기준선 추가 (수렴 속도 및 안정성 비교 기준)
    plt.axhline(y=200, color='r', linestyle='-', linewidth=1, label='Success Threshold (200)')
    
    plot_filename = 'learning_curve_comparison.png'
    plt.savefig(plot_filename)
    print(f"✅ Learning curve saved to {plot_filename}. Please check the output directory.")
    try:
        plt.show() 
    except Exception:
        pass 

# 📌 하이퍼파라미터 튜닝 함수 (find_optimal_hyperparameters) 제거

def main():
    """Run experiments for all three algorithms"""
    algorithms = ["DQN", "Double_DQN", "Dueling_DQN"]

    print("Choose algorithm to run:")
    print("1. DQN")
    print("2. Double DQN")
    print("3. Dueling DQN")
    print("4. Run all algorithms")
    # 📌 5. Hyperparameter Tuning 옵션 제거

    choice = input("Enter your choice (1-4): ") # 📌 사용자 입력 범위 변경

    # Ask about rendering
    render_choice = input("Enable GUI visualization during training? (y/n): ").lower()
    render = render_choice in ['y', 'yes']

    # 📌 모델 불러오기 여부 묻기
    load_choice = input("Load previously saved model to continue training? (y/n): ").lower()
    load_model = load_choice in ['y', 'yes']

    q_net_to_evaluate = None
    all_history = [] 

    if choice == "1":
        q_net_to_evaluate, _, _ = run_experiment("DQN", render, load_model) 
    elif choice == "2":
        q_net_to_evaluate, _, _ = run_experiment("Double_DQN", render, load_model) 
    elif choice == "3":
        q_net_to_evaluate, _, _ = run_experiment("Dueling_DQN", render, load_model) 
    elif choice == "4":
        # 📌 튜닝 로직 제거 및 고정 파라미터로 순차 실행
        print("\n--- Running All Algorithms with Fixed Parameters ---")
        
        for alg in algorithms:
            q_net, _, history = run_experiment(alg, render, load_model)
            all_history.append(history)
            evaluate_model(q_net, render=False) 
            
        plot_results(all_history, algorithms) 
        return
    # 📌 5번 튜닝 옵션 관련 로직 제거
    else:
        print("Invalid choice, running DQN by default")
        q_net_to_evaluate, _ , _ = run_experiment("DQN", render, load_model)

    # 📌 단일 알고리즘 선택 시, 학습이 끝난 후 평가 실행
    if q_net_to_evaluate is not None:
        evaluate_model(q_net_to_evaluate, render=False) 

if __name__ == '__main__':
    main()