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
learning_rate = 0.001
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

# 💡 run_experiment 함수에 lr과 decay_rate 인자 추가
def run_experiment(algorithm_type="DQN", render=False, load_model=False, lr=learning_rate, decay_rate=0.995):
    """지정된 알고리즘과 하이퍼파라미터로 실험을 실행하고 학습된 모델을 반환합니다."""
    global epsilon
    
    # 튜닝 중에는 학습된 모델을 저장하거나 불러오지 않음 (단일 실험 모드에서만 파일 저장/로드)
    is_tuning = (lr != learning_rate or decay_rate != 0.995) 

    if is_tuning:
        # 튜닝 모드에서 실행할 경우, 어떤 알고리즘을 튜닝하는지 명시 (옵션 4의 첫 단계)
        print(f"  [Tuning] {algorithm_type} - LR: {lr:.1e}, Decay: {decay_rate:.4f}...")
    else:
        print(f"\n=== Running {algorithm_type} Experiment ===")

    if render:
        env = gym.make('LunarLander-v3', render_mode='human')
        # 튜닝 모드에서는 렌더링 메시지 생략
        if not is_tuning:
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
    
    # 📌 모델 불러오기 (Load Model): 튜닝 모드에서는 불러오지 않음
    if load_model and os.path.exists(model_path) and not is_tuning:
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
    # 💡 전달받은 lr로 Optimizer 초기화
    optimizer = optim.Adam(q.parameters(), lr=lr) 
    
    # 💡 학습 곡선 저장을 위한 리스트 추가
    score_history = []
    
    # 💡 에피소드 횟수는 1000으로 유지
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
        
        # 💡 전달받은 decay_rate로 Epsilon decay 적용
        epsilon = max(0.01, epsilon * decay_rate) 

        if n_epi % PRINT_INTERVAL == 0 and n_epi != 0: 
            avg_score = score / PRINT_INTERVAL
            if not is_tuning: # 튜닝 중에는 출력 생략
                print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, avg_score, memory.size(), epsilon*100))
            score_history.append((n_epi, avg_score))
            score = 0.0

    env.close()
    
    # 📌 모델 저장 (Save Model): 튜닝 모드에서는 저장하지 않음
    if not is_tuning:
        torch.save(q.state_dict(), model_path)
        print(f"\nModel for {algorithm_type} saved to {model_path}")
    
    return q, q_target, score_history 

def evaluate_model(q_net, env_name='LunarLander-v3', num_episodes=100, render=False):
    """
    최종 학습된 모델을 사용하여 성능(평균 리턴값, 성공률, 평균 길이)을 평가합니다.
    성공 기준은 스코어 200점 이상입니다.
    """
    # 튜닝 모드에서는 평가 결과를 명시적으로 출력하지 않음
    print_summary = True if num_episodes == 100 else False
    
    if print_summary:
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

    if print_summary:
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

# 💡 하이퍼파라미터 튜닝 함수 이름 변경 및 최적 파라미터 반환하도록 수정
def find_optimal_hyperparameters(algorithm_type="Dueling_DQN"):
    """
    주요 하이퍼파라미터(Learning Rate, Epsilon Decay Rate)를 그리드 탐색하고 최적 파라미터를 반환합니다.
    """
    print(f"\n=======================================================")
    print(f"🎯 Starting Hyperparameter Tuning for {algorithm_type}")
    print(f"=======================================================")

    # 💡 탐색 범위 확장: Learning Rate는 1e-5부터 1e-2까지 7개
    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    # 💡 탐색 범위 확장: Decay Rate는 0.990부터 0.999까지 5개
    decay_rates = [0.990, 0.993, 0.995, 0.997, 0.999]

    best_score = -np.inf
    best_params = None
    
    results = []
    
    total_combinations = len(learning_rates) * len(decay_rates)
    print(f"Total combinations to test: {total_combinations} combinations.")

    # 그리드 탐색 시작
    for lr in learning_rates:
        for decay in decay_rates:
            # 모델 학습 (run_experiment): 튜닝 모드는 is_tuning=True로 자동 설정되어 출력 생략
            q_net, _, _ = run_experiment(algorithm_type, False, False, lr=lr, decay_rate=decay)
            
            # 모델 평가 (evaluate_model)
            avg_return, success_rate, avg_length = evaluate_model(q_net, num_episodes=50)

            results.append({
                'lr': lr,
                'decay': decay,
                'avg_return': avg_return,
                'success_rate': success_rate
            })

            if avg_return > best_score:
                best_score = avg_return
                best_params = {'lr': lr, 'decay': decay}

    print("\n=======================================================")
    print("📈 Tuning Results Summary:")
    print(f"{'LR':<8} {'Decay Rate':<12} {'Avg Return':<12} {'Success Rate'}")
    print("-" * 45)
    for res in results:
        print(f"{res['lr']:.1e:<8} {res['decay']:.4f:<12} {res['avg_return']:.2f:<12} {res['success_rate']:.2f}%")
        
    print("\n🥇 Best Hyperparameters Found:")
    print(f"   Learning Rate: {best_params['lr']:.1e}")
    print(f"   Decay Rate: {best_params['decay']:.4f}")
    print(f"   Best Avg Return: {best_score:.2f}")
    print("=======================================================")
    
    # 💡 최적 파라미터 반환
    return best_params


def main():
    """Run experiments for all three algorithms"""
    algorithms = ["DQN", "Double_DQN", "Dueling_DQN"]

    print("Choose algorithm to run:")
    print("1. DQN")
    print("2. Double DQN")
    print("3. Dueling DQN")
    print("4. Run all algorithms (Comparison Plot)")
    print("5. Hyperparameter Tuning (Find Optimal LR/Decay)") # 💡 튜닝 옵션 추가

    choice = input("Enter your choice (1-5): ")

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
        # 1. Optimal parameter tuning (Dueling_DQN을 기준으로 최적 파라미터 찾기)
        print("\n--- Step 1: Finding Optimal Hyperparameters for Comparison (using Dueling_DQN) ---")
        best_params = find_optimal_hyperparameters("Dueling_DQN") 
        
        optimal_lr = best_params['lr']
        optimal_decay = best_params['decay']
        
        print(f"\n--- Optimal Parameters Selected for Comparison: LR={optimal_lr:.1e}, Decay={optimal_decay:.4f} ---")
        print("--- Step 2: Running All Algorithms with Optimal Parameters ---")
        
        for alg in algorithms:
            # 2. Run experiment with optimal parameters
            # 튜닝된 LR과 Decay Rate를 run_experiment에 전달
            q_net, _, history = run_experiment(alg, render, load_model, lr=optimal_lr, decay_rate=optimal_decay)
            all_history.append(history)
            evaluate_model(q_net, render=False) 
            
        # 3. Plot results
        plot_results(all_history, algorithms) 
        return
    elif choice == "5": # 💡 튜닝 모드 실행
        # Dueling DQN에 대한 튜닝 실행 
        find_optimal_hyperparameters("Dueling_DQN")
        return
    else:
        print("Invalid choice, running DQN by default")
        q_net_to_evaluate, _ , _ = run_experiment("DQN", render, load_model)

    # 📌 단일 알고리즘 선택 시, 학습이 끝난 후 평가 실행
    if q_net_to_evaluate is not None:
        evaluate_model(q_net_to_evaluate, render=False) 

if __name__ == '__main__':
    main()