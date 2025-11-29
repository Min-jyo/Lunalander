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
import pandas as pd

# --- Global Parameters (Used as defaults/overridden) ---
# NOTE: These are defaults, actual values are set in run_test from TEST_CONFIGS
learning_rate = 0.001
gamma = 0.98
buffer_limit = 50000        
batch_size = 32
tau = 1e-3                  
epsilon = 1.0               
PRINT_INTERVAL = 20         
TRAINING_EPISODES = 1000  # Full training episodes for graph resolution
EVAL_EPISODES = 20        # Evaluation episodes for final table metrics


# --- Classes and Core Functions ---

class ReplayBuffer():
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
        if x.dim() == 1: x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon: return random.randint(0,3)
        else: return out.argmax(dim=1).item() 

class DuelingQnet(nn.Module):
    def __init__(self):
        super(DuelingQnet, self).__init__()
        self.fc1 = nn.Linear(8, 128)
        self.fc_value = nn.Linear(128, 128)
        self.fc_adv = nn.Linear(128, 128)
        self.value = nn.Linear(128, 1)
        self.adv = nn.Linear(128, 4)
    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        v = F.relu(self.fc_value(x))
        a = F.relu(self.fc_adv(x))
        v = self.value(v)
        a = self.adv(a)
        a_avg = torch.mean(a, dim=1, keepdim=True)
        q = v + a - a_avg
        return q
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon: return random.randint(0,3)
        else: return out.argmax(dim=1).item()

# 📌 train_dqn/train_double_dqn 함수가 하이퍼파라미터를 인수로 받지 않고, 
# 전역 변수를 사용하므로, run_test에서 전역 변수를 먼저 설정해야 합니다.
def train_dqn(q, q_target, memory, optimizer):
    # Uses global batch_size, gamma, tau
    s,a,r,s_prime,done_mask = memory.sample(batch_size)
    q_out = q(s)
    q_a = q_out.gather(1,a)
    max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
    target = r + gamma * max_q_prime * done_mask
    loss = F.mse_loss(q_a, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    for target_param, local_param in zip(q_target.parameters(), q.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

def train_double_dqn(q, q_target, memory, optimizer):
    # Uses global batch_size, gamma, tau
    s,a,r,s_prime,done_mask = memory.sample(batch_size)
    q_out = q(s)
    q_a = q_out.gather(1,a)
    argmax_Q = q(s_prime).max(1)[1].unsqueeze(1)
    max_q_prime = q_target(s_prime).gather(1, argmax_Q)
    target = r + gamma * max_q_prime * done_mask
    loss = F.mse_loss(q_a, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    for target_param, local_param in zip(q_target.parameters(), q.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

def evaluate_model(q_net, num_episodes):
    env = gym.make('LunarLander-v3')
    total_score, success_count, total_length = 0.0, 0, 0.0
    for _ in range(num_episodes):
        s, _ = env.reset()
        done, episode_score, episode_length = False, 0.0, 0 
        with torch.no_grad():
            while not done:
                a = q_net.sample_action(torch.from_numpy(s).float(), 0.0)
                s_prime, r, terminated, truncated, info = env.step(a)
                done = (terminated or truncated)
                s = s_prime; episode_score += r; episode_length += 1
                if done: break
        total_score += episode_score
        if episode_score >= 200: success_count += 1
        total_length += episode_length
    env.close()
    return total_score / num_episodes, (success_count / num_episodes) * 100, total_length / num_episodes

# 💡 run_test 함수 수정: LR, Gamma를 인수로 받아 optimizer 생성에 사용
def run_test(alg_type, train_fn, n_episodes, lr_val):
    global epsilon
    env = gym.make('LunarLander-v3')
    
    # Network Selection
    if alg_type == "Dueling_DQN":
        q, q_target = DuelingQnet(), DuelingQnet()
    else:
        q, q_target = Qnet(), Qnet()

    q_target.load_state_dict(q.state_dict())
    epsilon = 1.0 
    memory = ReplayBuffer()
    
    # 📌 현재 테스트의 LR 값으로 Optimizer 생성
    optimizer = optim.Adam(q.parameters(), lr=lr_val) 
    
    score, score_history = 0.0, []
    
    # Gamma, batch_size는 전역에서 설정된 현재 테스트 값을 사용합니다.
    print(f"  [Training Start] LR: {lr_val}, Gamma: {gamma}, Batch: {batch_size}, Decay: 0.995")
    
    for n_epi in range(n_episodes): 
        s, _ = env.reset()
        done = False
        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, terminated, truncated, info = env.step(a)
            done = (terminated or truncated)
            memory.put((s,a,r,s_prime, 0.0 if done else 1.0))
            s = s_prime
            score += r
            if memory.size()>2000:
                # train_fn 호출. 내부적으로 global gamma, batch_size 사용
                train_fn(q, q_target, memory, optimizer) 
            if done: break
            
        epsilon = max(0.01, epsilon * 0.995)
        
        if n_epi % PRINT_INTERVAL == 0 and n_epi != 0: 
            avg_score = score / PRINT_INTERVAL
            print(f"    Epi: {n_epi:<4} / {n_episodes} | Avg Score: {avg_score:.2f} | Buffer: {memory.size():<5} | Epsilon: {epsilon*100:.1f}%")
            score_history.append((n_epi, avg_score))
            score = 0.0

    env.close()
    avg_return, success_rate, avg_length = evaluate_model(q, EVAL_EPISODES)
    print(f"  [Evaluation End] Avg Return: {avg_return:.2f}, Success Rate: {success_rate:.2f}%")
    return avg_return, success_rate, avg_length, score_history

# 💡 Plotting 함수: 모든 6개 라인을 한 그래프에 그립니다.
def plot_results(all_history_data, all_history_labels):
    """학습 곡선(평균 리턴값 vs 에피소드)을 시각화합니다."""
    
    # Define color/style pairs for clear separation of V1 vs V2
    styles = [
        ('darkviolet', 'DQN V1', '--'), ('darkgreen', 'Dueling V1', '-'), ('red', 'Double V1', '-.'),
        ('dodgerblue', 'DQN V2', '--'), ('lime', 'Dueling V2', '-'), ('orange', 'Double V2', '-.'),
    ]
    
    plt.figure(figsize=(10, 6))
    
    for history, label in zip(all_history_data, all_history_labels):
        # 📌 수정된 부분: s 대신 (color, l, ls) 튜플 전체를 반환하도록 수정
        style_tuple = next((color, l, ls) for color, l, ls in styles if l == label or (label.startswith(l.split()[0]) and label.endswith(l.split()[-1])))
        color, linestyle = style_tuple[0], style_tuple[2]

        episodes = [item[0] for item in history]
        scores = [item[1] for item in history]
        
        plt.plot(episodes, scores, label=label, color=color, linestyle=linestyle, linewidth=2 if linestyle == '-' else 1)

    plt.title('DQN Algorithms Performance Comparison (Average Return per Episode)')
    plt.xlabel(f'Episode (Average of {PRINT_INTERVAL} episodes)') 
    plt.ylabel('Average Return (Score)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', ncol=2, fontsize='small') # 범례를 2열로 표시
    
    plt.axhline(y=200, color='r', linestyle='--', linewidth=1, label='Success Threshold (200)')
    
    plot_filename = 'learning_curve_comparison.png'
    plt.savefig(plot_filename)
    print(f"\n✅ Learning curve saved to {plot_filename}.")
    try:
        plt.show() 
    except Exception:
        pass 


# --- Main Execution Loop ---
TEST_CONFIGS = [
    # V1 Configuration
    {'version': 'V1', 'alg': 'DQN', 'fn': train_dqn, 'params': {'lr': 0.005, 'gamma': 0.98, 'batch_size': 32}},
    {'version': 'V1', 'alg': 'Dueling DQN', 'fn': train_double_dqn, 'params': {'lr': 0.005, 'gamma': 0.98, 'batch_size': 32}}, 
    {'version': 'V1', 'alg': 'Double DQN', 'fn': train_double_dqn, 'params': {'lr': 0.005, 'gamma': 0.98, 'batch_size': 32}},
    
    # V2 Configuration
    {'version': 'V2', 'alg': 'DQN', 'fn': train_dqn, 'params': {'lr': 0.001, 'gamma': 0.99, 'batch_size': 64}},
    {'version': 'V2', 'alg': 'Dueling DQN', 'fn': train_double_dqn, 'params': {'lr': 0.001, 'gamma': 0.99, 'batch_size': 64}},
    {'version': 'V2', 'alg': 'Double DQN', 'fn': train_double_dqn, 'params': {'lr': 0.001, 'gamma': 0.99, 'batch_size': 64}},
]

results = []
all_history_data = []
all_history_labels = []

print(f"==============================================================")
print(f"🔥 Starting 6 Total Experiments (Training: {TRAINING_EPISODES}, Evaluation: {EVAL_EPISODES})")
print(f"==============================================================")

for i, config in enumerate(TEST_CONFIGS):
    # 📌 global 선언 제거: 스크립트 레벨에서 바로 변경합니다.
    learning_rate = config['params']['lr']
    gamma = config['params']['gamma']
    batch_size = config['params']['batch_size']
    
    alg_name = f"{config['alg']} {config['version']}"
    
    print(f"\n--- Running Test {i+1}/6: {alg_name} ---")
    
    # Run training and collect history (LR 값만 run_test에 인수로 전달)
    avg_return, success_rate, avg_length, score_history = run_test(
        config['alg'].replace(' ', '_'), config['fn'], TRAINING_EPISODES, learning_rate
    )
    
    # Collect results for table
    results.append({
        'Algorithm': alg_name,
        '평균리턴': f"{avg_return:.2f}",
        '성공률': f"{success_rate:.2f}%",
        '에피소드길이': f"{avg_length:.1f}",
        '수렴속도': '라인 그래프 참조',
        '학습안정성': '라인 그래프 참조',
        '데이터효율성': '고정'
    })
    
    # Collect history for graph
    all_history_data.append(score_history)
    all_history_labels.append(alg_name)

# --- Final Result Output (Table & Graph) ---
print(f"\n==============================================================")
print(f"✅ All Experiments Completed (Evaluation Episodes: {EVAL_EPISODES} each)")
print(f"==============================================================")

df = pd.DataFrame(results)
print(df.to_string(index=False))

# Generate and display the final comparison graph
plot_results(all_history_data, all_history_labels)