# 라이브러리
import gymnasium as gym # Cart-Pole 환경을 불러오기 위한 강화 학습 환경 라이브러리
import collections # deque를 사용하여 효율적인 리플레이 버퍼를 구현
import random # 경험 리플레이를 위해 버퍼에서 경험을 무작위로 샘플링하고, epsilon-greedy를 위한 무작위 행동 선택에 사용

# 딥러닝 신경망 구축 및 훈련을 위해 사용
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 하이퍼파라미터
learning_rate = 0.0005 # 신경망 업데이트 속도 (학습 속도)
gamma = 0.98 # 감가율 (미래 보상의 중요도)
buffer_limit = 50000 # 리플레이 버퍼의 최대 크기
batch_size = 32 # 한 번의 훈련에 사용할 경험(transition)의 개수

class ReplayBuffer():
    def __init__(self): # collections.deque 를 사용하여 크기 제한(50000)이 있는 큐를 생성합니다. (크기가 차면 오래된 경험이 자동으로 삭제)
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition): # Agent가 환경에서 얻은 경험(s, a, r, s', done_mask)을 버퍼에 저장합니다.
        self.buffer.append(transition)

    def sample(self, n): # 버퍼에서 (batch_size=32)의 경험을 무작위로 샘플링하여 가져옵니다.
        mini_batch = random.sample(self.buffer, n) # 경험 리플레이의 핵심단계입니다. 저장된 경험들을 섞어서 훈련의 안정성을 높입니다.
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], [] # 무작위로 뽑은 경험들을 요소별로 분류하여 담을 준비를 합니다.

        for transition in mini_batch: # 미니-배치에 있는 모든 경험을 하나씩 처리합니다.
            s, a, r, s_prime, done_mask = transition # 하나의 경험이 (s, a ,r, s', done_mask) 다섯 가지 정보의 묶음으로 이루어져 있음을 보여줍니다.
            s_lst.append(s) # 현재 상태(s) 를 상태 리스트(s_lst)에 추가합니다.
            a_lst.append([a]) # 행동(a)을 리스트 안에 담아 행동 리스트(a_lst)에 추가합니다.
            r_lst.append([r]) # 보상(r)을 리스트 안에 담아 보상 리스트(r_lst)에 추가합니다.
            s_prime_lst.append(s_prime) # 다음상태(s' -s_prime)를 다음 상태 리스트(s_prime_lst)에 추가합니다.
            done_mask_lst.append([done_mask]) # 종료 마스크를 리스트 안에 담아 종료 마스크 리스트에 추가합니다.

            return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                   torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                   torch.tensor(done_mask_lst) 
            # 리스트 들을 torch.tensor로 변환하여 반환합니다. 이 텐서들은 신경망 훈련의 입력 데이터로 사용됩니다.
            # 보상 리스트와 다음 상태 리스트를 텐서로 변환하여 반환합니다.
            # 종료 마스크 리스트를 텐서로 변환하여 반환합니다.
    def size(self):
        return len(self.buffer) # 현재 버퍼에 저장된 경험의 개수를 반환합니다.

class Qnet(nn.Module): # Qnet 클래스는 기본 DQN 알고리즘에 사용되는 신경망(Q 함수 근사자)을 정의합니다.
    def __init__(self): 
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128) # 입력층. Cart-Pole의 4개 상태 (cart position, velocity, pole angle, angular velocity) 를 입력받습니다.
        self.fc2 = nn.Linear(128, 128) 
        self.fc3 = nn.Linear(128, 2) # 출력층. Cart-Pole의 2개 행동 (Left, Right)에 대한 Q 값 2개를 출력합니다.

    def forward(self, x): # 데이터가 신경망을 통과하는 순서입니다. 두개의 은닉층을 통과하여 ReLU 활성화 함수를 사용합니다.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon): # epsilon-greedy 정책을 구현합니다. 무작위 수가 e보다 작으면 무작위 행동을 반환, 그렇지 않으면 신경망이 출력한 Q값중 가장 큰 Q값에 해당하는 행동을 반환
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else:
            return out.argmax().item()

class DuelingQnet(nn.Module): # Dueling DQN 알고리즘에 사용되는 신경망을 정의합니다. 이 네트워크는 Q값을 가치(Value) 와 이점(Advantage)으로 분리하여 학습합니다.
    def __init__(self):
        super(DuelingQnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc_value = nn.Linear(128, 128)
        self.fc_adv = nn.Linear(128, 128)
        self.value = nn.Linear(128, 1)
        self.adv = nn.Linear(128, 2)

    def forward(self, x): # 신경망을 가치 스트림(v)과 이점 스트림(a)으로 나눕니다. 최종 값 (Q = v + (A-평균(A)) 공식으로 계산됩니다. 이는 Q = 가치 + (이점 - 평균이점) 형태로, 학습 안정성을 높이는 효과가 있습니다.
        x = F.relu(self.fc1(x))
        v = F.relu(self.fc_value(x))
        a = F.relu(self.fc_adv(x))
        v = self.value(v)
        a = self.adv(a)
        a_avg = torch.mean(a, dim=1, keepdim=True)  # Fixed dimension issue
        q = v + a - a_avg
        return q

    def sample_action(self, obs, epsilon): # epsilon-greedy 정책을 사용하여 행동을 선택합니다.
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else:
            return out.argmax().item()

def train_dqn(q, q_target, memory, optimizer): # Q 네트워크를 학습시키는 함수를 정의합니다.
    """Standard DQN training""" # 이 함수가 표준 DQN 훈련을 위한 것임을 나타내는 주석입니다.
    for i in range(10): # 리플레이 버퍼에서 샘플링한 미니-배치를 사용하여 10번 반복 훈련하는 루프를 시작합니다.
        s,a,r,s_prime,done_mask = memory.sample(batch_size) # 리플레이 버퍼에서 미니-배치 크기(batch_size = 32) 만큼의 경험(s, a, r, s', done_mask)을 무작위로 샘플링합니다.

        q_a = q_out.gather(1,a) # 출력된 Q값(q_out)중에서, 미니-배치에 기록된 실제 수행했던 행동에 해당하는 Q값만을 추출합니다.

        # DQN <- 이부분이 DQN의 핵심 계산임을 알리는 주석
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1) # 타겟 Q 네트워크를 사용하여 다음 상태에서의 모든 Q값 중 최대값을 계산

        target = r + gamma * max_q_prime * done_mask # 목표 값(y) 을 벨만 최적 방정식을 이용해 계산합니다. / 게임 종료시 max Q' 항을 0으로 만듭니다.
        loss = F.mse_loss(q_a, target) # 예측값 q_a 과 목표값 사이의 MSE (평균 제곱 오차)를 손실(loss)로 계산합니다.

        optimizer.zero_grad() # 이전에 계산된 **기울기(Gradient)**를 초기화합니다.
        loss.backward() # 계산된 손실(loss)을 기반으로 **메인 Q 네트워크(q)**의 파라미터들에 대한 기울기를 계산합니다.
        optimizer.step() # 계산된 기울기를 사용하여 경사 하강법에 따라 메인 Q 네트워크의 파라미터(θ)를 실제로 업데이트합니다.

def train_double_dqn(q, q_target, memory, optimizer): # train_double_dqn 함수를 정의하며, 훈련에 필요한 메인 Q 네트워크(q), 타겟 Q 네트워크(q_target), 리플레이 버퍼(memory), 최적화 도구(optimizer)를 인수로 받습니다.
    """Double DQN training""" # 이 함수가 Double DQN 훈련을 위한 것임을 나타내는 주석입니다.
    for i in range(10): # 리플레이 버퍼에서 샘플링한 미니-배치를 사용하여 10번 반복 훈련하는 루프를 시작합니다.
        s,a,r,s_prime,done_mask = memory.sample(batch_size) # 리플레이 버퍼에서 미니-배치 크기(batch_size)만큼의 경험(s, a, r, s’, done_mask)을 무작위로 샘플링하여 가져옵니다.

        q_out = q(s) # 현재 상태를 메인 Q 네트워크에 입력하여 모든 행동에 대한 Q 값을 출력합니다
        q_a = q_out.gather(1,a) # 출력된 Q 값(q_out) 중에서, 미니-배치에 기록된 실제 수행했던 행동 에 해당하는 Q 값만을 추출합니다.

        # Double DQN <- 이 부분이 Double DQN의 핵심 계산임을 알리는 주석입니다.
        argmax_Q = q(s_prime).max(1)[1].unsqueeze(1) # 메인 Q 네트워크(q)를 사용하여 다음 상태 에서 최적의 행동이 무엇인지 선택합니다
        max_q_prime = q_target(s_prime).gather(1, argmax_Q) # 타겟 Q 네트워크를 사용하여 앞서가 선택한 최적 행동의 Q 값을 계산합니다.

        target = r + gamma * max_q_prime * done_mask # 목표 값(y)을 벨만 방정식을 이용해 계산합니다
        loss = F.mse_loss(q_a, target) # 예측 값(q_a)과 목표 값(target) 사이의 MSE (평균 제곱 오차)**를 손실(loss)로 계산합니다.

        optimizer.zero_grad() # 이전에 계산된 **기울기(Gradient)**를 초기화합니다.
        loss.backward() # 계산된 손실(loss)을 기반으로 **메인 Q 네트워크(q)**의 파라미터들에 대한 기울기를 계산합니다.
        optimizer.step() # 계산된 기울기를 사용하여 경사 하강법에 따라 메인 Q 네트워크의 파라미터(θ)를 실제로 업데이트합니다.

def run_experiment(algorithm_type="DQN", render=False): # run_experiment 함수를 정의하며,기본값 DQN인 algorithm_type과 환경 시각화 여부를 결정하는 render 인수를 받습니다.
    """Run experiment with specified algorithm type""" # 이 함수가 지정된 알고리즘 타입으로 실험을 실행함을 알리는 주석입니다.
    print(f"\n=== Running {algorithm_type} Experiment ===") # 현재 어떤 알고리즘의 실험이 시작되었는지 터미널에 출력합니다.

    if render: # render 인수가 $\text{True}$이면 시각화 모드로 환경을 생성합니다.
        env = gym.make('CartPole-v1', render_mode='human') # Cart-Pole 환경을 GUI 창에 시각화하여 보여줄 수 있도록 만듭니다.
        print("Rendering enabled - GUI window will show CartPole visualization") # 시각화 모드가 활성화되었음을 사용자에게 알립니다.
    else: # render 인수가 $\text{False}$이면 시각화 없이 환경을 생성합니다.
        env = gym.make('CartPole-v1') # Cart-Pole 환경을 시각화 없이(빠른 속도로) 만듭니다.

    # Select network and training function based on algorithm type <- 알고리즘 타입에 따라 신경망 구조와 훈련 함수를 선택하는 주석입니다.
    if algorithm_type == "Dueling_DQN": # 알고리즘 타입이 **Dueling_DQN**인 경우입니다.
        q = DuelingQnet() # 메인 Q 네트워크는 DuelingQnet 구조를 사용합니다.
        q_target = DuelingQnet() # 타겟 Q 네트워크 역시 DuelingQnet 구조를 사용합니다.
        train_fn = train_double_dqn  # Dueling uses Double DQN training <- Dueling DQN은 안정성 향상을 위해 train_double_dqn 함수를 훈련 함수로 사용합니다.
    else: # 알고리즘 타입이 DQN 또는 Double_DQN인 경우입니다.
        q = Qnet() # 메인 Q 네트워크는 표준 Qnet 구조를 사용합니다.
        q_target = Qnet() # 타겟 Q 네트워크 역시 표준 Qnet 구조를 사용합니다.
        train_fn = train_dqn if algorithm_type == "DQN" else train_double_dqn # DQN 이면 train_dqn 을 Double_DQN이면 train_double_dqn을 훈련 함수로 선택

    q_target.load_state_dict(q.state_dict()) # 메인 네트워크(q)의 초기 가중치를 **타겟 네트워크(q_target)에 복사하여 초기화합니다.
    memory = ReplayBuffer() # 리플레이 버퍼 객체를 생성합니다.

    print_interval = 20 # 점수와 버퍼 크기 등을 출력할 에피소드 간격을 20으로 설정합니다.
    score = 0.0 # 누적 점수를 초기화합니다. 이 점수는 print_interval 간격으로 평균을 내기 위해 사용됩니다.
    optimizer = optim.Adam(q.parameters(), lr=learning_rate) # 메인 Q 네트워크(q)의 파라미터를 최적화할 Adam 옵티마이저를 설정하고, learning_rate를 적용합니다.

    for n_epi in range(3000): # 총 3,000개의 에피소드에 대해 학습 루프를 시작합니다.
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1% <- 에피소드 번호(n_epi)에 따라 epsilon 값을 0.08에서 0.01까지 선형적으로 감소시킵니다 (ϵ 감소**).
        s, _ = env.reset() # 환경을 초기 상태로 되돌리고, **초기 상태 s**를 받습니다.
        done = False # 에피소드 종료 플래그를 $\text{False}$로 초기화합니다.

        while not done: # 에피소드가 종료($\text{done}$이 True)될 때까지 환경과의 상호작용 루프를 반복합니다.
            a = q.sample_action(torch.from_numpy(s).float(), epsilon) # 메인 Q 네트워크를 사용하여 ϵ-greedy 정책에 따라 행동 $\mathbf{a}$를 선택합니다.
            s_prime, r, terminated, truncated, info = env.step(a) # 선택한 행동 $\mathbf{a}$를 환경에 적용하고, 다음 상태 s′ 보상 r, 종료 여부 $\mathbf{terminated}$와 truncated, 기타 정보를 받습니다.
            done = (terminated or truncated) # 환경의 종료 조건(terminated 또는 truncated)이 충족되었는지 확인하여 최종 종료 플래그 done을 결정합니다.
            done_mask = 0.0 if done else 1.0 # 종료 시(done=True) 0.0을, 종료되지 않았을 시(done=False) 1.0을 갖는 마스크를 만듭니다.
            memory.put((s,a,r/100.0,s_prime, done_mask)) # 얻은 경험(s, a, r, s’, done_mask)을 리플레이 버퍼에 저장합니다. (보상 $\mathbf{r}$을 100으로 나누어 정규화).
            s = s_prime # 현재 상태 $\mathbf{s}$를 다음 상태 $\mathbf{s'}$로 업데이트합니다.

            score += r # 이번 단계에서 받은 보상 $\mathbf{r}$을 누적 점수에 합산합니다.
            if done: # 에피소드가 종료된 경우입니다.
                break # 상호작용 루프(while not done)를 종료하고 다음 에피소드로 넘어갑니다.

        if memory.size()>2000: # 리플레이 버퍼의 경험 개수가 2,000개를 초과했을 때만 훈련을 시작합니다.
            train_fn(q, q_target, memory, optimizer) # 선택된 훈련 함수(train_dqn 또는 train_double_dqn)를 호출하여 신경망을 업데이트합니다.

        if n_epi%print_interval==0 and n_epi!=0: # 에피소드 번호가 20 배수이고 첫 에피소드(0)가 아닐 때 다음을 수행합니다.
            q_target.load_state_dict(q.state_dict()) # 메인 네트워크(q)의 최신 파라미터를 타겟 네트워크(q_target)에 복사하여 업데이트합니다.
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100)) # 현재 에피소드 번호, 평균 점수, 버퍼 크기, ϵ 값을 포맷에 맞춰 출력합니다.
            score = 0.0 # 출력 후 평균 점수 계산을 위해 누적 점수를 다시 0으로 초기화합니다.

    env.close() # 모든 학습이 끝난 후 환경을 닫습니다.
    return q, q_target # 최종적으로 학습된 메인 Q 네트워크(q)와 타겟 Q 네트워크(q_target)를 반환합니다.

def main(): # main 함수를 정의하며, 프로그램의 주요 실행 로직을 담고 있습니다.
    """Run experiments for all three algorithms""" # 세 가지 알고리즘에 대한 실험을 실행하는 함수임을 알리는 주석입니다.
    algorithms = ["DQN", "Double_DQN", "Dueling_DQN"] # 사용 가능한 세 가지 알고리즘 이름을 담는 리스트를 정의합니다.

    print("Choose algorithm to run:") # 사용자에게 실행할 알고리즘을 선택하라는 안내 메시지를 출력합니다.
    print("1. DQN") # 옵션 1번 (DQN)을 출력합니다.
    print("2. Double DQN") # 옵션 2번 (Double DQN)을 출력합니다.
    print("3. Dueling DQN") # 옵션 3번 (Dueling DQN)을 출력합니다.
    print("4. Run all algorithms") # 옵션 4번 (모두 실행)을 출력합니다.

    choice = input("Enter your choice (1-4): ") # 사용자로부터 1에서 4 사이의 숫자를 입력받아 choice 변수에 저장합니다.

    # Ask about rendering # 환경 시각화(GUI) 여부를 묻는 주석입니다.
    render_choice = input("Enable GUI visualization? (y/n): ").lower() # 사용자에게 GUI 시각화 활성화 여부를 묻고, 입력값을 소문자로 변환하여 저장합니다.
    render = render_choice in ['y', 'yes'] # render_choice가 'y' 또는 'yes'이면 render 변수를 $\text{True}$로 설정하고, 아니면 $\text{False}$로 설정합니다.

    if choice == "1": # 사용자의 선택이 **"1"**인 경우입니다.
        run_experiment("DQN", render) # DQN 알고리즘으로 실험을 실행합니다.
    elif choice == "2": # 사용자의 선택이 **"2"**인 경우입니다.
        run_experiment("Double_DQN", render) # Double DQN 알고리즘으로 실험을 실행합니다.
    elif choice == "3": # 사용자의 선택이 **"3"**인 경우입니다.
        run_experiment("Dueling_DQN", render) # Dueling DQN 알고리즘으로 실험을 실행합니다.
    elif choice == "4": # 사용자의 선택이 **"4"**인 경우입니다.
        for alg in algorithms: # 정의된 모든 알고리즘 리스트(algorithms)를 순회하는 루프를 시작합니다.
            run_experiment(alg, render) # 루프를 돌며 각 알고리즘으로 실험을 실행합니다.
    else: # 사용자가 유효하지 않은 선택을 한 경우입니다.
        print("Invalid choice, running DQN by default") # 잘못된 선택을 했으므로 기본값으로 $\text{DQN}$을 실행함을 알립니다.
        run_experiment("DQN", render) # 기본값인 DQN 알고리즘으로 실험을 실행합니다.

if __name__ == '__main__': # 현재 파일이 직접 실행될 때만 아래 코드를 실행하도록 하는 표준 파이썬 구문입니다.
    main() # main 함수를 호출하여 프로그램 실행을 시작합니다.