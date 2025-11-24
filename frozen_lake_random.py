import gymnasium as gym
from tqdm import tqdm
import numpy as np

# 1. 환경 생성
# 'FrozenLake-v1' 환경을 생성합니다.
# is_slippery=True (기본 값) : 바닥이 미끄러워 의도한 방향으로 100% 이동하지 않을 수 있습니다 (확률적 환경)

env = gym.make("FrozenLake-v1", render_mode="ansi")
# env = gym.make("FrozenLake-v1", render_mode="human") # 시각화를 원하시면 'human'으로 변경

# 2. 하이퍼파라미터 설정
num_episodes = 1000 # 총 에피소드 수
num_timesteps = 50 # 에피소드당 최대 스텝 수 (Horizon) [cite: 3180]

# 3. 결과 추적을 위한 변수 초기화
total_reward = 0 # 총 보상 (성공 횟수와 동일, 성공 시 +1 보상) [cite: 3181, 2904]
total_timestep = 0   # 총 스텝 수 (각 에피소드가 끝날 때까지 걸린 시간의 합) [cite: 3182, 3190]
successful_episodes = 0 # 성공한 에피소드 횟수

# 4. 에피소드 실행 및 상호작용 시작
print(f"--- {num_episodes}개 에피소드 실행 (Random Policy) ---")

for i in tqdm(range(num_episodes)):
    # 환경을 초기화하고 초기 상태(s0)를 얻습니다. 
    state, info = env.reset()

# 에피소드당 보상 및 스텝 수 초기화
    episode_reward = 0
    
    # 최대 스텝 수까지 환경과 상호작용합니다.
    for t in range(num_timesteps):
        # 1) 행동 선택: 랜덤 정책 (Action Space에서 무작위로 선택) 
        random_action = env.action_space.sample()
        
        # 2) 행동 수행: 선택한 행동을 수행하고 다음 상태(s'), 보상(r), 종료 여부 등을 얻습니다. [cite: 3123, 3134]
        new_state, reward, terminated, truncated, info = env.step(random_action)
        
        # 보상 누적 [cite: 3187]
        episode_reward += reward
        
        # 3) 종료 조건 확인: 에피소드가 끝났는지 확인 (도착 또는 시간 초과) 
        if terminated or truncated:
            break
        
        # 4) 상태 업데이트: 다음 상태를 현재 상태로 설정
        state = new_state
    
    # 에피소드 종료 후 전체 통계 업데이트
    total_reward += episode_reward
    total_timestep += t
    if episode_reward > 0: # FrozenLake에서는 성공 시에만 보상 +1을 얻습니다. [cite: 2904]
        successful_episodes += 1

# 5. 결과 출력
print(f"\n총 성공한 에피소드 수: {successful_episodes} / {num_episodes}")
print("성공률: %.2f%%" % (successful_episodes / num_episodes * 100))
print("에피소드당 평균 스텝 수: %.2f" % (total_timestep / num_episodes))
# 에이전트가 목표에 도달하지 못하고 일찍 종료된 경우(예: 구멍에 빠짐), 스텝 수 t는 작게 기록됩니다.