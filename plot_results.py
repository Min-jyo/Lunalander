import numpy as np
import matplotlib.pyplot as plt
import argparse

def moving_average(data, window_size):
    """이동 평균을 계산합니다."""
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def plot_results(dqn_data, dueling_dqn_data, window_size=100):
    """DQN과 Dueling DQN의 학습 결과를 시각화하고 비교 분석합니다."""
    
    # --- 1. 데이터 로드 ---
    dqn_returns = dqn_data['returns']
    dqn_lengths = dqn_data['lengths']
    dqn_successes = dqn_data['successes']
    
    dueling_returns = dueling_dqn_data['returns']
    dueling_lengths = dueling_dqn_data['lengths']
    dueling_successes = dueling_dqn_data['successes']
    
    num_episodes = len(dqn_returns)
    episodes = np.arange(num_episodes)

    # --- 2. 이동 평균 계산 ---
    dqn_returns_smooth = moving_average(dqn_returns, window_size)
    dueling_returns_smooth = moving_average(dueling_returns, window_size)
    
    dqn_lengths_smooth = moving_average(dqn_lengths, window_size)
    dueling_lengths_smooth = moving_average(dueling_lengths, window_size)
    
    # 성공률은 이동 평균으로 계산
    dqn_success_rate = moving_average(dqn_successes, window_size) * 100
    dueling_success_rate = moving_average(dueling_successes, window_size) * 100
    
    # 이동 평균으로 인해 줄어든 길이만큼 x축 조정
    x_axis = np.arange(window_size - 1, num_episodes)

    # --- 3. 그래프 시각화 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    
    # 평균 리턴 그래프
    axs[0].plot(x_axis, dqn_returns_smooth, label='DQN')
    axs[0].plot(x_axis, dueling_returns_smooth, label='Dueling DQN')
    axs[0].set_ylabel('Average Return')
    axs[0].set_title(f'Average Return per Episode (Smoothed over {window_size} episodes)')
    axs[0].legend()
    # 수렴 목표선
    axs[0].axhline(y=200, color='r', linestyle='--', label='Success Threshold (200)')

    # 성공률 그래프
    axs[1].plot(x_axis, dqn_success_rate, label='DQN')
    axs[1].plot(x_axis, dueling_success_rate, label='Dueling DQN')
    axs[1].set_ylabel('Success Rate (%)')
    axs[1].set_title(f'Success Rate (Smoothed over {window_size} episodes)')
    axs[1].legend()

    # 에피소드 길이 그래프
    axs[2].plot(x_axis, dqn_lengths_smooth, label='DQN')
    axs[2].plot(x_axis, dueling_lengths_smooth, label='Dueling DQN')
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Average Episode Length')
    axs[2].set_title(f'Average Episode Length (Smoothed over {window_size} episodes)')
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig('comparison_plot.png')
    print("\nComparison plot saved as 'comparison_plot.png'")
    # plt.show() # 로컬 실행 시 주석 해제하여 바로 확인

    # --- 4. 최종 성능 지표 및 학습 효율성 분석 ---
    print("\n--- Performance Comparison ---")
    
    # 분석 구간 (예: 마지막 100개 에피소드)
    last_n = 100
    
    # DQN 최종 성능
    dqn_final_avg_return = np.mean(dqn_returns[-last_n:])
    dqn_final_success_rate = np.mean(dqn_successes[-last_n:]) * 100
    dqn_final_avg_length = np.mean(dqn_lengths[-last_n:])
    
    # Dueling DQN 최종 성능
    dueling_final_avg_return = np.mean(dueling_returns[-last_n:])
    dueling_final_success_rate = np.mean(dueling_successes[-last_n:]) * 100
    dueling_final_avg_length = np.mean(dueling_lengths[-last_n:])

    print(f"\n[Final Performance (Last {last_n} Episodes)]")
    print(f"                       |   DQN   | Dueling DQN")
    print(f"-----------------------|---------|-------------")
    print(f"Average Return         | {dqn_final_avg_return:7.2f} | {dueling_final_avg_return:11.2f}")
    print(f"Success Rate (%)       | {dqn_final_success_rate:7.2f} | {dueling_final_success_rate:11.2f}")
    print(f"Average Episode Length | {dqn_final_avg_length:7.2f} | {dueling_final_avg_length:11.2f}")

    # 수렴 속도 (처음으로 이동 평균 점수가 200을 넘는 지점)
    try:
        dqn_convergence_episode = np.where(dqn_returns_smooth >= 200)[0][0] + window_size
    except IndexError:
        dqn_convergence_episode = "Did not converge"

    try:
        dueling_convergence_episode = np.where(dueling_returns_smooth >= 200)[0][0] + window_size
    except IndexError:
        dueling_convergence_episode = "Did not converge"

    print(f"\n[Learning Efficiency & Stability]")
    print(f"                                   |     DQN     | Dueling DQN")
    print(f"-----------------------------------|-------------|-------------")
    print(f"Convergence Speed (Episodes to 200) | {str(dqn_convergence_episode):^11} | {str(dueling_convergence_episode):^11}")

    # 학습 안정성 (수렴 이후 점수의 표준 편차)
    if isinstance(dqn_convergence_episode, int):
        dqn_stability = np.std(dqn_returns[dqn_convergence_episode:])
    else:
        dqn_stability = "N/A"
        
    if isinstance(dueling_convergence_episode, int):
        dueling_stability = np.std(dueling_returns[dueling_convergence_episode:])
    else:
        dueling_stability = "N/A"

    print(f"Learning Stability (Stddev post-conv)| {str(round(dqn_stability, 2) if isinstance(dqn_stability, (int, float)) else 'N/A'):^11} | {str(round(dueling_stability, 2) if isinstance(dueling_stability, (int, float)) else 'N/A'):^11}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dqn_results', type=str, default='dqn_results.npz', help='Path to the DQN results file (.npz)')
    parser.add_argument('--dueling_results', type=str, default='dueling_dqn_results.npz', help='Path to the Dueling DQN results file (.npz)')
    args = parser.parse_args()

    try:
        # 저장된 결과 파일 로드
        dqn_results = np.load(args.dqn_results)
        dueling_dqn_results = np.load(args.dueling_results)
        
        plot_results(dqn_results, dueling_dqn_results)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find results file: {e.filename}")
        print("Please ensure the file paths are correct and the training jobs have been completed.")

