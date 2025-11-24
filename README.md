# 실행명령어
$ python dqn_lander_py.py

$ python dueling_dqn_lander_v1.py

# 실행시간
dqn_lander 보다 dueling_dqn_lander의 실행시간이 2배 발생
유의미한 결과값은 발생하지 못함, 에피소드 길이를 길게 하였으나 정상적으로 되지 않음

# 실행결과
$ python3 plot_results.py --dqn_results dqn_results_v1.npz --dueling_results dueling_dqn_results_v1.npz

# 실행결과 이미지
comparison_plot.png
