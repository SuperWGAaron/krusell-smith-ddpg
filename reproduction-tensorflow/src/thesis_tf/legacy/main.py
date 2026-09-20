import os
import sys
import functools
import numpy as np
import time
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from ddpg import *
from parameters_ddpg import *


print('Using TensorFlow {}'.format(tf.__version__))
print('Using NumPy {}'.format(np.__version__))

# allow labels in Chinese characters
chinese_label = os.environ.get('THESIS_TF_ENGLISH', '1') != '1'
if chinese_label:
    from matplotlib import rcParams
    rcParams['font.family'] = ['Heiti TC']
file_suffix = 'eng' if not chinese_label else 'chn'

n_pop = int(os.environ.get('THESIS_TF_N_POP', '100'))
n_steps = int(os.environ.get('THESIS_TF_N_STEPS', '30'))
n_life, n_round = int(os.environ.get('THESIS_TF_N_LIFE', '5000')), int(os.environ.get('THESIS_TF_N_ROUND', '5'))
n_life_0 = int(os.environ.get('THESIS_TF_N_LIFE_0', '10000'))
simu_pop = int(os.environ.get('THESIS_TF_SIMU_POP', '10000'))
simu_round = int(os.environ.get('THESIS_TF_SIMU_ROUND', '100'))
K_0 = float(os.environ.get('THESIS_TF_K_0', '40'))
B = None # np.load(os.getcwd()+'/test/B.npy')
np.random.seed(int(os.environ.get('THESIS_TF_SEED', '0')))
np.set_printoptions(precision=6)

idx = int(os.environ.get('THESIS_TF_IDX', '30'))
foldername = 'round_{}_life_{}_step_{}_pop_{}_idx_{}'.format(n_round, n_life, n_steps, n_pop, idx)
filedir = os.path.join(os.getcwd(), 'test', foldername, '')
os.makedirs(filedir+'ckpt/', exist_ok=True)
# stdout_file = os.path.join(filedir, 'stdout.txt')
# sys.stdout = open(stdout_file, 'w')
# print = functools.partial(print, flush=True)

test = int(os.environ.get('THESIS_TF_TRAIN', '0'))
mfg = MFG(ksp=ksp, n_steps=n_steps, K_0=K_0, B=B, net_pm=net_pm, filedir=filedir)
if test:
    mfg.solve_policy(n_life=n_life, n_round=n_round, simu_pop=simu_pop, simu_round=simu_round, n_life_0=n_life_0, print_ctr=1)
else:
    mfg.agent.actor.load_checkpoint(idx=n_round, path=os.path.join(filedir, 'ckpt', 'Actor'))
    mfg.round_ctr = n_round

score_history = np.load(filedir + 'score_history.npy', allow_pickle=True)
reg_stat = np.load(filedir + 'reg_stat.npz', allow_pickle=True)

# Figure: Updates in ALM coefficients
B_list = reg_stat['B']
B_round_list = reg_stat['B_round']
fig, ax = plt.subplots()
if not chinese_label:
    legend = ['cons_b', 'slope_b', 'cons_g', 'slope_g']
    xlabel = 'round'
    ylabel = 'coefficient value'
else:
    legend = ['截距(坏时期)', '斜率(坏时期)', '截距(好时期)', '斜率(好时期)']
    xlabel = '训练轮数'
    ylabel = '系数值'
ax_obj = ax.plot(B_list)
ax.legend(iter(ax_obj), legend)
ax.set(xlabel=xlabel,
    ylabel=ylabel,
    # title='ALM coefficients',
    )
fig.savefig(filedir+'ALM_coeff_'+file_suffix+'.jpg', dpi=200)

# Figure: moving average of scores
window = 1000
n_game = len(score_history)
running_avg = np.empty(n_game)
for i in range(n_game):
    running_avg[i] = np.mean(score_history[max(0, i+1-window):i+1])

fig, ax = plt.subplots()
if not chinese_label:
    xlabel = 'game'
    ylabel = 'average score'
else:
    xlabel = '游戏'
    ylabel = '得分(移动平均)'
ax.plot(running_avg)
ax.set(xlabel=xlabel,
    ylabel=ylabel,
    # title='moving average of scores',
    )
fig.savefig(filedir+'score_history_'+file_suffix+'.jpg', dpi=200)

# Figure: one simulation of aggregate capital path
simulate_data = True
if simulate_data:
    zi_path, _ = mfg.env.generate_shock()
    K_path_ALM = mfg.simulate_ALM(zi_path=zi_path, B=B_list[-1])
    paths = mfg.simulate_dist(zi_path=zi_path, n_pop=10000, random=True)
    K_path_sml = paths['K']
    K_path_plot = np.savez(filedir+'Equilibrium_K_path.npz',
                        K_path_ALM=np.array(K_path_ALM),
                        K_path_sml=np.array(K_path_sml))

K_path_plot = np.load(filedir+'Equilibrium_K_path.npz')
K_path_ALM = K_path_plot['K_path_ALM']
K_path_sml = K_path_plot['K_path_sml']
fig, ax = plt.subplots()
if not chinese_label:
    legend_1 = 'approximation by ALM'
    legend_2 = 'simulation'
    xlabel = 'period'
    ylabel = 'aggregate capital'
else:
    legend_1 = '总资本运动律近似'
    legend_2 = '蒙特卡洛模拟'
    xlabel = '时期/步数'
    ylabel = '总资本'
ax.plot(K_path_ALM, 'b', linestyle='--', label=legend_1)
ax.plot(K_path_sml, 'r', linestyle='-', label=legend_2)
ax.set(xlabel=xlabel,
    ylabel=ylabel,
    # title='Equilibrium capital path: simulation vs. approximation',
    )
ax.legend()
fig.savefig(filedir+'Equilibrium_K_path_'+file_suffix+'.jpg', dpi=200)

# Figure: Equilibrium ALM
idx = n_round
B0, B1, B2, B3 = B_list[idx]
k_vec = np.arange(1e-3, 80, 1)
kp_vec_b = np.zeros_like(k_vec)
kp_vec_g = np.zeros_like(k_vec)
for i in range(len(k_vec)):
    kp_vec_b[i] = np.exp(B0 + B1 * np.log(k_vec[i]))
    kp_vec_g[i] = np.exp(B2 + B3 * np.log(k_vec[i]))

fig, ax = plt.subplots()
if not chinese_label:
    legend_1 = 'bad'
    legend_2 = 'good'
    legend_3 = '45 degree'
    xlabel = 'aggregate capital today'
    ylabel = 'aggregate capital tomorrow'
else:
    legend_1 = '坏时期'
    legend_2 = '好时期'
    legend_3 = '45度'
    xlabel = '当期总资本'
    ylabel = '下期总资本'
ax.plot(k_vec, kp_vec_b, c='b', linestyle='--', label=legend_1)
ax.plot(k_vec, kp_vec_g, c='r', linestyle='-', label=legend_2)
ax.plot(k_vec, k_vec, 'k:', label=legend_3)
ax.set(xlabel=xlabel,
        ylabel=ylabel,
        # title='ALM after round {}'.format(idx),
        )
ax.legend()
fig.savefig(filedir+'Equilibrium_ALM_'+file_suffix+'.jpg', dpi=200)

# Figure: updates in equilibrium ALM
# plot_idx = [0, 1, 4, 9, 14, 19]
plot_idx = range(len(B_round_list))
# if len(B_list) < 8:
#     plot_idx = range(len(B_list))
# else:
#     step = 1 + len(B_list) // 4
#     plot_idx = [0, 1] + [i for i in range(2, len(B_list))[::-step]]

simulate_data = True
if simulate_data:
    zi_path, _ = mfg.env.generate_shock()
    for i in plot_idx:
        K_path_ALM = mfg.simulate_ALM(zi_path=zi_path, B=B_list[i])
        np.save(filedir+'K_path_ALM_plot_'+str(i)+'.npy', K_path_ALM)        

fig, ax = plt.subplots()
for i in plot_idx:
    K_path_ALM = np.load(filedir+'K_path_ALM_plot_'+str(i)+'.npy')
    ax.plot(K_path_ALM, label=str(i))

if not chinese_label:
    xlabel = 'period'
    ylabel = 'aggregate capital'
else:
    xlabel = '时期/步数'
    ylabel = '总资本'
ax.set(xlabel=xlabel,
    ylabel=ylabel,
    # title='Equilibrium ALM after each round',
    )    
ax.legend()
fig.savefig(filedir+'Equilibrium_ALM_after_each_round_selected_'+file_suffix+'.jpg', dpi=200)

# Figure: updates in (non-equilibrium) ALM
# plot_idx = [0, 1, 4, 9, 14, 19]
plot_idx = range(len(B_round_list))
# if len(B_round_list) < 8:
#     plot_idx = range(len(B_round_list))
# else:
#     step = 1 + len(B_round_list) // 4
#     plot_idx = [0, 1] + [i for i in range(2, len(B_round_list))[::-step]]

simulate_data = True
if simulate_data:
    zi_path, _ = mfg.env.generate_shock()
    for i in plot_idx:
        K_path_ALM = mfg.simulate_ALM(zi_path=zi_path, B=B_round_list[i])
        np.save(filedir+'K_path_ALM_nonEq'+str(i)+'.npy', K_path_ALM)

fig, ax = plt.subplots()
for i in plot_idx:
    K_path_ALM = np.load(filedir+'K_path_ALM_nonEq'+str(i)+'.npy')
    ax.plot(K_path_ALM, label=str(i))

if not chinese_label:
    xlabel = 'period'
    ylabel = 'aggregate capital'
else:
    xlabel = '时期/步数'
    ylabel = '总资本'
ax.set(xlabel=xlabel,
    ylabel=ylabel,
    # title='ALM given by each round',
    )
ax.legend()
fig.savefig(filedir+'ALM_given_by_each_round_'+file_suffix+'.jpg', dpi=200)

# Figure: policy function at K = 40 and good state
idx = n_round
K = K_0
mfg.agent.actor.load_checkpoint(idx=idx)
k_vec = np.arange(1e-3, 80, 1)[:, np.newaxis]
K_vec = K * np.ones_like(k_vec)
zi_vec_g = np.ones_like(k_vec)
epsi_vec_u = np.zeros_like(k_vec)
epsi_vec_e = np.ones_like(k_vec)
state_u = np.concatenate([k_vec, K_vec, zi_vec_g, epsi_vec_u], axis=1)
state_e = np.concatenate([k_vec, K_vec, zi_vec_g, epsi_vec_e], axis=1)
action_u = mfg.agent.actor.predict(state_u)
action_e = mfg.agent.actor.predict(state_e)
kp_vec_u = k_vec * action_u
kp_vec_e = k_vec * action_e

fig, ax = plt.subplots()
if not chinese_label:
    legend_1 = 'unemployed'
    legend_2 = 'employed'
    legend_3 = '45 degree'
    xlabel = 'capital today'
    ylabel = 'capital tomorrow'
else:
    legend_1 = '失业'
    legend_2 = '雇佣'
    legend_3 = '45度'
    xlabel = '当期资本'
    ylabel = '下期资本'
ax.plot(k_vec, kp_vec_u, c='b', linestyle='--', label=legend_1)
ax.plot(k_vec, kp_vec_e, c='r', linestyle='-', label=legend_2)
ax.plot(k_vec, k_vec, 'k:', label=legend_3)
ax.legend()
ax.set(xlabel=xlabel,
    ylabel=ylabel,
    # title='Individual policy function at K = {} when good state'.format(K),
    )
fig.savefig(filedir+'policy_good_state_'+file_suffix+'.jpg', dpi=200)
