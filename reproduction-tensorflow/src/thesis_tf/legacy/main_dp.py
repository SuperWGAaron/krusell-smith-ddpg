import os
import matplotlib.pyplot as plt
from dynamic_programming import *
from parameters_dp import *

# allow labels in Chinese characters
chinese_label = os.environ.get('THESIS_TF_ENGLISH', '1') != '1'
if chinese_label: 
    from matplotlib import rcParams
    rcParams['font.family'] = ['Heiti TC']
file_suffix = 'eng' if not chinese_label else 'chn'

# # solve for policy, value function, and ALM
# ksp, kss = solve_UMP(ksp, kss, max_iter=100, filename=filenames['kss'])
kss['B'] = np.load(filenames['B'])
kss['value'] = np.load(filenames['value'])
kss['k_opt'] = np.load(filenames['k_opt'])

# plot one simulated capital path, ALM, and policy
k0 = 40
n_pop, n_steps = 100, 30
n_round = 1000

# Figure: aggregate capital path (simulation vs approximation)
simulate_data = os.environ.get('THESIS_TF_RESIMULATE', '0') == '1'
T_discard = 1
if simulate_data:
    z_ind_path, eps_ind_path = generate_shock(ksp, n_pop, n_steps)
    _, K_path = simulate_capital_path(ksp, k0, z_ind_path, eps_ind_path, kss)

    B = kss['B']
    B0, B1, B2, B3 = B
    B_cons = B0 * (1-z_ind_path) + B2 * z_ind_path
    B_slope = B1 * (1-z_ind_path) + B3 * z_ind_path
    K_path_approx = np.zeros_like(K_path)
    K_path_approx[T_discard-1] = K_path[T_discard-1]
    time_index = np.arange(T_discard, len(K_path))
    for t in time_index:
        K_path_approx[t] = np.exp(B_cons[t-1] + B_slope[t-1]*np.log(K_path_approx[t-1]))
    
    np.savez(filedir+'paths_plot.npz',
             K_path=K_path,
             K_path_approx=K_path_approx)
else:
    paths_plot =  np.load(filedir+'paths_plot.npz')
    K_path = paths_plot['K_path']
    K_path_approx = paths_plot['K_path_approx']

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
ax.plot(np.arange(T_discard-1, len(K_path)),
    K_path_approx[T_discard-1:], 
    label=legend_1,
    color='b',
    linestyle='--')
ax.plot(np.arange(T_discard-1, len(K_path)),
    K_path[T_discard-1:],
    label=legend_2,
    color='r',
    linestyle='-')
ax.set(xlabel=xlabel,
    ylabel=ylabel,
    # title='Equilibrium capital path: simulation vs. approximation',
    )
ax.legend()
fig.savefig(filedir+'DP_Equilibrium_K_path_'+file_suffix+'.jpg', dpi=200)

# Figure: ALM
B0, B1, B2, B3 = kss['B']   # (B0, B1) for good times, (B2, B3) for bad times
k_vec = np.arange(1e-3, 80, 1)
kp_vec_b = np.zeros_like(k_vec)
kp_vec_g = np.zeros_like(k_vec)
for i in range(len(k_vec)):
    kp_vec_b[i] = np.exp(B2 + B3 * np.log(k_vec[i]))
    kp_vec_g[i] = np.exp(B0 + B1 * np.log(k_vec[i]))

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
fig.savefig(filedir+'DP_Equilibrium_ALM_'+file_suffix+'.jpg', dpi=200)

# Figure: policy function
k_vec = np.arange(1e-3, 80, 1)
kp_vec_u, kp_vec_e = np.zeros_like(k_vec), np.zeros_like(k_vec)
k_opt_u = interp2d(ksp.K_grid, ksp.k_grid, kss['k_opt'][:, :, 2])
k_opt_e = interp2d(ksp.K_grid, ksp.k_grid, kss['k_opt'][:, :, 0])
for i, k in enumerate(k_vec):
    kp_vec_u[i] = k_opt_u(k0, k)
    kp_vec_e[i] = k_opt_e(k0, k)

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
ax.plot(k_vec, kp_vec_u, c='b', linestyle='--', label=legend_1)     # 失业, unemployed
ax.plot(k_vec, kp_vec_e, c='r', linestyle='-', label=legend_2)    # 雇佣, employed
ax.plot(k_vec, k_vec, 'k:', label=legend_3)  # 45度, 45 degree
ax.set(xlabel=xlabel,  # 当期资本, capital today
    ylabel=ylabel,  # 下期资本, capital tomorrow
    # title='Individual policy function at K = {} when good state'.format(K),
    )
ax.legend()
fig.savefig(filedir+'DP_policy_good_state_'+file_suffix+'.jpg', dpi=200)


# fig, ax = plt.subplots(1, 3)
# ax[0].plot(k_path[0], color='b')
# ax[0].plot(c_path[0], color='r')
# ax[1].plot(w_path[0], label='wealth')
# ax[2].plot(u_path[0], label='utility')
# ax[0].set_title('capital(blue) and consumption(red)')
# ax[1].set_title('wealth')
# ax[2].set_title('utility. Total = {}'.format(np.nansum(u_path, axis=1)[0]))
# fig.legend()
# plt.savefig(filedir+'paths_500.png')
# plt.show()

