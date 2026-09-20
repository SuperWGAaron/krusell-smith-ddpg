import os
import numpy as np
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import seaborn as sns
import dynamic_programming as dp
import parameters_dp as pmdp
import ddpg
import parameters_ddpg as pmddpg

# allow labels in Chinese characters
chinese_label = os.environ.get('THESIS_TF_ENGLISH', '1') != '1'
if chinese_label:
    from matplotlib import rcParams
    rcParams['font.family'] = ['Heiti TC']
    # rc('font', **{'family': 'serif','serif':['Heiti TC']})    
    # plt.rcParams['pdf.fonttype'] = 42
file_suffix = 'eng' if not chinese_label else 'chn'

filedir_ddpg = pmddpg.filedir
filedir_dp = pmdp.filedir

seed = int(os.environ.get('THESIS_TF_SEED', '0'))
np.random.seed(seed)

k0 = 40
n_pop, n_steps = 1, 30

# dynamic programming
filenames = pmdp.filenames
ksp, kss = pmdp.ksp, pmdp.kss
kss['B'] = np.load(filenames['B'])
kss['value'] = np.load(filenames['value'])
kss['k_opt'] = np.load(filenames['k_opt'])
kss['R2'] = np.load(filenames['R2'])

# DDPG
mfg = ddpg.MFG(ksp=pmddpg.ksp, n_steps=n_steps, K_0=k0, B=None, net_pm=pmddpg.net_pm, filedir=filedir_ddpg)
mfg.agent.actor.load_checkpoint(idx=pmddpg.n_round, path=os.path.join(filedir_ddpg, 'ckpt', 'Actor'))

# simulate paths of consumption, capital, and wealth
simulate_data = os.environ.get('THESIS_TF_RESIMULATE', '0') == '1'
if simulate_data:
    # dynamic programming
    z_ind_path, eps_ind_path = dp.generate_shock(ksp, n_pop, n_steps)
    k_path_dp, _ = dp.simulate_capital_path(ksp, k0, z_ind_path, eps_ind_path, kss)
    w_path_dp, c_path_dp, u_path_dp = dp.genearte_paths(ksp, k0, z_ind_path, eps_ind_path, kss)
    k_path_dp, w_path_dp, c_path_dp, u_path_dp = k_path_dp[0], w_path_dp[0], c_path_dp[0], u_path_dp[0]

    # DDPG
    paths = mfg.simulate_dist(zi_path=z_ind_path, epsi_path=eps_ind_path, n_pop=n_pop, n_steps=n_steps, random=False)
    k_path, w_path, c_path, u_path = paths['k'], paths['w'], paths['c'], paths['u']
    k_path, w_path, c_path, u_path = k_path[0], w_path[0], c_path[0], u_path[0]

    # save data
    paths_plot_dp = np.savez(filedir_dp+'paths_plot_dp.npz',
                            k_path=k_path_dp,
                            w_path=w_path_dp,
                            c_path=c_path_dp,
                            u_path=u_path_dp)
    paths_plot_ddpg = np.savez(filedir_ddpg+'paths_plot_ddpg.npz',
                               k_path=k_path,
                               w_path=w_path,
                               c_path=c_path,
                               u_path=u_path)
else:
    # read data
    paths_plot_dp = np.load(filedir_dp+'paths_plot_dp.npz')
    paths_plot_ddpg = np.load(filedir_ddpg+'paths_plot_ddpg.npz')
    k_path_dp, w_path_dp, c_path_dp, u_path_dp = (paths_plot_dp['k_path'], 
        paths_plot_dp['w_path'], paths_plot_dp['c_path'], paths_plot_dp['u_path'])
    k_path, w_path, c_path, u_path = (paths_plot_ddpg['k_path'], 
        paths_plot_ddpg['w_path'], paths_plot_ddpg['c_path'], paths_plot_ddpg['u_path'])

# Figure: consumption and capital over one path
fig, ax = plt.subplots()
if not chinese_label:
    legend_1 = 'consumption - dynamic programming'
    legend_2 = 'capital - dynamic programming'
    legend_3 = 'consumption - DDPG'
    legend_4 = 'capital - DDPG'
    xlabel = 'period'
    ylabel = 'consumption/capital'
else:
    legend_1 = '消费(动态规划)'
    legend_2 = '资本(动态规划)'
    legend_3 = '消费(DDPG)'
    legend_4 = '资本(DDPG)'
    xlabel = '时期/步数'
    ylabel = '消费/资本'
ax.plot(c_path_dp, c='r', linestyle='-', label=legend_1) # 消费(动态规划), consumption - dynamic programming
ax.plot(k_path_dp, c='r', linestyle='--', label=legend_2)    # 资本(动态规划), capital - dynamic programming
ax.plot(c_path, c='b', linestyle='-', label=legend_3)   # 消费(DDPG), consumption - DDPG
ax.plot(k_path, c='b', linestyle='--', label=legend_4)  # 资本(DDPG), capital - DDPG
ax.set(xlabel=xlabel,     # 时期/步数, period
       ylabel=ylabel)    # 消费/资本, consumption/capital
ax.legend()
fig.savefig(filedir_dp+'comparison_consumption_capital_'+file_suffix+'.jpg', dpi=200)

# Figure: wealth over one path
fig, ax = plt.subplots()
if not chinese_label:
    legend_1 = 'dynamic programming'
    legend_2 = 'DDPG'
    xlabel = 'period'
    ylabel = 'income'
else:
    legend_1 = '动态规划'
    legend_2 = 'DDPG'
    xlabel = '时期/步数'
    ylabel = '收入'
ax.plot(w_path_dp, c='r', linestyle='-', label=legend_1)   # 动态规划, dynamic programming
ax.plot(w_path, c='b', linestyle='--', label=legend_2)
ax.set(xlabel=xlabel,     # 时期/步数, period
       ylabel=ylabel)     # 收入, income
ax.legend()
fig.savefig(filedir_dp+'comparison_income_'+file_suffix+'.jpg', dpi=200)

# Figure: utility over one path
fig, ax = plt.subplots()
if not chinese_label:
    legend_1 = 'dynamic programming'
    legend_2 = 'DDPG'
    xlabel = 'period'
    ylabel = 'utility'
else:
    legend_1 = '动态规划'
    legend_2 = 'DDPG'
    xlabel = '时期/步数'
    ylabel = '效用'
ax.plot(u_path_dp, c='r', linestyle='-', label=legend_1)   # 动态规划, dynamic programming
ax.plot(u_path, c='b', linestyle='--', label=legend_2)
ax.set(xlabel=xlabel,     # 时期/步数, period
       ylabel=ylabel)     # 效用, utility
ax.legend()
fig.savefig(filedir_dp+'comparison_utility_'+file_suffix+'.jpg', dpi=200)

# simulation of discounted utility
simulate_data = os.environ.get('THESIS_TF_RESIMULATE', '0') == '1'
if simulate_data:
    n_round = 500
    n_pop = 5000
    utility_dp, utility_ddpg = [], []
    discount = np.power(pmddpg.ksp['beta'], range(n_steps))

    print('simulating discounted utility by DP...')
    print('round = ', end='', flush=True)
    ctr = 0
    while ctr < n_round:
        print('{}...'.format(ctr), end='', flush=True)
        z_ind_path, eps_ind_path = dp.generate_shock(ksp, n_pop=n_pop, n_steps=n_steps)
        _, _, u_path_dp = dp.genearte_paths(ksp, k0, z_ind_path, eps_ind_path, kss)
        u_vec_dp = np.sum(u_path_dp * discount, axis=1)
        if np.isfinite(u_vec_dp).all():
            utility_dp.append(u_vec_dp)
            ctr += 1
        else:
            print('FAILED...', end='', flush=True)

    print('\n simulating discounted utility by DDPG...')
    print('round = ', end='', flush=True)
    ctr = 0
    while ctr < n_round:
        print(ctr, '...', end='', flush=True)
        zi_path, epsi_path = mfg.env.generate_shock(n_steps=n_steps, n_pop=n_pop)
        paths = mfg.simulate_dist(zi_path, epsi_path, n_pop=n_pop, n_steps=n_steps)
        u_path_ddpg = paths['u']
        u_vec_ddpg = np.sum(u_path_ddpg * discount, axis=1)
        if np.isfinite(u_vec_ddpg).all():
            utility_ddpg.append(u_vec_ddpg)
            ctr += 1
        else:
            print('FAILED...', end='', flush=True)

    # write simulation data
    utility_dp = np.array(utility_dp).flatten()
    utility_ddpg = np.array(utility_ddpg).flatten()
    np.save(filedir_dp + 'DP_utility.npy', utility_dp)
    np.save(filedir_ddpg + 'DDPG_utility.npy', utility_ddpg)

# simulation of ALM
simulate_data = os.environ.get('THESIS_TF_RESIMULATE', '0') == '1'
if simulate_data:
    n_round = 300
    n_pop = 5000

    B_dp = kss['B']
    reg_stat_ddpg = np.load(filedir_ddpg+'reg_stat.npz', allow_pickle=True)
    B_ddpg = reg_stat_ddpg['B'][-1]

    print('\n simulating ALM by DP...')
    print('round = ', end='', flush=True)
    B0, B1, B2, B3 = B_dp
    Kb_dp, Kg_dp, Kb_ALM_dp, Kg_ALM_dp = [], [], [], []
    for ctr in range(n_round):
        print('{}...'.format(ctr), end='', flush=True)
        zi_path, epsi_path = mfg.env.generate_shock(n_steps=n_steps, n_pop=n_pop)
        _, K_dp = dp.simulate_capital_path(ksp, k0, zi_path, epsi_path, kss)
        for t in range(n_steps-1):
            if zi_path[t] == 0:
                K_ = np.exp(B0 + B1*np.log(K_dp[t]))
                Kb_dp.append(K_dp[t+1])
                Kb_ALM_dp.append(K_)
            else:
                K_ = np.exp(B2 + B3*np.log(K_dp[t]))
                Kg_dp.append(K_dp[t+1])
                Kg_ALM_dp.append(K_)

    print('\n simulating ALM by DDPG...')
    print('round = ', end='', flush=True)
    B0, B1, B2, B3 = B_ddpg
    Kb_ddpg, Kg_ddpg, Kb_ALM_ddpg, Kg_ALM_ddpg = [], [], [], []
    for ctr in range(n_round):
        print(ctr, '...', end='', flush=True)
        zi_path, epsi_path = mfg.env.generate_shock(n_steps=n_steps, n_pop=n_pop) 
        paths = mfg.simulate_dist(zi_path, epsi_path, n_pop=n_pop, n_steps=n_steps)
        K_ddpg = paths['K']

        for t in range(n_steps-1):
            if zi_path[t] == 0:
                K_ = np.exp(B0 + B1*np.log(K_ddpg[t]))
                Kb_ddpg.append(K_ddpg[t+1])
                Kb_ALM_ddpg.append(K_)
            else:
                K_ = np.exp(B2 + B3*np.log(K_ddpg[t]))
                Kg_ddpg.append(K_ddpg[t+1])
                Kg_ALM_ddpg.append(K_)

        # K_ddpg, K_ALM_ddpg = K_dp[:-1], K_ALM_dp[:-1]
        # K_ddpg, K_ALM_ddpg = K_ddpg[:-1], K_ALM_ddpg[:-1]
        # Kb_dp_, Kg_dp_ = K_dp[bad_times], K_dp[good_times]
        # Kb_ALM_dp_, Kg_ALM_dp_ = K_ALM_dp[bad_times], K_ALM_dp[good_times]
        # Kb_ddpg_, Kg_ddpg_ = K_ddpg[bad_times], K_ddpg[good_times]
        # Kb_ALM_ddpg_, Kg_ALM_ddpg_ = K_ALM_ddpg[bad_times], K_ALM_ddpg[good_times]

        # Kb_dp.append(Kb_dp_)
        # Kg_dp.append(Kg_dp_)
        # Kb_ALM_dp.append(Kb_ALM_dp_)
        # Kg_ALM_dp.append(Kg_ALM_dp_)
        # Kb_ddpg.append(Kb_ddpg_)
        # Kg_ddpg.append(Kg_ddpg_)
        # Kb_ALM_ddpg.append(Kb_ALM_ddpg_)
        # Kg_ALM_ddpg.append(Kg_ALM_ddpg_)

    # write simulation data
    Kb_dp, Kg_dp = np.array(Kb_dp).flatten(), np.array(Kg_dp).flatten()
    Kb_ALM_dp, Kg_ALM_dp = np.array(Kb_ALM_dp).flatten(), np.array(Kg_ALM_dp).flatten()
    Kb_ddpg, Kg_ddpg = np.array(Kb_ddpg).flatten(), np.array(Kg_ddpg).flatten()
    Kb_ALM_ddpg, Kg_ALM_ddpg = np.array(Kb_ALM_ddpg).flatten(), np.array(Kg_ALM_ddpg).flatten()
    np.save(filedir_dp + 'DP_Kb.npy', Kb_dp)
    np.save(filedir_dp + 'DP_Kg.npy', Kg_dp)
    np.save(filedir_dp + 'DP_Kb_ALM.npy', Kb_ALM_dp)
    np.save(filedir_dp + 'DP_Kg_ALM.npy', Kg_ALM_dp)
    np.save(filedir_ddpg + 'DDPG_Kb.npy', Kb_ddpg)
    np.save(filedir_ddpg + 'DDPG_Kg.npy', Kg_ddpg)
    np.save(filedir_ddpg + 'DDPG_Kb_ALM.npy', Kb_ALM_ddpg)
    np.save(filedir_ddpg + 'DDPG_Kg_ALM.npy', Kg_ALM_ddpg)

# read simulation data
utility_dp = np.load(filedir_dp+'DP_utility.npy')
utility_ddpg = np.load(filedir_ddpg+'DDPG_utility.npy')
Kb_dp = np.load(filedir_dp + 'DP_Kb.npy')
Kg_dp = np.load(filedir_dp + 'DP_Kg.npy')
Kb_ALM_dp = np.load(filedir_dp + 'DP_Kb_ALM.npy')
Kg_ALM_dp = np.load(filedir_dp + 'DP_Kg_ALM.npy')
Kb_ddpg = np.load(filedir_ddpg + 'DDPG_Kb.npy')
Kg_ddpg = np.load(filedir_ddpg + 'DDPG_Kg.npy')
Kb_ALM_ddpg = np.load(filedir_ddpg + 'DDPG_Kb_ALM.npy')
Kg_ALM_ddpg = np.load(filedir_ddpg + 'DDPG_Kg_ALM.npy')

# Figure: density plot
fig, ax = plt.subplots()
if not chinese_label:
    legend_1 = 'dynamic programming'
    legend_2 = 'DDPG'
    xlabel = 'sum of discounted utility'
    ylabel = 'density'
else:
    legend_1 = '动态规划'
    legend_2 = 'DDPG'
    xlabel = '折现效用和'
    ylabel = '密度'
sns.distplot(utility_dp, ax=ax, color='r', label=legend_1) # 动态规划, dynamic programming
sns.distplot(utility_ddpg, ax=ax, color='b', label=legend_2)
ax.set_xlabel(xlabel)     # 折现效用和, sum of discounted utility
ax.set_ylabel(ylabel)        # 密度, density
# ax.set_title('Empirical distribution of discounted utility')
ax.legend()
fig.savefig(filedir_dp+'utility_'+file_suffix+'.jpg', dpi=200)


# summary statistics
mean_dp, std_dp, max_dp, min_dp = np.mean(utility_dp), \
    np.std(utility_dp), np.max(utility_dp), np.min(utility_dp)
mean_ddpg, std_ddpg, max_ddpg, min_ddpg = np.mean(utility_ddpg), \
    np.std(utility_ddpg), np.max(utility_ddpg), np.min(utility_ddpg)
rel = (mean_ddpg - mean_dp) / mean_dp
RMSE_b_dp = np.sqrt(np.mean(np.power(Kb_dp - Kb_ALM_dp, 2)))
RMSE_g_dp = np.sqrt(np.mean(np.power(Kg_dp - Kg_ALM_dp, 2)))
RMSE_b_ddpg = np.sqrt(np.mean(np.power(Kb_ddpg - Kb_ALM_ddpg, 2)))
RMSE_g_ddpg = np.sqrt(np.mean(np.power(Kg_ddpg - Kg_ALM_ddpg, 2)))
print(RMSE_b_dp, RMSE_g_dp, RMSE_b_ddpg, RMSE_g_ddpg)

B_dp = kss['B']
reg_stat_ddpg = np.load(filedir_ddpg+'reg_stat.npz', allow_pickle=True)
B_ddpg = reg_stat_ddpg['B'][-1]
R2_dp = kss['R2']
R2_ddpg = reg_stat_ddpg['R2'][-1]
with open(filedir_dp+'summary.txt', 'w') as f:
    f.write('Using TensorFlow {}, NumPy {} \n'.format(ddpg.tf.__version__, np.__version__))
    f.write('Numpy random seed = {} \n\n'.format(seed))
    f.write('Evaluation of discounted utility from simulation \n')
    f.write('dynamic programming \n')
    f.write('mean = {}, std = {} \n'.format(mean_dp, std_dp))
    f.write('max = {}, min = {} \n'.format(max_dp, min_dp))
    f.write('DDPG \n')
    f.write('mean = {}, std = {} \n'.format(mean_ddpg, std_ddpg))
    f.write('max = {}, min = {} \n'.format(max_ddpg, min_ddpg))
    f.write('relative difference in mean of DDPG over DP = {} \n\n'.format(rel))
    f.write('Aggregate law of motion of capital \n')
    f.write('dynamic programming \n')
    f.write('good: cons = {}, slope = {}, R2 = {}, RMSE = {} \n'.format(
        B_dp[0], B_dp[1], R2_dp[0], RMSE_g_dp))
    f.write('bad: cons = {}, slope = {}, R2 = {}, RMSE = {} \n'.format(
        B_dp[2], B_dp[3], R2_dp[1], RMSE_b_dp))
    f.write('DDPG \n')
    f.write('good: cons = {}, slope = {}, R2 = {}, RMSE = {} \n'.format(
        B_ddpg[2], B_ddpg[3], R2_ddpg[1], RMSE_g_ddpg))
    f.write('bad: cons = {}, slope = {}, R2 = {}, RMSE = {} \n'.format(
        B_ddpg[0], B_ddpg[1], R2_ddpg[0], RMSE_b_ddpg))

    f.close()
with open(filedir_dp+'summary.txt', 'r') as f:
    print(f.read())
