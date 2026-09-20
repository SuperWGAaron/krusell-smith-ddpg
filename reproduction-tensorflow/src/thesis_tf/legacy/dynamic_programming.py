import os
import time
import numpy as np
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from collections import namedtuple
from scipy.interpolate import interp2d
from scipy.optimize import minimize_scalar
from sklearn import linear_model
from sklearn.metrics import r2_score

KSParameter = namedtuple('KSParameter', ['u', 'beta', 'alpha', 'delta', 'theta',
                        'l_bar', 'k_min', 'k_max', 'k_grid', 'K_min', 'K_max', 
                        'K_grid', 'z_grid', 'eps_grid', 's_grid', 'k_size', 
                        'K_size', 'z_size', 'eps_size', 's_size', 'ug', 'ub', 
                        'transmat', 'MC', 'mu'])

TransitionMatrix = namedtuple('TransitionMatrix', ['P', 'Pz', 'Peps_gg',
                        'Peps_bb', 'Peps_gb', 'Peps_bg'])


class MarkovChain(object):
    def __init__(self, transmat, x0=None):
        self.transmat = transmat
        self.n_states = transmat.shape[0]
        self.x0 = np.random.randint(self.n_states) if x0 is None else x0
        self.transmat_rowsum = np.cumsum(transmat, axis=1)

    def draw(self, xprev):
        return np.sum(np.random.uniform() > self.transmat_rowsum[xprev, :])

    def draw_path(self, xprev=None, n_obs=1):
        xprev = self.x0 if xprev is None else xprev
        x_path = [xprev,]
        for ii in range(n_obs):
            x_path.append(self.draw(x_path[-1]))
        return np.array(x_path)


def create_transmat(ug=0.04, ub=0.1, zg_avg_dur=8.0, zb_avg_dur=8.0,
                    ug_avg_dur=1.5, ub_avg_dur=2.5, 
                    puu_rel_gb2bb=1.25, puu_rel_bg2gg=0.75):

        # probability of remaining in good state
        pgg = 1.0 - 1.0 / zg_avg_dur
        # probability of remaining in bad state
        pbb = 1.0 - 1.0 / zb_avg_dur
        # probability of changing from g to b
        pgb = 1.0 - pgg
        # probability of changing from b to g
        pbg = 1.0 - pbb  
        
        # prob. of 0 to 0 cond. on g to g
        p00_gg = 1.0 - 1.0 / ug_avg_dur
        # prob. of 0 to 0 cond. on b to b
        p00_bb = 1.0 - 1.0 / ub_avg_dur
        # prob. of 0 to 1 cond. on g to g
        p01_gg = 1.0 - p00_gg
        # prob. of 0 to 1 cond. on b to b
        p01_bb = 1.0 - p00_bb
        
        # prob. of 0 to 0 cond. on g to b
        p00_gb = puu_rel_gb2bb * p00_bb
        # prob. of 0 to 0 cond. on b to g
        p00_bg = puu_rel_bg2gg * p00_gg
        # prob. of 0 to 1 cond. on g to b
        p01_gb = 1.0 - p00_gb
        # prob. of 0 to 1 cond. on b to g
        p01_bg = 1.0 - p00_bg
        
        # prob. of 1 to 0 cond. on  g to g
        p10_gg = (ug - ug*p00_gg) / (1.0 - ug)
        # prob. of 1 to 0 cond. on b to b
        p10_bb = (ub - ub*p00_bb) / (1.0 - ub)
        # prob. of 1 to 0 cond. on g to b
        p10_gb = (ub - ug*p00_gb) / (1.0 - ug)
        # prob. of 1 to 0 cond on b to g
        p10_bg = (ug - ub*p00_bg) / (1.0 - ub)
        # prob. of 1 to 1 cond. on  g to g
        p11_gg = 1.0 - p10_gg
        # prob. of 1 to 1 cond. on b to b
        p11_bb = 1.0 - p10_bb
        # prob. of 1 to 1 cond. on g to b
        p11_gb = 1.0 - p10_gb
        # prob. of 1 to 1 cond on b to g
        p11_bg = 1.0 - p10_bg
        
        #      (b0)         (b1)        (g0)       (g1)
        P= np.array([
            [pbb*p00_bb, pbb*p01_bb, pbg*p00_bg, pbg*p01_bg],
            [pbb*p10_bb, pbb*p11_bb, pbg*p10_bg, pbg*p11_bg],
            [pgb*p00_gb, pgb*p01_gb, pgg*p00_gg, pgg*p01_gg],
            [pgb*p10_gb, pgb*p11_gb, pgg*p10_gg, pgg*p11_gg]
            ])
        Pz = np.array([[pbb, pbg], [pgb, pgg]])
        Peps_bb = np.array([[p00_bb, p01_bb], [p10_bb, p11_bb]])
        Peps_bg = np.array([[p00_bg, p01_bg], [p10_bg, p11_bg]])
        Peps_gb = np.array([[p00_gb, p01_gb], [p10_gb, p11_gb]])
        Peps_gg = np.array([[p00_gg, p01_gg], [p10_gg, p11_gg]])

        transmat = TransitionMatrix(P=P, Pz=Pz, Peps_bb=Peps_bb,
                        Peps_bg=Peps_bg, Peps_gb=Peps_gb, Peps_gg=Peps_gg)

        return transmat


def create_KSParameter(beta=0.99, alpha=0.36, delta=0.025, theta=1.0, k_min=0,
    k_max=1000, k_size=100, K_min=30, K_max=50, K_size=4, z_min=0.99,
    z_max=1.01, z_size=2, eps_min=0.0, eps_max=1.0, eps_size=2, ug=0.04,
    ub=0.1, zg_avg_dur=8, zb_avg_dur=8, ug_avg_dur=1.5, ub_avg_dur=2.5,
    puu_rel_gb2bb=1.25, puu_rel_bg2gg=0.75, mu=0, degree=7):

    u = np.log if theta == 1.0 else \
        lambda x: (np.power(x, 1.0-theta) - 1.0) / (1.0 - theta)

    l_bar = 1.0 / (1.0 - ub)
    k_grid = k_min + (k_max - k_min) * \
        (np.arange(k_size) / (k_size-1)) ** degree
    k_grid[0], k_grid[-1] = k_min, k_max
    K_grid = np.linspace(K_min, K_max, K_size)
    z_grid = np.linspace(z_min, z_max, z_size)
    eps_grid = np.linspace(eps_min, eps_max, eps_size)
    s_grid = None
    s_size = z_size * eps_size
    transmat = create_transmat(ug, ub, zg_avg_dur, zb_avg_dur, ug_avg_dur,
                    ub_avg_dur, puu_rel_gb2bb, puu_rel_bg2gg)
    MC = dict((
        ('z_eps', MarkovChain(transmat.P)),
        ('z', MarkovChain(transmat.Pz)), 
        ((0, 0), MarkovChain(transmat.Peps_bb)),
        ((0, 1), MarkovChain(transmat.Peps_bg)),
        ((1, 0), MarkovChain(transmat.Peps_gb)),
        ((1, 1), MarkovChain(transmat.Peps_gg))        
    ))
    
    ksp = KSParameter(u=u, beta=beta, alpha=alpha, delta=delta, theta=theta,
         l_bar=l_bar, k_min=k_min, k_max=k_max, k_grid=k_grid,
         K_min=K_min, K_max=K_max, K_grid=K_grid, z_grid=z_grid,
         eps_grid=eps_grid, s_grid=s_grid, k_size=k_size, K_size=K_size,
         z_size=z_size, eps_size=eps_size, s_size=z_size*eps_size, 
         ug=ug, ub=ub, transmat=transmat, MC=MC, mu=mu)
    
    return ksp

def create_KSSolution(ksp, value_file=None, B_file=None):
    if value_file:
        res = np.load(value_file)
        k_opt = res['k_opt']
        value = res['value']
    else:
        k_opt = 0.9 * ksp.k_grid[:, np.newaxis, np.newaxis, np.newaxis] * \
            np.ones((ksp.k_size, ksp.K_size, ksp.z_size, ksp.eps_size))
        value = ksp.u(0.1/0.9 * k_opt) / (1.0 - ksp.beta)
    B = np.load(B_file) if B_file else np.array([0.0, 1.0, 0.0, 1.0])
    R2 = np.array([0.0, 0.0])

    kss = dict((
        ('k_opt', k_opt),
        ('value', value),
        ('B', B),
        ('R2', R2),
    ))
    return kss


def compute_intr(alpha, z, K, L):
    return alpha * z * (L / K) ** (1.0-alpha)


def compute_wage(alpha, z, K, L):
    return (1.0-alpha) * z * (K / L) ** alpha

def compute_wealth(k, K, z, eps, L, alpha, delta, l_bar, mu):
    wealth = (compute_intr(alpha, z, K, L) + 1.0 - delta) * k + \
        compute_wage(alpha, z, K, L) * (eps*l_bar + (1-eps)*mu)
    return wealth

def compute_Kp_L(K, z_ind, B, ksp):
    B0, B1, B2, B3 = B
    if z_ind == 0:
        Kp = np.exp(B0 + B1*np.log(K))
        L = ksp.l_bar * (1.0 - ksp.ub)
    else:
        Kp = np.exp(B2 + B3*np.log(K))
        L = ksp.l_bar * (1.0 - ksp.ug)
    Kp = np.clip(Kp, ksp.K_min, ksp.K_max)
    return Kp, L


def generate_shock(ksp, n_pop, n_steps):
    MC, ub, ug = ksp.MC, ksp.ub, ksp.ug
    z_path = MC['z'].draw_path(n_obs=n_steps)
    eps_path = np.zeros((n_pop, n_steps), dtype=int)
    eps_path[:, 0] = np.random.uniform(size=n_pop) > \
        (ub if z_path[0] == 0 else ug)
    # eps_path[:, 0] = eps_path[:, 0].astype(bool)
    for t in range(n_steps-1):
        z_index = (z_path[t], z_path[t+1])
        for i in range(n_pop):
            eps_path[i, t+1] = MC[z_index].draw(eps_path[i, t])

    if n_pop >= 100:
        ideal_um_b, ideal_um_g = int(ub*n_pop), int(ug*n_pop)
        for t in range(n_steps):
            actual_um = np.sum(eps_path[:, t] == 0)
            ideal_um = ideal_um_b if z_path[t] == 0 else ideal_um_g
            gap = ideal_um - actual_um
            if gap > 0:
                replace = np.random.choice(np.where(eps_path[:, t] == 1)[0],
                    size=gap, replace=False)
                eps_path[replace, t] = 0
            elif gap < 0:
                replace = np.random.choice(np.where(eps_path[:, t] == 0)[0],
                    size=-gap, replace=False)
                eps_path[replace, t] = 1

    return z_path, eps_path


def solve_UMP(ksp, kss, max_iter=100, tol=1e-3, method='VFI', filename=None):
    k_size, K_size, z_size, eps_size = ksp.k_size, ksp.K_size, ksp.z_size, ksp.eps_size
    ctr = 0
    print('Start solving UMP by VFI...')
    while True:
        time_start = time.time()
        value_old = np.copy(kss['value'])
        for z_ind in range(z_size):
            for eps_ind in range(eps_size):
                [max_rhs(k_ind, K_ind, z_ind, eps_ind, ksp, kss)
                    for k_ind in range(k_size) for K_ind in range(K_size)]

        print('Round = {:d}, time spent = {:.4f}'.format(ctr, time.time()-time_start))

        if filename is not None:
            np.savez(filename, value=kss['value'], k_opt=kss['k_opt'])

        if np.max(np.abs(kss['value'] - value_old)) < tol:
            print('Solve UMP by VFI successfully! Rounds run = %d' % ctr)
            break
        elif ctr == max_iter - 1:
            print('Solve UMP by VFI failed! Rounds run = %d' % ctr)
            break
        ctr += 1
    return ksp, kss


def max_rhs(k_ind, K_ind, z_ind, eps_ind, ksp, kss):
    time_start = time.time()
    alpha, delta, l_bar, mu = ksp.alpha, ksp.delta, ksp.l_bar, ksp.mu
    k, K, z, eps = ksp.k_grid[k_ind], ksp.K_grid[K_ind], ksp.z_grid[z_ind], ksp.eps_grid[eps_ind]
    k_min, k_max = ksp.k_grid[0], ksp.k_grid[-1]

    Kp, L = compute_Kp_L(K, z_ind, kss['B'], ksp)
    wealth = compute_wealth(k, K, z, eps, L, alpha, delta, l_bar, mu)

    obj = lambda kp: - rhs_bellman(k, kp, K, z_ind, eps_ind, ksp, kss)
    res = minimize_scalar(obj, bounds=(k_min, min(k_max, wealth)), method='bounded')
    kss['value'][k_ind, K_ind, z_ind, eps_ind] = res.fun
    kss['k_opt'][k_ind, K_ind, z_ind, eps_ind] = res.x
    print('time = {}'.format(time.time()-time_start))
    return res.fun


def rhs_bellman(k, kp, K, z_ind, eps_ind, ksp, kss):
    alpha, delta, l_bar, mu = ksp.alpha, ksp.delta, ksp.l_bar, ksp.mu
    z, eps = ksp.z_grid[z_ind], ksp.eps_grid[eps_ind]
    Kp, L = compute_Kp_L(K, z_ind, kss['B'], ksp)
    wealth = compute_wealth(k, K, z, eps, L, alpha, delta, l_bar, mu)
    con = wealth - kp
    expec = compute_expectation(kp, Kp, z_ind, eps_ind, ksp, kss)
    return ksp.u(con) + ksp.beta*expec


def compute_expectation(kp, Kp, z_ind, eps_ind, ksp, kss):
    beta, P = ksp.beta, ksp.transmat.P
    k_grid, K_grid = ksp.k_grid, ksp.K_grid
    k_grid_mesh, K_grid_mesh = np.meshgrid(k_grid, K_grid)
    z_eps_ind = z_ind * ksp.z_size + eps_ind
    expec = 0

    for z_eps_p_ind in range(ksp.s_size):
        value_itp = interp2d(k_grid_mesh, K_grid_mesh, 
            kss['value'][:, :, z_ind, eps_ind], kind='linear')
        expec += P[z_eps_ind, z_eps_p_ind] * value_itp(kp, Kp)
    return expec


def simulate_capital_path(ksp, k0, z_ind_path, eps_ind_path, kss):
    n_pop, n_steps = eps_ind_path.shape
    k_grid, K_grid = ksp.k_grid, ksp.K_grid
    k_grid_mesh, K_grid_mesh = np.meshgrid(k_grid, K_grid)
    k_opt = dict((
        ((0, 0), interp2d(K_grid, k_grid, kss['k_opt'][:, :, 3])),
        ((0, 1), interp2d(K_grid, k_grid, kss['k_opt'][:, :, 1])),
        ((1, 0), interp2d(K_grid, k_grid, kss['k_opt'][:, :, 2])),
        ((1, 1), interp2d(K_grid, k_grid, kss['k_opt'][:, :, 0])),
    ))

    k_path = np.zeros((n_pop, n_steps))
    k_path[:, 0] = k0
    K_path = np.zeros(n_steps)
    K_path[0] = k0
    for t in range(n_steps-1):
        z_ind = z_ind_path[t]
        for i in range(n_pop):
            eps_ind = eps_ind_path[i, t]
            k_path[i, t+1] = k_opt[(z_ind, eps_ind)](K_path[t], k_path[i, t])
        K_path[t+1] = np.mean(k_path[:, t+1])

    return k_path, K_path


def regress_ALM(K_path, z_path):
    bad_times = z_path[:-1] == np.min(z_path)
    good_times = z_path[:-1] == np.max(z_path)
    K_b, K_g = K_path[:-1][bad_times], K_path[:-1][good_times]
    Kp_b, Kp_g = K_path[1:][bad_times], K_path[1:][good_times]
    n_b, n_g = len(K_b), len(K_g)
    X_b = np.array([np.ones(nb), K_b]).T
    X_g = np.array([np.ones(ng), K_g]).T

    reg = linear_model.LinearRegression()
    reg.fit(X_b, Kp_b)
    B0, B1 = reg.intercept_, reg.coef_[0]
    Kp_b_pred = reg.predict(X_b)
    R2_b = r2_score(Kp_b, Kp_b_pred)

    reg.fit(X_g, Kp_g)
    B2, B3 = reg.intercept_, reg.coef_[0]
    Kp_g_pred = reg.predict(X_g)
    R2_g = r2_score(Kp_g, Kp_g_pred)

    return [B0, B1, B2, B3], [R2_b, R2_g]


def genearte_paths(ksp, k0, z_ind_path, eps_ind_path, kss):
    n_pop, n_steps = eps_ind_path.shape
    k_path, K_path = simulate_capital_path(
        ksp, k0, z_ind_path, eps_ind_path, kss)
    c_path, u_path, w_path = np.zeros((n_pop, n_steps)), \
        np.zeros((n_pop, n_steps)), np.zeros((n_pop, n_steps))

    alpha, delta, l_bar, mu, ub, ug = ksp.alpha, ksp.delta, ksp.l_bar, ksp.mu, ksp.ub, ksp.ug
    z_grid, eps_grid = ksp.z_grid, ksp.eps_grid
    z_min = z_grid[0]
    u = ksp.u
    for t in range(n_steps-1):
        z_ind, K = z_ind_path[t], K_path[t]
        z = z_grid[z_ind]
        L = l_bar * (1.0 - (ub if z == z_min else ug))
        for i in range(n_pop):
            k, kp, eps = k_path[i, t], k_path[i, t+1], eps_grid[eps_ind_path[i, t]]
            wealth = compute_wealth(k, K, z, eps, L, alpha, delta, l_bar, mu)
            cons = wealth - kp
            w_path[i, t] = wealth
            c_path[i, t] = cons
            u_path[i, t] = u(cons)
    return w_path, c_path, u_path


def plot_ALM(K_path, z_ind_path, B, T_discard=1, filename=None, dpi=None):
    B0, B1, B2, B3 = B
    B_cons = B0 * (1-z_ind_path) + B2 * z_ind_path
    B_slope = B1 * (1-z_ind_path) + B3 * z_ind_path
    K_path_approx = np.zeros_like(K_path)
    K_path_approx[T_discard-1] = K_path[T_discard-1]
    time_index = np.arange(T_discard, len(K_path))
    for t in time_index:
        K_path_approx[t] = np.exp(B_cons[t-1] + B_slope[t-1]*np.log(K_path_approx[t-1]))

    fig, ax = plt.subplots()
    ax.plot(np.arange(T_discard-1, len(K_path)),
        K_path_approx[T_discard-1:], 
        label='approximation by ALM',   # 总资本运动律近似, approximation by ALM
        color='b',
        linestyle='--')
    ax.plot(np.arange(T_discard-1, len(K_path)),
        K_path[T_discard-1:],
        label='simulation',     # 蒙特卡洛模拟, simulation
        color='r',
        linestyle='-')
    ax.set(xlabel='period',     # 时期/步数, period
        ylabel='aggregate capital',     # 总资本, aggregate capital
        # title='Equilibrium capital path: simulation vs. approximation',
        )
    ax.legend()
    if filename is not None:
        fig.savefig(filename, dpi=dpi)


def plot_fig1(K_path, B, filename=None):
    B0, B1, B2, B3 = B
    K_min, K_max = min(K_path), max(K_path)
    K_grid = np.linspace(K_min, K_max, 100)
    K_b = np.exp(B0 + B1*np.log(K_grid))
    K_g = np.exp(B2 + B3*np.log(K_grid))

    fig, ax = plt.subplots()
    ax.plot(K_grid, K_b, label='Bad', c='b', linestyle=':')
    ax.plot(K_grid, K_g, label='Good', c='r', linestyle='-')
    ax.plot(K_grid, K_grid, label='45 degree')
    ax.set_title("FIG1: Tomorrow's vs. today's aggregate capital")
    ax.legend()
    plt.show()
    if filename is not None:
        plt.save(filename)


def plot_fig2(ksp, kss, k_eval_point):
    pass

