import numpy as np
import os
from ddpg import create_transmat
from utils import MarkovChain

transmat_pm = dict((
    ('ug', 0.04),
    ('ub', 0.1),
    ('zg_avg_dur', 8),
    ('zb_avg_dur', 8),
    ('ug_avg_dur', 1.5),
    ('ub_avg_dur', 2.5),
    ('puu_rel_gb2bb', 1.25),
    ('puu_rel_bg2gg', 0.75),
))

env_pm = dict((
    ('beta', 0.8),
    ('alpha', 0.36),
    ('delta', 0.025),
    ('theta', 1.0),
    ('mu', 0.0),
    ('z_range', (0.99, 1.01)),
    ('eps_range', (0, 1)),
    ('l_bar', 1.0 / (1.0 - transmat_pm['ub'])),
))

transmat = create_transmat(transmat_pm)
mc = dict((key, MarkovChain(value)) for (key, value) in transmat.items())
utility = np.log if env_pm['theta'] == 1.0 else \
            lambda x: (np.power(x, 1.0-env_pm['theta']) - 1.0) / (1.0 - env_pm['theta'])

ksp = {}
ksp.update(transmat_pm)
ksp.update(env_pm)
ksp.update({'transmat': transmat})
ksp.update({'mc': mc})
ksp.update({'u': utility})

net_pm = dict((
    ('state_dims', [4,]),
    ('action_dims', 1),
    ('lr_actor', 0.00001),
    ('lr_critic', 0.00005),
    ('layer1_size', 200),
    ('layer2_size', 100),
    ('layer3_size', 300),
    ('tau', 0.01),
    ('max_size', 1000000),
    ('batch_size', 1024),
    ('gamma', ksp['beta']),
))

n_pop = int(os.environ.get('THESIS_TF_N_POP', '100'))
n_steps = int(os.environ.get('THESIS_TF_N_STEPS', '30'))
n_life, n_round = int(os.environ.get('THESIS_TF_N_LIFE', '5000')), int(os.environ.get('THESIS_TF_N_ROUND', '20'))
n_life_0 = int(os.environ.get('THESIS_TF_N_LIFE_0', '10000'))
idx = int(os.environ.get('THESIS_TF_IDX', '50'))

foldername = 'round_{}_life_{}_step_{}_pop_{}_idx_{}'.format(n_round, n_life, n_steps, n_pop, idx)
filedir = os.path.join(os.getcwd(), 'test', foldername, '')
os.makedirs(filedir+'ckpt/', exist_ok=True)

K_0 = float(os.environ.get('THESIS_TF_K_0', '40'))
B = None # np.load(os.getcwd()+'/test/B.npy')
