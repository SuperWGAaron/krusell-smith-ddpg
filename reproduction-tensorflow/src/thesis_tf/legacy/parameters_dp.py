import os
from dynamic_programming import *

filedir= os.path.join(os.getcwd(), 'test', 'dynamic_programming/')
filenames = dict((
    ('k_opt', filedir + 'kss_k_opt.npy'),
    ('value', filedir + 'kss_value.npy'),
    ('B', filedir + 'B.npy'),
    ('R2', filedir + 'R2.npy')
))

# B = np.array([0.11830234577220701, 0.9671851305292907, 
#     0.14183883236187064, 0.9621930634125834])
# B = np.array([0.14199999107197706, 0.9609525795413728,
#     0.15338886785866188, 0.9592537018104501])
# np.save(filenames['B'], B)

ksp = create_KSParameter(beta=0.8, k_size=100, k_min=0.1, k_max=1000, 
    K_size=20, K_min=0.1, K_max=50)
kss = create_KSSolution(ksp)