import os
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.initializers import random_uniform
import statsmodels.api as sm
from utils import MarkovChain


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)


class ReplayBuffer_ddpg(object):
    def __init__(self, max_size, state_shape, action_dims):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *state_shape))
        self.new_state_memory = np.zeros((self.mem_size, *state_shape))
        self.action_memory = np.zeros((self.mem_size, action_dims))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        
        # not_done = np.where(self.terminal_memory == 1.0)[0]
        # if batch_size > sum(not_done):
        #     return None, None, None, None, None
        not_done = max_mem
        batch = np.random.choice(not_done, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class ReplayBuffer_pretrain(object):
    def __init__(self, max_size, state_shape, action_dims):
        self.max_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.max_size, *state_shape))
        self.action_memory = np.zeros((self.max_size, action_dims))

    def store(self, states, actions):
        batch_size = np.shape(states)[0]
        index = (np.arange(batch_size) + self.mem_cntr) % self.max_size
        self.state_memory[index] = states
        self.action_memory[index] = actions
        self.mem_cntr += batch_size
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.max_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        return states, actions

# ckpt_dir=os.path.join(os.getcwd(), 'temp/ddpg')
# checkpoint_file = os.path.join(ckpt_dir, name +'_ddpg.ckpt')


class Actor(object):
    def __init__(self, name, state_dims, action_dims,
                 sess, lr, fc1_dims, fc2_dims, batch_size=64, lr_pretrain=0.00001,
                 filedir=os.path.join(os.getcwd())):
        self.name = name    
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.sess = sess
        self.lr = lr
        self.lr_pretrain = lr_pretrain
        self.pretrain_buffer = ReplayBuffer_pretrain(10000, state_dims, action_dims)
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.batch_size = batch_size
        self.ckpt_dir = os.path.join(filedir, 'ckpt', name, '')
        self.M = 2.0

        self.build_network()
        self.params = tf.trainable_variables(scope=self.name)
        self.saver = tf.train.Saver(self.params, max_to_keep=500)

        self.unnormalized_actor_gradients = tf.gradients(
            self.mu, self.params, -self.action_gradient)

        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size),
                                        self.unnormalized_actor_gradients))

        self.optimize = tf.train.AdamOptimizer(self.lr).\
                    apply_gradients(zip(self.actor_gradients, self.params))

        self.pretrain_grad = tf.gradients(self.pretrain_loss, self.params)
        self.opt_pretrain = tf.train.AdamOptimizer(self.lr_pretrain).\
            apply_gradients(zip(self.pretrain_grad, self.params))

        self.initializer = tf.initialize_variables(self.params)

    def build_network(self):
        with tf.variable_scope(self.name):
            self.state = tf.placeholder(tf.float32,
                                        shape=[None, *self.state_dims],
                                        name='states')

            self.action_gradient = tf.placeholder(tf.float32,
                                          shape=[None, self.action_dims],
                                          name='gradients')
            
            net0 = tf.layers.batch_normalization(self.state)

            f1 = 1. / np.sqrt(self.fc1_dims)
            net1 = tf.layers.dense(net0, units=self.fc1_dims,
                            kernel_initializer=tf.glorot_normal_initializer(),
                            bias_initializer=tf.glorot_normal_initializer(),            
                                    #  kernel_initializer=random_uniform(-f1, f1),
                                    #  bias_initializer=random_uniform(-f1, f1),
                            #    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                     )
            net1 = tf.layers.batch_normalization(net1)
            net1 = tf.nn.relu(net1)

            f2 = 1. / np.sqrt(self.fc2_dims)
            net2 = tf.layers.dense(net1, units=self.fc2_dims,
                            kernel_initializer=tf.glorot_normal_initializer(),
                            bias_initializer=tf.glorot_normal_initializer(),  
                            #    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                    #  kernel_initializer=random_uniform(-f2, f2),
                                    #  bias_initializer=random_uniform(-f2, f2),
                                     )
            net2 = tf.layers.batch_normalization(net2)
            net2 = tf.nn.relu(net2)

            f3 = 0.003
            self.mu = tf.layers.dense(net2, units=self.action_dims,
                            activation='relu',
                            kernel_initializer=tf.glorot_normal_initializer(),
                            bias_initializer=tf.glorot_normal_initializer(),
                            kernel_regularizer=tf.keras.regularizers.l2(0.01),
                            # kernel_initializer= random_uniform(-f3, f3),
                            # bias_initializer=random_uniform(-f3, f3)
                            )
            self.mu = tf.div(self.M - tf.nn.relu(self.M - self.mu), self.M)

            self.mu_pretrain_target = tf.placeholder(tf.float32,
                                            shape=[None, 1],
                                            name='mu_target')
            self.pretrain_loss = tf.losses.mean_squared_error(self.mu, self.mu_pretrain_target)
            # self.pretrain_grad = tf.gradients(self.pretrain_loss, self.params)
            # self.opt_pretrain = tf.train.AdamOptimizer(self.lr_pretrain).apply(self.pretrain_grad)

    def pretrain(self, states, mu_pretrain_target):
        loss, _ = self.sess.run(
            fetches=[self.pretrain_loss, self.opt_pretrain],
            feed_dict={self.state: states,
                       self.mu_pretrain_target: mu_pretrain_target}
                       )
        return loss

    def predict(self, states):
        return self.sess.run(self.mu, feed_dict={self.state: states})

    def train(self, states, gradients):
        self.sess.run(self.optimize,
                      feed_dict={self.state: states,
                                 self.action_gradient: gradients})

    def load_checkpoint(self, idx=0, path=None):
        # print("...Loading checkpoint...")
        path = self.ckpt_dir if path is None else path
        ckptfile = os.path.join(path, str(idx)+'.ckpt')
        self.saver.restore(self.sess, ckptfile)

    def save_checkpoint(self, idx=0, path=None):
        # print("...Saving checkpoint...")
        path = self.ckpt_dir if path is None else path
        ckptfile = os.path.join(path, str(idx)+'.ckpt')
        self.saver.save(self.sess, ckptfile)


class Critic(object):
    def __init__(self, name, state_dims, action_dims,
                 sess, lr, fc1_dims, fc2_dims, batch_size=64, lr_pretrain=0.00001,
                 filedir=os.path.join(os.getcwd())):
        self.name = name   
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.sess = sess
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.batch_size = batch_size
        self.ckpt_dir = os.path.join(filedir, 'ckpt', name, '')

        self.build_network()
        self.params = tf.trainable_variables(scope=self.name)
        self.saver = tf.train.Saver(self.params, max_to_keep=500)

        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.action_gradients = tf.gradients(self.q, self.actions)

    def build_network(self):
        with tf.variable_scope(self.name):
            self.state = tf.placeholder(tf.float32,
                                        shape=[None, *self.state_dims],
                                        name='states')

            self.actions = tf.placeholder(tf.float32,
                                          shape=[None, self.action_dims],
                                          name='actions')

            self.q_target = tf.placeholder(tf.float32,
                                           shape=[None,1],
                                           name='targets')
            # action_in = tf.concat([self.state, self.actions], 1)
            net0 = tf.layers.batch_normalization(self.state)
    
            f1 = 1. / np.sqrt(self.fc1_dims)
            net1 = tf.layers.dense(net0, units=self.fc1_dims,
                            kernel_initializer=tf.glorot_normal_initializer(),
                            bias_initializer=tf.glorot_normal_initializer(),
                            #    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                    #  kernel_initializer=random_uniform(-f1, f1),
                                    #  bias_initializer=random_uniform(-f1, f1),
                                     )
            net1 = tf.layers.batch_normalization(net1)
            # batch1 = tf.identity(dense1)
            net1 = tf.nn.relu(net1)

            f2 = 1. / np.sqrt(self.fc2_dims)
            net2 = tf.layers.dense(net1, units=self.fc2_dims,
                            kernel_initializer=tf.glorot_normal_initializer(),
                            bias_initializer=tf.glorot_normal_initializer(),
                            #    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                    #  kernel_initializer=random_uniform(-f2, f2),
                                    #  bias_initializer=random_uniform(-f2, f2),
                                     )
            net2 = tf.layers.batch_normalization(net2)

            action_in = tf.layers.batch_normalization(self.actions)
            action_in = tf.layers.dense(action_in, units=self.fc2_dims,
                            kernel_initializer=tf.glorot_normal_initializer(),
                            bias_initializer=tf.glorot_normal_initializer(),
                            #    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                    #  kernel_initializer=random_uniform(-f2, f2),
                                    #  bias_initializer=random_uniform(-f2, f2),
                                    # activation='relu',
                                        )
            action_in = tf.layers.batch_normalization(action_in)

            state_actions = tf.add(net2, action_in)
            state_actions = tf.nn.relu(state_actions)
            # state_actions = tf.concat(0, [batch2, action_in])

            f3 = 0.003
            self.q = tf.layers.dense(state_actions, units=1,
                            kernel_initializer=tf.glorot_normal_initializer(),
                            bias_initializer=tf.glorot_normal_initializer(),
                            #    kernel_initializer=random_uniform(-f3, f3),
                            #    bias_initializer=random_uniform(-f3, f3),
                               kernel_regularizer=tf.keras.regularizers.l2(0.01),
                               )

            self.loss = tf.losses.mean_squared_error(self.q_target, self.q)

    def predict(self, states, actions):
        return self.sess.run(self.q,
                             feed_dict={self.state: states,
                                        self.actions: actions})

    def train(self, states, actions, q_target):
        loss, _ = self.sess.run(
            fetches=[self.loss, self.optimize],
            feed_dict={self.state: states,
                       self.actions: actions,
                       self.q_target: q_target})
        return loss

    def get_action_gradients(self, states, actions):
        return self.sess.run(self.action_gradients,
                             feed_dict={self.state: states,
                                        self.actions: actions})

    def load_checkpoint(self, idx=0, path=None):
        # print("...Loading checkpoint...")
        path = self.ckpt_dir if path is None else path
        ckptfile = os.path.join(path, str(idx)+'.ckpt')
        self.saver.restore(self.sess, ckptfile)

    def save_checkpoint(self, idx=0, path=None):
        # print("...Saving checkpoint...")
        path = self.ckpt_dir if path is None else path
        ckptfile = os.path.join(path, str(idx)+'.ckpt')
        self.saver.save(self.sess, ckptfile)


class Agent(object):
    def __init__(self, sess, state_dims, action_dims=1,
                 lr_actor=0.00005, lr_critic=0.0005, layer1_size=400, layer2_size=300,
                 batch_size=64, tau=0.001, gamma=0.99, max_size=1000000,
                 filedir=os.path.join(os.getcwd())):
        self.sess = sess
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma  # discount factor
        self.max_size = max_size
        self.filedir = filedir
        self.reset_memory()

        self.actor = Actor('Actor', state_dims, action_dims, 
                           self.sess, lr_actor, layer1_size, layer2_size, 
                           self.batch_size, filedir=filedir)
        self.critic = Critic('Critic', state_dims, action_dims,
                             self.sess, lr_critic, layer1_size, layer2_size, 
                             self.batch_size, filedir=filedir)

        self.target_actor = Actor('TargetActor', state_dims, action_dims, 
                                  self.sess, lr_actor, layer1_size, layer2_size, 
                                  self.batch_size, filedir=filedir)
        self.target_critic = Critic('TargetCritic', state_dims, action_dims,
                                    self.sess, lr_critic, layer1_size, layer2_size, 
                                    self.batch_size, filedir=filedir)

        self.noise = OUActionNoise(mu=np.zeros(action_dims), sigma=0.05)

        # define ops here in __init__ otherwise time to execute the op
        # increases with each execution.
        self.update_target_critic = \
            [self.target_critic.params[i].assign(
                      tf.multiply(self.critic.params[i], self.tau) \
                    + tf.multiply(self.target_critic.params[i], 1. - self.tau))
            for i in range(len(self.target_critic.params))]

        self.update_target_actor = \
            [self.target_actor.params[i].assign(
                      tf.multiply(self.actor.params[i], self.tau) \
                    + tf.multiply(self.target_actor.params[i], 1. - self.tau))
            for i in range(len(self.target_actor.params))]

        self.sess.run(tf.global_variables_initializer())

        self.update_target_parameters(first=True)

    def update_target_parameters(self, first=False):
        if first:
            old_tau = self.tau
            self.tau = 1.0
            self.target_critic.sess.run(self.update_target_critic)
            self.target_actor.sess.run(self.update_target_actor)
            self.tau = old_tau
        else:
            self.target_critic.sess.run(self.update_target_critic)
            self.target_actor.sess.run(self.update_target_actor)
    
    def update_network_parameters(self, actor_params=None, critic_params=None):
        if actor_params is not None:
            self.sess.run([self.actor.params[i].assign(actor_params[i])
                           for i in range(len(actor_params))])
        if critic_params is not None:
            self.sess.run([self.critic.params[i].assign(critic_params[i])
                           for i in range(len(critic_params))])

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
    
    def reset_memory(self):
        self.memory = ReplayBuffer_ddpg(self.max_size, self.state_dims, self.action_dims)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        # state = np.atleast_2d(state)
        mu = self.actor.predict(state) # returns list of list
        noise = self.noise()
        # noise = np.random.uniform(low=-0.03, high=0.03) if np.random.uniform() < 0.9 else np.random.uniform()
        # noise = 0.0
        # mu_prime = np.clip(mu+noise, 1.0e-6, 1.0-1.0e-6)
        mu_prime = mu + noise

        return mu_prime[0]

    def pretrain_actor(self, env, 
                 batch_size=1000, sample_size=64, load=True,
                 max_iter=100, epoch=100, tol=1e-4):
        print('Pretraining Actor network...')
        path = os.path.join('/', *self.filedir.split('/')[:-2])
        if load:
            ckptfile = os.path.join(path, '887.ckpt.meta')
            if os.path.exists(ckptfile):
                self.actor.load_checkpoint(887, path=path)
                print('Pretrained model loaded.')
                return
            print('Pretrained model not found. Prepare pretraining...')
        self.prememory = ReplayBuffer_pretrain(10000,
            self.state_dims, self.action_dims)
        loss_history = []
        for i in range(max_iter):
            states = env.states_generator(batch_size)
            actions = env.actions_generator(states)
            self.prememory.store(states, actions)
            for j in range(epoch):
                states_, actions_ = self.prememory.sample_buffer(sample_size)
                loss = self.actor.pretrain(states_, actions_)
                loss_history.append(loss)
            mean_loss = np.mean(loss_history[-sample_size:])
            print('Round = {}, pretrain mean loss = {:.6f}'.format(i, mean_loss))
            if mean_loss < tol:
                print('Pretrain finished! Mean loss < {} after {} rounds'.format(tol, i))
                break
            if i == max_iter - 1:
                print('Max number of {} rounds reached! '
                    'Last mean loss = {:.6f}. Exit pretrain.'.format(max_iter, mean_loss))

        np.save(self.filedir+'pretrain_actor_loss.npy', np.array(loss_history))
        self.actor.save_checkpoint(idx=887, path=path)
        self.update_target_parameters(first=True)
    
    def pretrain_critic(self, env, 
                 batch_size=1000, sample_size=64, load=True,
                 max_iter=100, epoch=100, tol=1e-4):
        print('Pretraining Critic network...')
        path = os.path.join('/', *self.filedir.split('/')[:-2])
        if load:
            ckptfile = os.path.join(path, '997.ckpt.meta')
            if os.path.exists(ckptfile):
                self.critic.load_checkpoint(997, path=path)
                print('Pretrained model loaded.')
                return
            print('Pretrained model not found. Prepare pretraining...')

        loss_history = []
        for i in range(max_iter):
            for _ in range(batch_size):
                obs = env.states_generator(batch_size=1)[0]
                # act = self.choose_action(obs)
                act = env.actions_generator(obs)
                new_state, reward, done = env.step(obs, act, print_ctr=10000)
                self.remember(obs, act, reward, new_state, int(done))
            for _ in range(epoch):
                loss = self.learn(behave=False, batch_size=sample_size)
                loss_history.append(loss)

            mean_loss = np.mean(loss_history[-sample_size:])
            print('Round = {}, pretrain mean loss = {:.6f}'.format(i, mean_loss))
            if mean_loss < tol:
                print('Pretrain finished! Mean loss < {} after {} rounds'.format(tol, i))
                break
            if i == max_iter - 1:
                print('Max number of {} rounds reached! '
                    'Last mean loss = {:.6f}. Exit pretrain.'.format(max_iter, mean_loss))

        np.save(self.filedir+'pretrain_critic_loss.npy', np.array(loss_history))
        self.critic.save_checkpoint(idx=997, path=path)
        self.update_target_parameters(first=True)
        self.reset_memory()

    def learn(self, batch_size=None, behave=True):
        batch_size = self.batch_size if batch_size is None else batch_size
        if self.memory.mem_cntr < batch_size:
            return
        # if self.memory.mem_cntr < 10000:
        #     return
        state, action, reward, new_state, done = \
                                      self.memory.sample_buffer(batch_size)
        # if state is None:
        #     return

        critic_value_ = self.target_critic.predict(new_state,
                                           self.target_actor.predict(new_state))
        target = []
        for j in range(batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j] *done[j])
        target = np.reshape(target, (batch_size, 1))

        loss = self.critic.train(state, action, target)

        if behave:
            a_outs = self.actor.predict(state)
            grads = self.critic.get_action_gradients(state, a_outs)
            self.actor.train(state, grads[0])
            self.update_target_parameters()
        return loss

    def save_models(self):
        print('saving models...')
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        print('loading models...')
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()


class ReplayBuffer_fp(object):
    def __init__(self, max_size, K_flow_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.K_flow_memory = np.zeros((self.mem_size, *K_flow_shape))

    def store_K_flow(self, K_flow):
        index = self.mem_cntr % self.mem_size
        self.K_flow_memory[index] = K_flow
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        K_flows = self.K_flow_memory[batch]

        return K_flows




def create_transmat(transmat_pm):
        ug, ub = transmat_pm['ug'], transmat_pm['ub']
        zg_avg_dur, zb_avg_dur, ug_avg_dur, ub_avg_dur = (
            transmat_pm['zg_avg_dur'], transmat_pm['zb_avg_dur'],
            transmat_pm['ug_avg_dur'], transmat_pm['ub_avg_dur'])
        puu_rel_gb2bb, puu_rel_bg2gg = (
            transmat_pm['puu_rel_gb2bb'], transmat_pm['puu_rel_bg2gg'])

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

        transmat = dict((
            ('z_eps', P),
            ('z', Pz),
            ((0, 0), Peps_bb),
            ((0, 1), Peps_bg),
            ((1, 0), Peps_gb),
            ((1, 1), Peps_gg),
        ))

        return transmat


class MFG(object):
    def __init__(self, ksp, n_steps, net_pm, K_0, B=None,
                filedir=os.path.join(os.getcwd(), 'test/')):
        self.ksp = ksp
        self.n_steps = n_steps
        B = B if B is not None else [0.0, 1.0, 0.0, 1.0]
        self.B_list = [B,]
        self.B_round_list = [B,]
        self.R2_list = [[0.0, 0.0],]
        self.R2_round_list = [[0.0, 0.0],]
        self.filedir = filedir
        self.sess = tf.Session()
        self.env = Environment(self.ksp, self.sess, K_0, B, self.n_steps)
        self.agent = Agent(self.sess,
            net_pm['state_dims'], net_pm['action_dims'],
            net_pm['lr_actor'], net_pm['lr_critic'], 
            net_pm['layer1_size'], net_pm['layer2_size'],
            net_pm['batch_size'], net_pm['tau'],
            self.ksp['beta'], net_pm['max_size'],
            filedir=filedir)
        
        self.round_ctr = 0
        self.score_history = []
        self.K_path_history = [self.env.K_path,]
        self.K_path_round_history = [self.env.K_path,]
    
    def pretrain(self):
        print('pretraining agent...')
        time_start = time.time()
        self.agent.pretrain_actor(self.env)
        self.agent.pretrain_critic(self.env)
        print('pretraining finished! time = {}'.format(time.time()-time_start))

    def play(self, max_step=None, print_ctr=30):
        env, agent = self.env, self.agent
        obs = env.reset()[0]
        done = False
        score = 0
        beta = self.ksp['beta']
        discount = 1.0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done = env.step(obs, act, max_step, print_ctr)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += discount * reward
            obs = new_state
            discount *= beta                
        self.score_history.append(score)
        return score
    
    def solve_policy(self, n_life=20, n_round=1, max_step=None, tol=1e-3,
            simu_pop=5000, simu_round=100, n_life_0=None, print_ctr=30):
        max_step = self.n_steps if max_step is None else max_step
        n_life_0 = n_life if n_life_0 is None else n_life_0
        n_life = (n_life_0,) + (n_life,) * (n_round-1)
        time_history = [time.time(),]
        print('Start to solve for equilibrium policy...')
        for t in range(n_round):
            self.round_ctr += 1
            time_round = time.time()
            print('running round %d ...'% self.round_ctr)
            # self.sess.run(tf.global_variables_initializer())
            self.agent.reset_memory()
            for i in range(n_life[t]):
                time_life = time.time()
                score = self.play(max_step=max_step, print_ctr=print_ctr)
                print('round', t, 'life', i, 'score %.2f' % score,
                      'trailing 100 games avg %.4f' % np.mean(self.score_history[-100:]),
                      'time = %.4f' % (time.time()-time_life))
            self.agent.actor.save_checkpoint(idx=self.round_ctr)

            B_round, R2_round = self.regress_ALM(n_pop=simu_pop, n_round=simu_round, random=False)
            self.B_round_list.append(B_round)
            self.R2_round_list.append(R2_round)
            K_path_round = self.simulate_ALM(B=B_round)
            self.K_path_round_history.append(K_path_round)

            B, R2 = self.regress_ALM(n_pop=simu_pop, n_round=simu_round, random=True)
            self.B_list.append(B)
            self.R2_list.append(R2)
            K_path = self.simulate_ALM(B=B)
            self.K_path_history.append(K_path)

            self.set_ALM(B)
            time_history.append(time.time())
            print('Round = {:d} finished! Time = {:.4f}'.format(
                self.round_ctr, time_history[-1]-time_history[-2]))

            if t > 0:
                B_old, B_new = np.array(self.B_list[-2]), np.array(B)
                diff = np.max(np.abs(B_old - B_new))
                if diff < tol and diff > 1e-7:
                    print('Convergence of ALM coeffients! Exit training.')
                    break
                elif t == n_round-1:
                    print('Maximum number of {} rounds reached! '
                        'Last difference of ALM coefficients is {}'.format(
                        n_round, diff))

        print('Total time spent = {}'.format(time_history[-1]-time_history[0]))
        K_path_history = np.array(self.K_path_history)
        K_path_round_history = np.array(self.K_path_round_history)
        np.save(self.filedir + 'K_path_history.npy', K_path_history)
        np.save(self.filedir + 'K_path_round_history.npy', K_path_round_history)
        np.save(self.filedir + 'score_history.npy', self.score_history)
        np.save(self.filedir + 'time_history.npy', np.array(time_history))
        np.savez(self.filedir + 'reg_stat.npz',
            B=np.array(self.B_list),
            B_round=np.array(self.B_round_list),
            R2=np.array(self.R2_list),
            R2_round=np.array(self.R2_round_list),
            )

    def randomize_policy(self, state):
        actor = self.agent.actor
        actor.save_checkpoint(idx=10000)
        action = []
        for i in np.arange(1, self.round_ctr+1):
            actor.load_checkpoint(idx=i)
            action.append(actor.predict(state))
        actor.load_checkpoint(idx=10000)
        return np.mean(np.concatenate(action, axis=1), axis=1, keepdims=True)
    
    def simulate_dist(self, zi_path=None, epsi_path=None, 
            n_pop=100, n_steps=None, random=False, discard=0, save=False):
        n_steps = self.n_steps if n_steps is None else n_steps
        compute_action = (lambda s: self.randomize_policy(s)) if random \
            else (lambda s: self.agent.actor.predict(s))

        if zi_path is None:
            zi_path, epsi_path = self.env.generate_shock(
                n_steps=n_steps, n_pop=n_pop)
        elif epsi_path is None:
            _, epsi_path = self.env.generate_shock(
                zi_path=zi_path, n_steps=n_steps, n_pop=n_pop)
        z_range, eps_range = self.ksp['z_range'], self.ksp['eps_range']
        z_path = np.array([z_range[zi] for zi in zi_path])
        eps_path = np.array([eps_range[epsi] for epsi in np.nditer(epsi_path)])
        eps_path = eps_path.reshape((n_pop, n_steps))

        k_path = np.zeros((n_pop, n_steps))
        K_path = np.zeros((n_steps,))
        k_path[:, 0] = self.env.K_0
        K_path[0] = self.env.K_0

        c_path = np.zeros_like(k_path)
        w_path = np.zeros_like(k_path)
        r_path = np.zeros_like(k_path)
        u_path = np.zeros_like(k_path)

        print('simulating capital path using {} policy...'.format(
            'random' if random else 'contigent'), end=' ')
        time_sim = time.time()
        for t in range(n_steps-1):
            k_vec = k_path[:, t]
            K = K_path[t]
            zi, z = zi_path[t], z_path[t]
            epsi_vec, eps_vec = epsi_path[:, t], eps_path[:, t]
            L = self.ksp['l_bar'] * (1.0 - \
                (zi * self.ksp['ug'] + (1.0-zi) * self.ksp['ub']))
            state = np.array([k_vec, K*np.ones(n_pop), zi*np.ones(n_pop), epsi_vec]).T
            action = compute_action(state)
            # wealth = self.env.compute_wealth(k_vec, K, z, eps_vec, L)
            # kp_vec = wealth * action[:, 0]
            kp_vec = k_vec * action[:, 0]
            k_path[:, t+1] = kp_vec
            K_path[t+1] = np.mean(kp_vec)

            wealth_vec = self.env.compute_wealth(k_vec, K, z, eps_vec, L)
            consumption_vec = wealth_vec - kp_vec
            w_path[:, t] = wealth_vec
            c_path[:, t] = consumption_vec
            u_path[:, t] = self.ksp['u'](consumption_vec)
        discount = np.power(self.ksp['beta'], np.arange(n_steps))
        u_vec = np.sum(u_path * discount, axis=1)
        print('Finish! time = {:.4f}'.format(time.time()-time_sim))
        
        # k_path = k_path[:, discard:]
        # K_path = K_path[discard:]
        # zi_path = zi_path[discard:]
        # epsi_path = epsi_path[:, discard:]
        paths = dict((
            ('k', k_path),
            ('K', K_path),
            ('zi', zi_path),
            ('epsi', epsi_path),
            ('w', w_path),
            ('c', c_path),
            ('u', u_path),
            ('utility', u_vec),
        ))

        if save:
            np.savez(self.filedir+'paths.npz', 
                k=k_path, K=K_path, zi=zi_path, epsi=epsi_path, w=w_path,
                c=c_path, u=u_path, utility=u_vec)
        
        return paths
    
    def regress_ALM(self, n_pop=100, n_round=10, random=False):
        time_start = time.time()
        print('Finding ALM coefficients...{} rounds will run...'.format(n_round))
        K_b, K_g, Kp_b, Kp_g = np.array([]), np.array([]), np.array([]), np.array([])
        for i in range(n_round):
            print('round = {:03d}'.format(i), end=' ')
            paths = self.simulate_dist(n_pop=n_pop, random=random)
            K_path, zi_path = paths['K'], paths['zi']
            bad_times = zi_path[:-1] == 0
            good_times = zi_path[:-1] == 1
            K_b_, K_g_ = K_path[:-1][bad_times], K_path[:-1][good_times]
            Kp_b_, Kp_g_ = K_path[1:][bad_times], K_path[1:][good_times]
            K_b = np.append(K_b, K_b_)
            K_g = np.append(K_g, K_g_)
            Kp_b = np.append(Kp_b, Kp_b_)
            Kp_g = np.append(Kp_g, Kp_g_)

        n_b, n_g = len(K_b), len(K_g)
        K_b, K_g = np.log(K_b), np.log(K_g)
        Kp_b, Kp_g = np.log(Kp_b), np.log(Kp_g)
        X_b = np.array([np.ones(n_b), K_b]).T
        X_g = np.array([np.ones(n_g), K_g]).T

        try:
            res_b = sm.OLS(Kp_b, X_b).fit()
            B0, B1 = res_b.params
            R2_b = res_b.rsquared
        except Exception:
            B0, B1 = self.env.B[0], self.env.B[1]
            R2_b = 0.0
        
        try:
            res_g = sm.OLS(Kp_g, X_g).fit()
            B2, B3 = res_g.params
            R2_g = res_g.rsquared
        except Exception:
            B2, B3 = self.env.B[2], self.env.B[3]
            R2_g = 0.0

        B = [B0, B1, B2, B3]
        R2 = [R2_b, R2_g]
        print('ALM found! time = {}'.format(time.time()-time_start))
        return B, R2
    
    def simulate_ALM(self, n_steps=None, B=None, zi_path=None):
        B = B if B is not None else self.B_list[-1]
        B0, B1, B2, B3 = B
        n_steps = self.n_steps if n_steps is None else n_steps
        if zi_path is None:
            zi_path, _ = self.env.generate_shock(n_steps=n_steps)
        K_path = np.zeros((n_steps,))
        K_path[0] = self.env.K_0
        for t in range(n_steps-1):
            B_cons = B0 if zi_path[t] == 0 else B2
            B_slope = B1 if zi_path[t] == 0 else B3
            K_path[t+1] = np.exp(B_cons + B_slope * np.log(K_path[t]))
        return K_path

    def set_ALM(self, B):
        self.env.B = B
        return
    
    def get_equil_policy(self):
        return lambda s: self.randomize_policy(s)


class Environment(object):
    def __init__(self, ksp, sess, K_0=40, B=None, n_steps=100):
        
        self.ksp = ksp
        self.sess = sess
        self.K_0 = K_0
        self.B = B if B is not None else [0.0, 1.0, 0.0, 1.0]
        self.n_steps = n_steps

        # self.zi_path, self.epsi_path = self.generate_shock()
        self.initialize_K_path(self.B)

        self.step_ctr = 0
        self.update_ctr = 1

    def generate_shock(self, zi_path=None, n_steps=None, n_pop=1):
        n_steps = self.n_steps if n_steps is None else n_steps
        mc = self.ksp['mc']

        zi_path = mc['z'].draw_path(n_obs=n_steps-1) \
            if zi_path is None else zi_path
        epsi_path = []
        for i in range(n_pop):
            epsi_path_t = [mc[(0, 0)].draw(),]
            for t in range(n_steps-1):
                z_index = (zi_path[t], zi_path[t+1])
                epsi_path_t.append(mc[z_index].draw(epsi_path_t[-1]))
            epsi_path.append(epsi_path_t)
        epsi_path = np.array(epsi_path[0]) if n_pop == 1 else np.array(epsi_path)

        if n_pop >= 100:
            ub, ug = self.ksp['ub'], self.ksp['ug']
            ideal_um_b, ideal_um_g = int(ub*n_pop), int(ug*n_pop)
            for t in range(n_steps):
                actual_um = np.sum(epsi_path[:, t] == 0)
                ideal_um = ideal_um_b if zi_path[t] == 0 else ideal_um_g
                gap = ideal_um - actual_um
                if gap > 0:
                    replace = np.random.choice(np.where(epsi_path[:, t] == 1)[0],
                        size=gap, replace=False)
                    epsi_path[replace, t] = 0
                elif gap < 0:
                    replace = np.random.choice(np.where(epsi_path[:, t] == 0)[0],
                        size=-gap, replace=False)
                    epsi_path[replace, t] = 1
        return zi_path, epsi_path

    def initialize_K_path(self, B=None):
        B = B if B is not None else self.B
        B0, B1, B2, B3 = B
        K_path = np.zeros((self.n_steps,))
        K_path[0] = self.K_0
        zi_path = self.ksp['mc']['z'].draw_path(n_obs=self.n_steps)
        for t in range(self.n_steps-1):
            B_cons = B0 if zi_path[t] == 0 else B2
            B_slope = B1 if zi_path[t] == 0 else B3
            K_path[t+1] = np.exp(B_cons + B_slope * np.log(K_path[t]))
        self.K_path = K_path
        return K_path
    
    def randomize_z(self, z_path):
        return z_path + np.random.normal(scale=0.05, size=z_path.shape)
    
    def randomize_eps(self, eps_path):
        return eps_path + np.random.uniform(high=0.3, size=eps_path.shape)

    # def reset(self):
    #     self.step_ctr = 0
    #     z_range, eps_range = self.ksp['z_range'], self.ksp['eps_range']
    #     zi_path, epsi_path = self.generate_shock(n_pop=1)
    #     z_path = np.array([z_range[zi] for zi in zi_path])
    #     eps_path = np.array([eps_range[epsi] for epsi in epsi_path])
    #     # z_path_rdm = self.randomize_z(z_path)
    #     # eps_path_rdm = self.randomize_eps(eps_path)
    #     z_path_rdm, eps_path_rdm = zi_path, epsi_path

    #     self.zi_path, self.z_path, self.z_path_rdm = \
    #         zi_path, z_path, z_path_rdm
    #     self.epsi_path, self.eps_path, self.eps_path_rdm = \
    #         epsi_path, eps_path, eps_path_rdm
    #     self.L_path = self.ksp['l_bar'] * (1.0 - 
    #         (zi_path*self.ksp['ug'] + (1.0-zi_path)*self.ksp['ub']))

    #     z_rdm, eps_rdm = z_path_rdm[0], eps_path_rdm[0]

    #     # k, K = self.K_0, self.K_0
    #     k, K = np.random.uniform(low=1e-3, high=2.0*self.K_0, size=(2,))
    #     # k, K = np.random.uniform(size=(2,))

    #     return np.array((k, K, z_rdm, eps_rdm))
    
    def reset(self):
        self.step_ctr = 0
        states = self.states_generator(batch_size=1)
        return states
    
    def step(self, state, action, max_step=None, print_ctr=30):
        self.step_ctr += 1
        ctr = self.step_ctr
        max_step = self.n_steps if max_step is None else max_step

        k, K, zi, epsi = state
        zi, epsi = int(zi), int(epsi)
        zi_epsi = zi + epsi
        zpi_epspi = self.ksp['mc']['z_eps'].draw(zi_epsi)
        zpi, epspi = zpi_epspi // 2, zpi_epspi % 2
        kp = k * action
        
        B0, B1, B2, B3 = self.B
        B_con = B0 if zi == 0 else B2
        B_slope = B1 if zi == 0 else B3
        Kp = np.exp(B_con + B_slope * np.log(K))

        statep = np.array((kp, Kp, zpi, epspi))

        L = self.ksp['l_bar'] * (1.0 - 
            (zi*self.ksp['ug'] + (1.0-zi)*self.ksp['ub']))
        alpha = self.ksp['alpha']
        z, eps = self.ksp['z_range'][zi], self.ksp['eps_range'][epsi]
        intr = self.compute_intr(alpha, z, K, L)
        wage = self.compute_wage(alpha, z, K, L)
        wealth = self.compute_wealth(k, K, z, eps, L)

        if kp >= wealth or kp <= 0.0:
            reward = -20.0
            done = True
        else:
            con, reward = self.compute_reward(wealth, kp)
            # if self.step_ctr == max_step:
            #     reward = reward / (1.0 - self.ksp['beta'])
            done = (self.step_ctr == max_step or kp <= 1.0e-4 or con <= 1.0e-4)

        
        k_ratio = kp / wealth
        if ctr % print_ctr == 0:
            print('step={}, k={}, K={:.6f}, L={:.4f}, intr={:.4f}, wage={:.4f}, wealth={}, action={}, k_ratio={}, reward={}'.format(
                ctr, k, K, L, intr, wage, wealth, action, k_ratio, reward))

        return [statep, reward, done]

    # def step(self, state, action, print_ctr=30):
    #     self.step_ctr += 1
    #     ctr = self.step_ctr
    #     k, K, z_rdm, eps_rdm = state
    #     z_rdm, eps_rdm = self.z_path[ctr-1], self.eps_path[ctr-1]
    #     L = self.L_path[ctr-1]
    #     alpha = self.ksp['alpha']

    #     intr = self.compute_intr(alpha, z_rdm, K, L)
    #     wage = self.compute_wage(alpha, z_rdm, K, L)
    #     wealth = self.compute_wealth(k, K, z_rdm, eps_rdm, L)
    #     # k_ratio = 0.5 + 0.4*action
    #     # kp = wealth * k_ratio
    #     # kp = k * action
    #     kp = k * action
    #     k_ratio = kp / wealth
    #     Kp, zp, epsp = self.K_path[ctr] / self.K_0, \
    #         self.z_path_rdm[ctr], self.eps_path_rdm[ctr]
        
    #     B0, B1, B2, B3 = self.B
    #     B_con = B0 if self.zi_path[ctr-1] == 0 else B2
    #     B_slope = B1 if self.zi_path[ctr-1] == 0 else B3
    #     Kp = np.exp(B_con + B_slope * np.log(K))

    #     statep = np.array((kp, Kp, zp, epsp))

    #     if kp >= wealth or kp <= 0.0:
    #         return [statep, -20.0, True]

    #     con, reward = self.compute_reward(wealth, kp)

    #     if ctr % print_ctr == 0:
    #         print('step={}, k={}, K={:.6f}, L={:.4f}, intr={:.4f}, wage={:.4f}, wealth={}, action={}, k_ratio={}, reward={}'.format(
    #             ctr, k, K, L, intr, wage, wealth, action, k_ratio, reward))

    #     done = (ctr == self.n_steps-1 or kp <= 1.0e-4 or con <= 1.0e-4)
    #     return [statep, reward, done]
    
    def compute_wealth(self, k, K, z, eps, L):
        alpha, delta = self.ksp['alpha'], self.ksp['delta']
        l_bar, mu = self.ksp['l_bar'], self.ksp['mu']

        intr = self.compute_intr(alpha, z, K, L)
        wage = self.compute_wage(alpha, z, K, L)
        wealth = (intr + 1 - delta) * k + wage * (eps*l_bar + (1.0-eps)*mu)
        return wealth

    def compute_reward(self, wealth, kp):
        con = wealth - kp
        reward = self.ksp['u'](con) if con > 0  else -10
        return con, reward

    def compute_intr(self, alpha, z, K, L):
        return alpha * z * (L / np.maximum(K, 1.0e-6))**(1.0 - alpha)

    def compute_wage(self, alpha, z, K, L):
        return (1.0 - alpha) * z * (K / L)**alpha
        
    def update_K_path(self, K_path_new, update_ctr=None):
        self.update_ctr += 1
        update_ctr = self.update_ctr if update_ctr is None else update_ctr
        self.K_path = 1.0/update_ctr * K_path_new + \
            (1.0-1.0/update_ctr) * self.K_path
        return self.K_path

    def simulate_capital_path(self, agent, zi_path, epsi_path):
        n_pop, n_steps = epsi_path.shape[0], epsi_path.shape[1]
        k_flow = np.zeros((n_pop, n_steps))
        K_flow_sml = np.zeros((n_steps,))
        k_ratio = np.zeros((n_pop, n_steps))
        k_flow[:, 0] = self.K_0
        K_flow_sml[0] = self.K_0
        ub, ug = self.transmat_pm['ub'], self.transmat_pm['ug']
        alpha, delta, l_bar, mu, z_range, eps_range = self.alpha, self.delta, self.l_bar, self.mu, self.z_range, self.eps_range
        K_0 = self.K_0

        for t in range(n_steps-1):
            K, z_ind = K_flow_sml[t], zi_path[t]
            z = z_range[z_ind]
            L = l_bar * (1.0 - (ub if z_ind == z_range[0] else ug))
            intr = alpha * z * (L / np.max((K, 1.0e-6))) ** (1.0 - alpha)
            wage = (1.0 - alpha) * z * (K / L) ** alpha
            for i in range(n_pop):
                k, eps_ind = k_flow[i, t], epsi_path[i, t]
                eps = eps_range[eps_ind]
                state = np.array((k, K, z_ind, eps_ind))[np.newaxis, :]
                k_ratio[i, t+1] = agent.actor.sess.run(
                    agent.actor.mu, feed_dict={agent.actor.state: state})

                wealth = (intr + 1.0 - delta) * k + \
                    wage * (eps * l_bar + (1.0-eps) * mu)
                # k_flow[i, t+1] = wealth * k_ratio[i, t+1]
                k_flow[i, t+1] = k * k_ratio[i, t+1]

            K_flow_sml[t+1] = np.mean(k_flow[:, t+1])
        k_flow = k_flow[:, -self.n_steps:]
        K_flow_sml = K_flow_sml[-self.n_steps:]

        return k_flow, K_flow_sml
    
    def states_generator(self, batch_size=1):
        k_K = np.random.uniform(low=1e-3, high=2.0*40.0, size=(batch_size, 2))
        z_eps = np.random.randint(low=0, high=2, size=(batch_size, 2))
        states = np.concatenate([k_K, z_eps], axis=1)
        return states

    def actions_generator(self, states):
        action = 0.7
        batch_size = np.shape(np.atleast_2d(states))[0]
        return action * np.ones((batch_size,1))

    