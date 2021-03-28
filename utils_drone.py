import itertools
import logging
import numpy as np
import time
import os
import pandas as pd
import subprocess


def check_dir(cur_dir):
    if not os.path.exists(cur_dir):
        return False
    return True


def copy_file(src_dir, tar_dir):
    cmd = 'sudo cp %s %s' % (src_dir, tar_dir)
    subprocess.check_call(cmd, shell=True)


def find_file(cur_dir, suffix='.ini'):
    for file in os.listdir(cur_dir):
        if file.endswith(suffix):
            return cur_dir + '/' + file
    logging.error('Cannot find %s file' % suffix)
    return None


def init_dir(base_dir, pathes=['log', 'data', 'model']):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    dirs = {}
    for path in pathes:
        cur_dir = base_dir + '/%s/' % path
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        dirs[path] = cur_dir
    return dirs


def init_log(log_dir):
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler('%s/%d.log' % (log_dir, time.time())),
                            logging.StreamHandler()
                        ])


def init_test_flag(test_mode):
    if test_mode == 'no_test':
        return False, False
    if test_mode == 'in_train_test':
        return True, False
    if test_mode == 'after_train_test':
        return False, True
    if test_mode == 'all_test':
        return True, True
    return False, False


class Counter:
    def __init__(self, total_step, test_step, log_step):
        self.counter = itertools.count(1)
        self.cur_step = 0
        self.cur_test_step = 0
        self.total_step = total_step
        self.test_step = test_step
        self.log_step = log_step
        self.stop = False

    def next(self):
        self.cur_step = next(self.counter)
        return self.cur_step

    def should_test(self):
        test = False
        if (self.cur_step - self.cur_test_step) >= self.test_step:
            test = True
            self.cur_test_step = self.cur_step
        return test

    def should_log(self):
        return (self.cur_step % self.log_step == 0)

    def should_stop(self):
        if self.cur_step >= self.total_step:
            return True
        return self.stop


class Trainer():
    def __init__(self, env, model, global_counter, output_path=None, model_path=None):
        self.cur_step = 0
        self.global_counter = global_counter
        self.env = env
        self.agent = self.env.agent
        self.model = model
        self.n_step = self.model.n_step
        # self.summary_writer = summary_writer
        # assert self.env.T % self.n_step == 0
        self.data = []
        self.output_path = output_path
        self.model_path = model_path
        self.env.train_mode = True

    # def _add_summary(self, reward, global_step, is_train=True):
    #     if is_train:
    #         self.summary_writer.add_scalar('train_reward', reward, global_step=global_step)
    #     else:
    #         self.summary_writer.add_scalar('test_reward', reward, global_step=global_step)

    def _get_policy(self, ob, done, mode='train'):
        if self.agent.startswith('ma2c'):
            self.ps = self.env.get_fingerprint()
            policy = self.model.forward(ob, done, self.ps)
        else:
            policy = self.model.forward(ob, done)
        action = []
        for pi in policy:
            if mode == 'train':
                action.append(np.random.choice(np.arange(len(pi)), p=pi))
            else:
                action.append(np.argmax(pi))
        return policy, np.array(action)

    def _get_value(self, ob, done, action):
        if self.agent.startswith('ma2c'):
            value = self.model.forward(ob, done, self.ps, np.array(action), 'v')
        else:
            self.naction = self.env.get_neighbor_action(action)
            if not self.naction:
                self.naction = np.nan
            value = self.model.forward(ob, done, self.naction, 'v')
        return value

    # def _log_episode(self, global_step, mean_reward, std_reward):
    #     log = {'agent': self.agent,
    #            'step': global_step,
    #            'test_id': -1,
    #            'avg_reward': mean_reward,
    #            'std_reward': std_reward}
    #     self.data.append(log)
    #     self._add_summary(mean_reward, global_step)
    #     self.summary_writer.flush()

    def perform(self, test_ind, gui=False):
        N_iter = 400
        is_centralized = False
        ob = self.env.reset()
        rewards = []
        # note this done is pre-decision to reset LSTM states!
        done = False
        # self.model.reset()
        for i in range(N_iter):
            if self.agent == 'greedy':
                action = self.model.forward(ob)
            else:
                # in on-policy learning, test policy has to be stochastic
                if self.env.name.startswith('atsc'):
                    policy, action = self._get_policy(ob, done)
                else:
                    # for mission-critic tasks like CACC, we need deterministic policy
                    policy, action = self._get_policy(ob, done, mode='test')
                self.env.update_fingerprint(policy)
            next_ob, reward, done, _ = self.env.step(action, i, is_centralized)
            rewards.append(reward)
            if done:
                break
            ob = next_ob
        mean_reward = np.mean(np.array(rewards))
        std_reward = np.std(np.array(rewards))
        return mean_reward, std_reward

    def explore(self, prev_ob, prev_done, episode):
        ob = prev_ob
        done = prev_done
        is_centralized = False
        N_iteration = 400
        for i in range(N_iteration):
            # pre-decision
            policy, action = self._get_policy(ob, done)
            # post-decision
            value = self._get_value(ob, done, action)
            # transition
            self.env.update_fingerprint(policy)
            next_ob, reward, done, _ = self.env.step(action, i, is_centralized)
            self.episode_rewards.append(reward)
            global_step = self.global_counter.next()
            self.cur_step += 1
            # collect experience
            if self.agent.startswith('ma2c'):
                self.model.add_transition(ob, self.ps, action, reward, value, done)
            else:
                self.model.add_transition(ob, self.naction, action, reward, value, done)
            # logging
            # if self.global_counter.should_log():
            #     logging.info('''Training: global step %d, episode step %d,
            #                        ob: %s, a: %s, pi: %s, r: %.2f, train r: %.2f, done: %r''' %
            #                  (global_step, self.cur_step,
            #                   str(ob), str(action), str(policy), reward, np.mean(reward), done))
            # terminal check must be inside batch loop for CACC env
            print ("Episode/Iteration: {0}/{1}, Actions: {2}, Rewards: {3}".format(episode, i, action, reward))
            if done:
                break
            ob = next_ob
        if done:
            R = np.zeros(self.model.n_agent)
        else:
            _, action = self._get_policy(ob, done)
            R = self._get_value(ob, done, action)
        return ob, done, R

    def run(self):
        N_episode = 100
        best_reward_train = -np.Inf
        best_reward_test = -np.Inf
        eval_period = 1
        # while not self.global_counter.should_stop():
        for episode in range(N_episode):
            # np.random.seed(self.env.seed)
            ob = self.env.reset()
            # note this done is pre-decision to reset LSTM states!
            done = True
            self.model.reset()
            self.cur_step = 0
            self.episode_rewards = []
            # while True:
            ob, done, R = self.explore(ob, done, episode)
            dt = N_episode - self.cur_step
            global_step = self.global_counter.cur_step
            self.model.backward(R, dt, global_step)
            rewards = np.array(self.episode_rewards)
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)

            if mean_reward > best_reward_train:
                print ("Better reward value in Train mode is obtained! The model is being saved!")
                best_reward_train = mean_reward 
                self.model.save(self.model_path, episode, "train")

            if episode % eval_period == 0:
                self.env.train_mode = False
                mean_reward, std_reward = self.perform(-1)
                self.env.train_mode = True
                if mean_reward > best_reward_test:
                    print ("Better reward value in Test mode is obtained! The model is being saved!")
                    best_reward_test = mean_reward
                    self.model.save(self.model_path, episode, "test")
                    

            # self._log_episode(global_step, mean_reward, std_reward)

        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')


class Tester(Trainer):
    def __init__(self, env, model, global_counter, output_path):
        super().__init__(env, model, global_counter)
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.data = []
        logging.info('Testing: total test num: %d' % self.test_num)

    def run_offline(self):
        # enable traffic measurments for offline test
        is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        rewards = []
        for test_ind in range(self.test_num):
            rewards.append(self.perform(test_ind))
            self.env.terminate()
            time.sleep(2)
            self.env.collect_tripinfo()
        avg_reward = np.mean(np.array(rewards))
        logging.info('Offline testing: avg R: %.2f' % avg_reward)
        self.env.output_data()

    def run_online(self, coord):
        self.env.cur_episode = 0
        while not coord.should_stop():
            time.sleep(30)
            if self.global_counter.should_test():
                rewards = []
                global_step = self.global_counter.cur_step
                for test_ind in range(self.test_num):
                    cur_reward = self.perform(test_ind)
                    self.env.terminate()
                    rewards.append(cur_reward)
                    log = {'agent': self.agent,
                           'step': global_step,
                           'test_id': test_ind,
                           'reward': cur_reward}
                    self.data.append(log)
                avg_reward = np.mean(np.array(rewards))
                self._add_summary(avg_reward, global_step)
                logging.info('Testing: global step %d, avg R: %.2f' %
                             (global_step, avg_reward))
                # self.global_counter.update_test(avg_reward)
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')


class Evaluator(Tester):
    def __init__(self, env, model, output_path, gui=False):
        self.env = env
        self.model = model
        self.agent = self.env.agent
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.gui = gui

    def run(self):
        if self.gui:
            is_record = False
        else:
            is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        time.sleep(1)
        for test_ind in range(self.test_num):
            reward, _ = self.perform(test_ind, gui=self.gui)
            self.env.terminate()
            logging.info('test %i, avg reward %.2f' % (test_ind, reward))
            time.sleep(2)
            self.env.collect_tripinfo()
        self.env.output_data()
