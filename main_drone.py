import sys
import time
import signal
import argparse
import os
import glob

import numpy as np
import torch
# import visdom
# import data
from models_drone import *
from comm_drone import CommNetMLP
from utils import *
from action_utils import parse_action_args
from trainer_drone import Trainer, Tester
from systematic_results import Reporter
from multi_processing import MultiProcessTrainer
from point_mass_formation import QuadrotorFormation

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch RL trainer')
# training
# note: number of steps per epoch = epoch_size X batch_size x nprocesses
parser.add_argument('--num_epochs', default=30000, type=int,
                    help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=80,
                    help='number of steps before each update (per thread)')
parser.add_argument('--nprocesses', type=int, default=1,
                    help='How many processes to run')
# model
parser.add_argument('--hid_size', default=128, type=int,
                    help='hidden layer size')
parser.add_argument('--recurrent', action='store_true', default=True,
                    help='make the model recurrent in time')
# optimization
parser.add_argument('--gamma', type=float, default=1.0,
                    help='discount factor')
parser.add_argument('--tau', type=float, default=1.0,
                    help='gae (remove?)')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed. Pass -1 for random seed')  # TODO: works in thread?
parser.add_argument('--normalize_rewards', action='store_true', default=False,
                    help='normalize rewards in each batch')
parser.add_argument('--lrate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--entr', type=float, default=0,
                    help='entropy regularization coeff')
parser.add_argument('--value_coeff', type=float, default=0.01,
                    help='coeff for value loss term')
# environment
parser.add_argument('--env_name', default="Cartpole",
                    help='name of the environment to run')
parser.add_argument('--max_steps', default=2, type=int,
                    help='force to end the game after this many steps')
parser.add_argument('--nactions', default='1', type=str,
                    help='the number of agent actions (0 for continuous). Use N:M:K for multiple actions')
parser.add_argument('--action_scale', default=1.0, type=float,
                    help='scale action output from model')
# other
parser.add_argument('--plot', action='store_true', default=False,
                    help='plot training progress')
parser.add_argument('--plot_env', default='main', type=str,
                    help='plot env name')
parser.add_argument('--save', default='/home/deepdrone/DroneSwarm_MP_IC3/model/', type=str,
                    help='save the model after training')
parser.add_argument('--save_every', default=2, type=int,
                    help='save the model after every n_th epoch')
parser.add_argument('--load', default='', type=str,
                    help='load the model')
parser.add_argument('--display', action="store_true", default=False,
                    help='Display environment state')


parser.add_argument('--random', action='store_true', default=False,
                    help="enable random model")

# CommNet specific args
parser.add_argument('--mode',  default="Test",
                    help="Train or Test mode")
parser.add_argument('--commnet', action='store_true', default=False,
                    help="enable commnet model")
parser.add_argument('--ic3net', action='store_true', default=True,
                    help="enable commnet model")
parser.add_argument('--nagents', type=int, default=2,
                    help="Number of agents (used in multiagent)")
parser.add_argument('--comm_mode', type=str, default='avg',
                    help="Type of mode for communication tensor calculation [avg|sum]")
parser.add_argument('--comm_passes', type=int, default=1,
                    help="Number of comm passes per step over the model")
parser.add_argument('--comm_mask_zero', action='store_true', default=False,
                    help="Whether communication should be there")
parser.add_argument('--mean_ratio', default=1.0, type=float,
                    help='how much coooperative to do? 1.0 means fully cooperative')
parser.add_argument('--rnn_type', default='MLP', type=str,
                    help='type of rnn to use. [LSTM|MLP]')
parser.add_argument('--detach_gap', default=10, type=int,
                    help='detach hidden state and cell state for rnns at this interval.'
                    + ' Default 10000 (very high)')
parser.add_argument('--comm_init', default='uniform', type=str,
                    help='how to initialise comm weights [uniform|zeros]')
parser.add_argument('--hard_attn', default=False, action='store_true',
                    help='Whether to use hard attention: action - talk|silent')
parser.add_argument('--comm_action_one', default=False, action='store_true',
                    help='Whether to always talk, sanity check for hard attention.')
parser.add_argument('--advantages_per_action', default=False, action='store_true',
                    help='Whether to multipy log porb for each chosen action with advantages')
parser.add_argument('--share_weights', default=False, action='store_true',
                    help='Share weights for hops')


init_args_for_env(parser)
args = parser.parse_args()

if args.ic3net:
    args.commnet = 1
    args.hard_attn = 1
    args.mean_ratio = 0

    # For TJ set comm action to 1 as specified in paper to showcase
    # importance of individual rewards even in cooperative games
    if args.env_name == "traffic_junction":
        args.comm_action_one = True

# env = data.init(args.env_name, args, False)
n_agents = 5
args.nagents = n_agents
N_frame = 5
visualization = False
is_centralized = False
env = QuadrotorFormation(n_agents=n_agents, N_frame=N_frame,
                         visualization=visualization, is_centralized=is_centralized)

# Enemy comm
args.nfriendly = args.nagents
if hasattr(args, 'enemy_comm') and args.enemy_comm:
    if hasattr(args, 'nenemies'):
        args.nagents += args.nenemies
    else:
        raise RuntimeError("Env. needs to pass argument 'nenemy'.")

# num_inputs = env.observation_dim
args.num_actions = env.n_action
args.naction_heads = env.n_action

# Multi-action
if not isinstance(args.num_actions, (list, tuple)):  # single action case
    args.num_actions = [args.num_actions]
args.dim_actions = env.dim_actions
# args.num_inputs = num_inputs

# Hard attention
if args.hard_attn and args.commnet:
    # add comm_action as last dim in actions
    args.num_actions = [*args.num_actions, 2]
    args.dim_actions = env.dim_actions + 1

# Recurrence
if args.commnet and (args.recurrent or args.rnn_type == 'LSTM'):
    args.recurrent = True
    args.rnn_type = 'LSTM'


parse_action_args(args)

if args.seed == -1:
    args.seed = np.random.randint(0, 10000)
torch.manual_seed(args.seed)

# print(args)


if args.commnet:
    policy_net = CommNetMLP(args, n_agents)
elif args.random:
    policy_net = Random(args, n_agents)
elif args.recurrent:
    policy_net = RNN(args, n_agents)
else:
    policy_net = MLP(args, n_agents)

if not args.display:
    display_models([policy_net])

# print ("policy_net: ", policy_net)

# share parameters among threads, but not gradients
for p in policy_net.parameters():
    p.data.share_memory_()

if args.nprocesses > 1:
    trainer = MultiProcessTrainer(args, lambda: Trainer(
        args, policy_net, env, is_centralized))
else:
    trainer = Trainer(args, policy_net, env, is_centralized)
    tester = Tester(args, policy_net, env, is_centralized)

disp_trainer = Trainer(args, policy_net, env)
disp_trainer.display = True

reporter = Reporter()


def disp():
    x = disp_trainer.get_episode()


log = dict()
log['epoch'] = LogField(list(), False, None, None)
log['reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['enemy_reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['success'] = LogField(list(), True, 'epoch', 'num_episodes')
log['steps_taken'] = LogField(list(), True, 'epoch', 'num_episodes')
log['add_rate'] = LogField(list(), True, 'epoch', 'num_episodes')
log['comm_action'] = LogField(list(), True, 'epoch', 'num_steps')
log['enemy_comm'] = LogField(list(), True, 'epoch', 'num_steps')
log['value_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['action_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['entropy'] = LogField(list(), True, 'epoch', 'num_steps')

# if args.plot:
#     vis = visdom.Visdom(env=args.plot_env)


def train_run(num_epochs):
    # load("./model/model_train_0.pt")
    best_reward_train = -np.Inf
    best_covered_grids = 0.0
    best_reward_test = -np.Inf
    eval_period = 100
    uncertainty_threshold = 0.6

    for ep in range(num_epochs):
        epoch_begin_time = time.time()
        stat = dict()
        # for n in range(args.epoch_size):
        # if n == args.epoch_size - 1 and args.display:
        #     trainer.display = True
        s, uncertainty_values = trainer.train_batch(ep)
        merge_stat(s, stat)
        # trainer.display = False

        epoch_time = time.time() - epoch_begin_time
        epoch = len(log['epoch'].data) + 1
        for k, v in log.items():
            if k == 'epoch':
                v.data.append(epoch)
            else:
                if k in stat and v.divide_by is not None and stat[v.divide_by] > 0:
                    stat[k] = stat[k] / stat[v.divide_by]
                v.data.append(stat.get(k, 0))

        # np.set_printoptions(precision=2)

        # print('Epoch {}\tReward {}\tTime {:.2f}s'.format(
        #         epoch, stat['reward'], epoch_time
        # ))

        # if 'enemy_reward' in stat.keys():
        #     print('Enemy-Reward: {}'.format(stat['enemy_reward']))
        # if 'add_rate' in stat.keys():
        #     print('Add-Rate: {:.2f}'.format(stat['add_rate']))
        # if 'success' in stat.keys():
        #     print('Success: {:.2f}'.format(stat['success']))
        # if 'steps_taken' in stat.keys():
        #     print('Steps-taken: {:.2f}'.format(stat['steps_taken']))
        # if 'comm_action' in stat.keys():
        #     print('Comm-Action: {}'.format(stat['comm_action']))
        # if 'enemy_comm' in stat.keys():
        #     print('Enemy-Comm: {}'.format(stat['enemy_comm']))

        # if args.plot:
        #     for k, v in log.items():
        #         if v.plot and len(v.data) > 0:
        #             vis.line(np.asarray(v.data), np.asarray(log[v.x_axis].data[-len(v.data):]),
        #             win=k, opts=dict(xlabel=v.x_axis, ylabel=k))

        mean_reward = np.mean(stat["reward"])
        uncertainty_map = np.zeros_like(uncertainty_values)
        uncertainty_map[uncertainty_values <= uncertainty_threshold] = 1.0
        sum_covered_grids = np.sum(uncertainty_map)
        if sum_covered_grids > best_covered_grids:
            print(
                "Better covered value in Train mode is obtained! The model is being saved!")
            best_covered_grids = sum_covered_grids
            save(args.save, ep + 1, "train")

        if (ep + 1) % eval_period == 0:
            print("\n In eval mode")
            test_stat = tester.test_batch(save=False)
            test_reward = np.mean(test_stat["reward"])
            if test_reward > best_reward_test:
                print("The model is being saved!")
                best_reward_test = test_reward
                save(args.save, ep + 1, "test")

        # if args.save != '':
        #     save(args.save)


def save(path, epoch, save_type):
    d = dict()
    d['policy_net'] = policy_net.state_dict()
    d['log'] = log
    d['trainer'] = trainer.state_dict()
    file_path = path + "model_" + save_type + "_" + str(epoch) + ".pt"
    # file_path = "./model/" + "model_" + save_type + "_" + str(epoch) + ".pt"
    torch.save(d, file_path)


def load(path):
    d = torch.load(path)
    # log.clear()
    policy_net.load_state_dict(d['policy_net'])
    log.update(d['log'])
    trainer.load_state_dict(d['trainer'])
    tester.load_state_dict(d['trainer'])


def signal_handler(signal, frame):
    print('You pressed Ctrl+C! Exiting gracefully.')
    if args.display:
        env.end_display()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

if args.load != '':
    load(args.load)

if args.mode == "Train":
    print("Train mode is executed! \n")
    time.sleep(1.0)
    train_run(args.num_epochs)
else:
    print("Test mode is executed! \n")
    time.sleep(1.0)
    model_path = './model'
    for selected_model in glob.glob(os.path.join(model_path, '*.pt')):
        print("Selected model is ", selected_model)
        load(selected_model)  # "./model/model_test_3900.pt"
        pkl_name = os.path.split(os.path.splitext(selected_model)[0])[
            1] + '.pkl'
        tester.test_batch(pkl_name)
        time.sleep(1.0)
        reporter.get_map_coverage(pkl_name)


if args.display:
    env.end_display()

if args.save != '' and args.mode == "Train":
    save(args.save, 0, "train")

if sys.flags.interactive == 0 and args.nprocesses > 1:
    trainer.quit()
    import os
    os._exit(0)
