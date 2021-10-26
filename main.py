import sys
import time
import signal
import argparse
import airsim
import pprint
from msgpackrpc.transport.tcp import ClientSocket

import numpy as np
import torch
import visdom
import data
import socket
from models import *
from comm import CommNetMLP
from utils import *
from action_utils import parse_action_args
from trainer import Trainer
from multi_processing import MultiProcessTrainer
from systematic_results import Reporter
import pickle
import glob
import os
from json_editor import Json_Editor

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch RL trainer')
# training
# note: number of steps per epoch = epoch_size X batch_size x nprocesses
parser.add_argument('--num_epochs', default=2000, type=int,
                    help='number of training epochs')
parser.add_argument('--epoch_size', type=int, default=1,
                    help='number of update iterations in an epoch')
parser.add_argument('--max_steps', default=1000, type=int,
                    help='force to end the game after this many steps')
parser.add_argument('--batch_size', type=int, default=200,
                    help='number of steps before each update (per thread)')
parser.add_argument('--nprocesses', type=int, default=1,
                    help='How many processes to run')
parser.add_argument('--train_thresh', type=float, default=0.3,
                    help='train threshold to stop the training')
parser.add_argument('--last_n_episode', type=int, default=10,
                    help='Last n episodes to check if training should be stopped')
parser.add_argument('--min_episode', type=int, default=500,
                    help='After min episodes to check if training should be stopped')
# model
parser.add_argument('--hid_size', default=128, type=int,
                    help='hidden layer size')
parser.add_argument('--recurrent', action='store_true', default=True,
                    help='make the model recurrent in time')
# optimization
parser.add_argument('--gamma', type=float, default=1.0,
                    help='discount factor')
# parser.add_argument('--tau', type=float, default=1.0,
#                     help='gae (remove?)')
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
# parser.add_argument('--env_name', default="Cartpole",
#                     help='name of the environment to run')
# parser.add_argument('--nactions', default='1', type=str,
#                     help='the number of agent actions (0 for continuous). Use N:M:K for multiple actions')
# parser.add_argument('--action_scale', default=1.0, type=float,
#                     help='scale action output from model')
# other
# parser.add_argument('--plot', action='store_true', default=False,
#                     help='plot training progress')
# parser.add_argument('--plot_env', default='main', type=str,
#                     help='plot env name')

# Abdullahtan al save kismini
parser.add_argument('--save', default="True", type=str,
                    help='save the model after training')

parser.add_argument('--save_every', default=100, type=int,
                    help='save the model after every n_th epoch')
parser.add_argument('--load', default="False", type=str,
                    help='load the model')
parser.add_argument('--display', action="store_true", default=False,
                    help='Display environment state')
parser.add_argument('--random', action='store_true', default=False,
                    help="enable random model")

# CommNet specific args
parser.add_argument('--commnet', action='store_true', default=False,
                    help="enable commnet model")
parser.add_argument('--ic3net', action='store_true', default=True,
                    help="enable commnet model")
parser.add_argument('--nagents', type=int, default=5,
                    help="Number of agents (used in multiagent)")
parser.add_argument('--nbots', type=int, default=0,
                    help="Number of bots (used in multiagent)")
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
# parser.add_argument('--test', default=False, type=bool,
#                     help='Train or Test')
parser.add_argument('--mode', default="Train", type=str,
                    help='Train or Test')   
parser.add_argument('--test-model', default="weight/planning.pt", type=str,
                    help='Model to test')    
parser.add_argument('--scenario', type=str, default='planning',
                    help='predator or planning ')
parser.add_argument('--airsim_vis', action='store_true', default=False,
                    help='Visualize in Airsim when testing')
parser.add_argument('--visualization', action='store_true', default=False,
                    help="enable commnet model")


# init_args_for_env(parser)
args = parser.parse_args()

# Data to be written
dictionary = {
    "name": args.scenario,
}

if args.ic3net:
    args.commnet = 1
    args.hard_attn = 1
    args.mean_ratio = 0

# Enemy comm
args.nfriendly = args.nagents
if hasattr(args, 'enemy_comm') and args.enemy_comm:
    if hasattr(args, 'nenemies'):
        args.nagents += args.nenemies
    else:
        raise RuntimeError("Env. needs to pass argument 'nenemy'.")

# visualization = True
is_centralized = False
N_frame = 5

if args.scenario == 'predator':
    from predator_prey import QuadrotorFormation
    env = QuadrotorFormation(n_agents=args.nagents, n_bots=args.nbots, visualization=args.visualization)
    if(args.airsim_vis == True):
        #Set Up JSON file for AirSim
        js_modifier = Json_Editor(2*args.nagents)
        js_modifier.modify()

elif args.scenario == 'planning':
    from planning import QuadrotorFormation
    env = QuadrotorFormation(n_agents=args.nagents, N_frame=N_frame,
                             visualization=args.visualization, is_centralized=is_centralized)
    if(args.airsim_vis == True):
        #Set Up JSON file for AirSim
        js_modifier = Json_Editor(args.nagents)
        js_modifier.modify()

else:
    print("Scenario is wrong. Please select: predator or planning")
num_inputs = 12
args.num_actions = env.n_action


# Multi-action
if not isinstance(args.num_actions, (list, tuple)):  # single action case
    args.num_actions = [args.num_actions]

args.dim_actions = 1
args.num_inputs = num_inputs

# Hard attention
if args.hard_attn and args.commnet:
    # add comm_action as last dim in actions
    args.num_actions = [*args.num_actions, 2]
    args.dim_actions = args.dim_actions + 1

# Recurrence
if args.commnet and (args.recurrent or args.rnn_type == 'LSTM'):
    args.recurrent = True
    args.rnn_type = 'LSTM'

parse_action_args(args)

if args.seed == -1:
    args.seed = np.random.randint(0, 10000)
torch.manual_seed(args.seed)

print(args)


if args.commnet:
    print("Policy Net: CommNetMLP")
    policy_net = CommNetMLP(args, num_inputs)
elif args.random:
    policy_net = Random(args, num_inputs)
elif args.recurrent:
    policy_net = RNN(args, num_inputs)
    print("Policy Net: RNN")
else:
    policy_net = MLP(args, num_inputs)

if not args.display:
    display_models([policy_net])

# share parameters among threads, but not gradients
for p in policy_net.parameters():
    p.data.share_memory_()

if args.nprocesses > 1:
    trainer = MultiProcessTrainer(args, lambda: Trainer(args, policy_net, env))
else:
    if args.scenario == 'predator':
        trainer = Trainer(args, policy_net, env, None)
        disp_trainer = Trainer(args, policy_net, env, None)
    elif args.scenario == 'planning':
        trainer = Trainer(args, policy_net, env, is_centralized)
        disp_trainer = Trainer(args, policy_net, env, is_centralized)

#disp_trainer = Trainer(args, policy_net, env)
disp_trainer.display = True


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

def run(num_epochs):
    episode_surv_rates = []

    takeoff = False
    if args.mode=='Train' or args.mode=='train':
        print("TRAIN MODE")
        for ep in range(num_epochs):
            epoch_begin_time = time.time()
            stat = dict()
            for n in range(args.epoch_size):
                if n == args.epoch_size - 1 and args.display:
                    trainer.display = True
                if args.scenario == "planning":
                    s, mean_surv_rate = trainer.train_batch(ep)
                    episode_surv_rates.append(mean_surv_rate)
                elif args.scenario == "predator":
                    s = trainer.train_batch(ep)
                
                merge_stat(s, stat)
                trainer.display = False

            epoch_time = time.time() - epoch_begin_time
            epoch = len(log['epoch'].data) + 1
            for k, v in log.items():
                if k == 'epoch':
                    v.data.append(epoch)
                else:
                    if k in stat and v.divide_by is not None and stat[v.divide_by] > 0:
                        stat[k] = stat[k] / stat[v.divide_by]
                    v.data.append(stat.get(k, 0))

            np.set_printoptions(precision=2)

            print('\n \tReward: {}\tTime {:.2f}s \n'.format(
                stat['reward'], epoch_time
            ))

            if 'enemy_reward' in stat.keys():
                print('Enemy-Reward: {}'.format(stat['enemy_reward']))
            if 'add_rate' in stat.keys():
                print('Add-Rate: {:.2f}'.format(stat['add_rate']))
            if 'success' in stat.keys():
                print('Success: {:.2f}'.format(stat['success']))
            # if 'steps_taken' in stat.keys():
            #     print('Steps-taken: {:.2f}'.format(stat['steps_taken']))
            # if 'comm_action' in stat.keys():
            #     print('Comm-Action: {}'.format(stat['comm_action']))
            if 'enemy_comm' in stat.keys():
                print('Enemy-Comm: {}'.format(stat['enemy_comm']))

            if args.scenario == "planning":
                if np.mean(episode_surv_rates[-args.last_n_episode:]) < args.train_thresh and len(episode_surv_rates) >= args.min_episode :
                    print ("Mean Surveillance of last 5 episodes: {0:.3}. Too low surveillance rate! Exit!".format(np.mean(episode_surv_rates[-args.last_n_episode:])))
                    if args.visualization:
                        env.close()
                    break

            if args.visualization:
                env.close()
            
            # if args.plot:
            #     for k, v in log.items():
            #         if v.plot and len(v.data) > 0:
            #             vis.line(np.asarray(v.data), np.asarray(log[v.x_axis].data[-len(v.data):]),
            #                      win=k, opts=dict(xlabel=v.x_axis, ylabel=k))

            if ep % args.save_every == 0 and ep != 0:
                save(ep)
                
    elif args.mode == 'Test' or  args.mode =='test':
        print('TEST MODE')
        batch = []
        total_pos_list = []
        agent_pos_list = []
        bot_pos_list = []

        agents_list = [agent for agent in range(args.nagents)]
        bots_list = [bot for bot in range(args.nbots)]

        HOST = '127.0.0.1'
        PORT = 9090
    
        # with socket.socket(socket.AF_INET, socket.SO_REUSEADDR) as clientSocket:
        #     clientSocket.connect((HOST, PORT))

        clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        clientSocket.connect((HOST, PORT))

        for ep in range(args.epoch_size):
            takeoff = False

            if args.scenario == 'predator':
                _, agent_pos, bot_pos = trainer.test_batch(ep)

                agent_pos_list.append(agent_pos)
                bot_pos_list.append(bot_pos)
                with open('./agents_position/agents_positions.pkl', 'wb') as f:
                    pickle.dump(agent_pos_list, f)
                with open('./agents_position/bots_positions.pkl', 'wb') as f:
                    pickle.dump(bot_pos_list, f)
                
            elif args.scenario == 'planning':
                _, agent_pos = trainer.test_batch(ep)
                total_pos_list.append(agent_pos)
                with open('./agents_position/agents_positions_planner.pkl', 'wb') as f:
                    pickle.dump(total_pos_list, f)

            trainer.display = False

            if(args.airsim_vis == True):
                client = airsim.MultirotorClient()
                client.confirmConnection()

                for drn in agents_list:
                    client.enableApiControl(True, f"Drone{drn+1}")
                if args.scenario == 'predator':
                    for bt in bots_list:
                        client.enableApiControl(True, f"Drone{args.nagents+bt+1}")

                for drn in agents_list:  
                    client.armDisarm(True, f"Drone{drn+1}")    
                if args.scenario == 'predator':
                    for bt in bots_list:
                        client.armDisarm(True, f"Drone{bt+1}")

                if not takeoff:
                    airsim.wait_key('Press any key to takeoff')
                    f_list = []
                    for drn in agents_list:
                        f_list.append(client.takeoffAsync(vehicle_name=f"Drone{drn+1}"))
                    if args.scenario == 'predator':
                        fb_list = []
                        for bt in bots_list:
                            fb_list.append(client.takeoffAsync(vehicle_name=f"Drone{args.nagents+bt+1}"))
                    for fx in f_list:
                        fx.join()
                    if args.scenario == 'predator':
                        for fb in fb_list:
                            fb.join()
                    takeoff = True

                i = 0
                if args.scenario == 'predator':
                    for agent_p, bot_p in zip(agent_pos, bot_pos):

                        info_list = []

                        curr_agentPos = [[agent_p[drn][0], agent_p[drn][1], agent_p[drn][2]] for drn in range(len(agents_list))]

                        curr_botPos = [[bot_p[bt][0], bot_p[bt][1], bot_p[bt][2]] for bt in range(len(bots_list))]

                        info_list.append(curr_agentPos)
                        info_list.append(curr_botPos)
                        
                        info_data = pickle.dumps(info_list)
                        clientSocket.send(info_data)

                        if i == 0:
                            airsim.wait_key('Press any key to take initial position')

                            for drn in agents_list:
                                client.moveToPositionAsync(agent_p[drn][0], agent_p[drn][1], agent_p[drn][2], 6, vehicle_name=f"Drone{drn+1}")
                                
                            for bt in bots_list:
                                client.moveToPositionAsync(bot_p[bt][0], bot_p[bt][1], bot_p[bt][2], 6, vehicle_name=f"Drone{args.nagents+bt+1}")

                            airsim.wait_key('Press any key to start')
                            time.sleep(0.1)

                        else:
                            for drn in agents_list:
                                client.moveToPositionAsync(agent_p[drn][0], agent_p[drn][1], agent_p[drn][2], 3, vehicle_name=f"Drone{drn+1}")
                                
                            for bt in bots_list:
                                client.moveToPositionAsync(bot_p[bt][0], bot_p[bt][1], bot_p[bt][2], 2, vehicle_name=f"Drone{args.nagents+bt+1}")

                            time.sleep(0.1)
                        i += 1
                    

                elif args.scenario == 'planning':
                    f_list = []
                    for agent_p in zip(agent_pos):
                        
                        if i == 0:
                            airsim.wait_key('Press any key to take initial position')
                            for drn in agents_list:
                                f_list.append(client.moveToPositionAsync(agent_p[0][drn][0], agent_p[0][drn][1], -agent_p[0][drn][2], 5, vehicle_name=f"Drone{drn+1}"))

                            for fx in f_list:
                                fx.join() 

                        else:
                            for drn in agents_list:
                                client.moveToPositionAsync(agent_p[0][drn][0], agent_p[0][drn][1], -agent_p[0][drn][2], 5, vehicle_name=f"Drone{drn+1}")
                            time.sleep(0.1)
                        i += 1
                    
                    print("TEST RESULTS")
                    reporter = Reporter()
                    file_list = glob.glob('./agents_position/*.pkl')
                    for i, file in enumerate(file_list):
                        print("{0}/{1} file {2} is loaded! \n".format(i +
                                                                    1, len(file_list), file))
                        pkl_name = os.path.split(os.path.splitext(file)[0])[1] + '.pkl'
                        reporter.get_map_coverage(pkl_name)
                    
                
                s_list = []
                b_list = []
                for drn in agents_list:
                    s_list.append(client.takeoffAsync(vehicle_name=f"Drone{drn+1}"))
                if args.scenario == 'predator':
                    for bt in bots_list:
                        b_list.append(client.takeoffAsync(vehicle_name=f"Drone{args.nagents+bt+1}"))

                for sx in s_list:
                    sx.join() 

                if args.scenario == 'predator':
                    for sbx in b_list:
                        sbx.join()

    else:
        print("Wrong Mode!!!")


def save(ep):
    current_dir =  os.path.abspath(os.path.dirname(__file__))
    file_path = current_dir + "/weight" + "/" + args.scenario + "_" + str(ep) + ".pt"
    d = dict()
    d['policy_net'] = policy_net.state_dict()
    d['log'] = log
    d['trainer'] = trainer.state_dict()
    torch.save(d, file_path)
    print("model saved")


def load(test_model):
    current_dir =  os.path.abspath(os.path.dirname(__file__))
    # file_path = current_dir + "/weight" + "/" + args.scenario + ".pt"
    file_path = current_dir + "/" + test_model

    d = torch.load(file_path)
    policy_net.load_state_dict(d['policy_net'])
    log.update(d['log'])
    trainer.load_state_dict(d['trainer'])
    print("Weights loaded")


def signal_handler(signal, frame):
    print('You pressed Ctrl+C! Exiting gracefully.')
    if args.display:
        env.end_display()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

if args.load == "True" or args.mode == "Test":
    load(args.test_model)


run(args.num_epochs)
if args.display:
    env.end_display()


if sys.flags.interactive == 0 and args.nprocesses > 1:
    trainer.quit()
    import os
    os._exit(0)

if (args.mode == 'test' or args.mode == 'Test')  and args.scenario == 'planning':
    print("TEST RESULTS")
    reporter = Reporter()
    file_list = glob.glob('./agents_positions_planner/*.pkl')
    for i, file in enumerate(file_list):
        print("{0}/{1} file {2} is loaded! \n".format(i +
                                                      1, len(file_list), file))
        pkl_name = os.path.split(os.path.splitext(file)[0])[1] + '.pkl'
        reporter.get_map_coverage(pkl_name)
