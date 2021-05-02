import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from models_drone import MLP
from action_utils import select_action, translate_action

def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CommNetMLP(nn.Module):
    """
    MLP based CommNet. Uses communication vector to communicate info
    between agents
    """
    def __init__(self, args, env):
        """Initialization method for this class, setup various internal networks
        and weights

        Arguments:
            MLP {object} -- Self
            args {Namespace} -- Parse args namespace
            n_agents {number} -- Number of agents
        """

        super(CommNetMLP, self).__init__()
        self.args = args
        self.env = env
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.comm_passes = args.comm_passes
        self.recurrent = args.recurrent
        self.encoder2_hid_size = 20
        self.num_inputs = 6*7*7
        self.encoder_out = self.encoder2_hid_size + self.nagents*3 + 1
        self.num_channels = 1#(n_agents + 1)*5 + 1 + 1
        self.net = nn.Sequential(
            nn.Conv2d(self.num_channels, 3, 3, 2),        
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),  
            nn.Conv2d(3, 6, 3, 1),         
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),  
            nn.Conv2d(6, 6, 3, 1),         
            nn.ReLU(),
            Flatten()
        ).apply(initialize_weights_he)

        self.net1d = nn.Conv1d(self.nagents*3 + 1,self.nagents*3 + 1, 1)


        self.continuous = args.continuous
        args.naction_heads = [self.env.n_action]
        # print ("naction_heads: ", args.naction_heads)
        # print ("comm_passes: ", args.comm_passes)
        # print ("hid_size: ", args.hid_size)
        # print ("batch_size: ", args.batch_size)
        if self.continuous:
            self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
            self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
        else:
            self.heads = nn.ModuleList([nn.Linear(self.encoder_out, o)
                                        for o in args.naction_heads])
            # print ("self.heads: ", self.heads)
        self.init_std = args.init_std if hasattr(args, 'comm_init_std') else 0.2

        # Mask for communication
        if self.args.comm_mask_zero:
            self.comm_mask = torch.zeros(self.nagents, self.nagents)
        else:
            self.comm_mask = torch.ones(self.nagents, self.nagents) \
                            - torch.eye(self.nagents, self.nagents)


        # Since linear layers in PyTorch now accept * as any number of dimensions
        # between last and first dim, num_agents dimension will be covered.
        # The network below is function r in the paper for encoding
        # initial environment stage
        self.encoder = nn.Linear(self.num_inputs, args.hid_size)
        self.encoder_2 = nn.Linear(args.hid_size, self.encoder2_hid_size)
        self.encoder_battery = nn.Linear(1, 1)

        # if self.args.env_name == 'starcraft':
        #     self.state_encoder = nn.Linear(num_inputs, num_inputs)
        #     self.encoder = nn.Linear(num_inputs * 2, args.hid_size)
        if args.recurrent:
            self.hidd_encoder = nn.Linear(args.hid_size, args.hid_size)

        if args.recurrent:
            self.init_hidden(args.batch_size)
            self.f_module = nn.LSTMCell(self.encoder_out, self.encoder_out)

        else:
            if args.share_weights:
                self.f_module = nn.Linear(args.hid_size, args.hid_size)
                self.f_modules = nn.ModuleList([self.f_module
                                                for _ in range(self.comm_passes)])
            else:
                self.f_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size)
                                                for _ in range(self.comm_passes)])
        # else:
            # raise RuntimeError("Unsupported RNN type.")

        # Our main function for converting current hidden state to next state
        # self.f = nn.Linear(args.hid_size, args.hid_size)
        if args.share_weights:
            self.C_module = nn.Linear(args.hid_size, args.hid_size)
            self.C_modules = nn.ModuleList([self.C_module
                                            for _ in range(self.comm_passes)])
        else:
            self.C_modules = nn.ModuleList([nn.Linear(self.encoder_out, self.encoder_out)
                                            for _ in range(self.comm_passes)])
        # self.C = nn.Linear(args.hid_size, args.hid_size)

        # initialise weights as 0
        if args.comm_init == 'zeros':
            for i in range(self.comm_passes):
                self.C_modules[i].weight.data.zero_()
        self.tanh = nn.Tanh()

        # print(self.C)
        # self.C.weight.data.zero_()
        # Init weights for linear layers
        # self.apply(self.init_weights)

        self.value_head = nn.Linear(self.encoder_out, 1)


    def get_agent_mask(self, batch_size, info):
        n = self.nagents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n

        agent_mask = agent_mask.view(1, 1, n)
        agent_mask = agent_mask.expand(batch_size, n, n).unsqueeze(-1)

        return num_agents_alive, agent_mask

    def forward_state_encoder(self, x, uncertainty_map, info):
        hidden_state, cell_state = None, None

        # num_agents_alive = np.sum(info['alive_mask'])

        # if self.continuous == False:
        #     self.heads = nn.ModuleList([nn.Linear(self.encoder_out, num_agents_alive)])

        if self.args.recurrent:
            x, extras = x
            uncertainty_out = self.net(uncertainty_map) # Input 82*82, Output 297
            uncertainty_out = self.encoder(uncertainty_out) # Input 297, Output 128
            uncertainty_out = self.encoder_2(uncertainty_out) # Input 128, Output 20

            x = x.unsqueeze(2) # Dimension (n_agents, n_agents*3 + 1, 1)
            x = self.net1d(x) # Dimension (n_agents, n_agents*3 + 1, 1)
            x = x.squeeze() # Dimension (n_agents, n_agents*3 + 1)
            uncertainty_out = uncertainty_out.repeat(self.nagents, 1) # Dimension (n_agents, 20)
            x = torch.cat((x, uncertainty_out), 1) # Dimension (n_agents, n_agents*3 + 1 + 20)            

            if self.args.rnn_type == 'LSTM':
                hidden_state, cell_state = extras
            else:
                hidden_state = extras
            # hidden_state = self.tanh( self.hidd_encoder(prev_hidden_state) + x)

            # print ("(fcn encoder) hiddent_state: " ,hidden_state.size())
            # print ("(fcn encoder) cell_state: " ,cell_state.size())
            # stop
        else:
            uncertainty_out = self.net(uncertainty_map) # Input 82*82, Output 297
            uncertainty_out = self.encoder(uncertainty_out) # Input 297, Output 128
            uncertainty_out = self.encoder_2(uncertainty_out) # Input 128, Output 20

            x = self.net1d(x) # output n_agents*3 + 1
            x = torch.cat((x, uncertainty_out), 1) # output 20  
            hidden_state = x

        return x, hidden_state, cell_state


    def forward(self, x, uncertainty_map, info={}):
        # TODO: Update dimensions
        """Forward function for CommNet class, expects state, previous hidden
        and communication tensor.
        B: Batch Size: Normally 1 in case of episode
        N: number of agents

        Arguments:
            x {tensor} -- State of the agents (N x num_inputs)
            prev_hidden_state {tensor} -- Previous hidden state for the networks in
            case of multiple passes (1 x N x hid_size)
            comm_in {tensor} -- Communication tensor for the network. (1 x N x N x hid_size)

        Returns:
            tuple -- Contains
                next_hidden {tensor}: Next hidden state for network
                comm_out {tensor}: Next communication tensor
                action_data: Data needed for taking next action (Discrete values in
                case of discrete, mean and std in case of continuous)
                v: value head
        """

        # if self.args.env_name == 'starcraft':
        #     maxi = x.max(dim=-2)[0]
        #     x = self.state_encoder(x)
        #     x = x.sum(dim=-2)
        #     x = torch.cat([x, maxi], dim=-1)
        #     x = self.tanh(x)

        x, hidden_state, cell_state = self.forward_state_encoder(x, uncertainty_map, info)
        # x = x.view(-1, x.size(-2), x.size(-1))
        # print ("x: ", x.size())
        # print ("hidden_state: ", hidden_state.size())
        n = self.nagents
        batch_size = int(x.size()[0] / n)
        
        # print ("n: ", n)
        # print ("batch_size: ", batch_size)

        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)
        # print ("num_agents_alive: ", num_agents_alive)
        

        # print ("num_agents_alive: ", num_agents_alive)
        # print ("agent_mask: ", agent_mask.size())
        # print ("comm_passes: ", self.comm_passes)

        # Hard Attention - action whether an agent communicates or not
        if self.args.hard_attn:
            comm_action = torch.tensor(info['comm_action'])
            comm_action_mask = comm_action.expand(batch_size, n, n).unsqueeze(-1)
            # action 1 is talk, 0 is silent i.e. act as dead for comm purposes.
            agent_mask *= comm_action_mask.double()

        agent_mask_transpose = agent_mask.transpose(1, 2)

        for i in range(self.comm_passes):
            # Choose current or prev depending on recurrent
            comm = hidden_state.view(batch_size, n, self.encoder_out) if self.args.recurrent else hidden_state

            # Get the next communication vector based on next hidden state
            comm = comm.unsqueeze(-2).expand(-1, n, n, self.encoder_out)

            # Create mask for masking self communication
            mask = self.comm_mask.view(1, n, n)
            mask = mask.expand(comm.shape[0], n, n)
            mask = mask.unsqueeze(-1)

            mask = mask.expand_as(comm)
            comm = comm * mask

            if hasattr(self.args, 'comm_mode') and self.args.comm_mode == 'avg' \
                and num_agents_alive > 1:
                comm = comm / (num_agents_alive - 1)

            # Mask comm_in
            # Mask communcation from dead agents
            comm = comm * agent_mask
            # Mask communication to dead agents
            comm = comm * agent_mask_transpose

            # Combine all of C_j for an ith agent which essentially are h_j
            comm_sum = comm.sum(dim=1)
            c = self.C_modules[i](comm_sum)


            if self.args.recurrent:
                # skip connection - combine comm. matrix and encoded input for all agents
                inp = x + c
                inp = inp.view(batch_size * n, self.encoder_out)

                output = self.f_module(inp, (hidden_state, cell_state))

                hidden_state = output[0]
                cell_state = output[1]

                # print ("hidden_state: ",hidden_state.size())
                # print ("cell_state: ",cell_state.size())

            else: # MLP|RNN
                # Get next hidden state from f node
                # and Add skip connection from start and sum them
                hidden_state = sum([x, self.f_modules[i](hidden_state), c])
                hidden_state = self.tanh(hidden_state)

        # v = torch.stack([self.value_head(hidden_state[:, i, :]) for i in range(n)])
        # v = v.view(hidden_state.size(0), n, -1)
        value_head = self.value_head(hidden_state)
        h = hidden_state.view(batch_size*n, self.encoder_out)

        # print ("value_head: ",value_head.size())
        # print ("h: ",h.size())

        if self.continuous:
            action_mean = self.action_mean(h)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            # will be used later to sample
            action = (action_mean, action_log_std, action_std)
        else:
            # discrete actions
            action = [F.log_softmax(head(h), dim=-1) for head in self.heads]
            # print ("action: ", action)

        if self.args.recurrent:
            return action, value_head, (hidden_state.clone(), cell_state.clone())
        else:
            return action, value_head

    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, self.init_std)

    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        return tuple(( torch.zeros(batch_size * self.nagents, self.encoder_out, requires_grad=True),
                       torch.zeros(batch_size * self.nagents, self.encoder_out, requires_grad=True)))

