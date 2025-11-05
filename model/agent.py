import copy
import os
import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.functional as F

from model.my_graph import MyGraph

from utils import *


class RewardNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

    def forward(self, s):
        return (self.fc(s) + 1) / 2.0


class ReplayMemory:
    def __init__(self, n_s, n_a, args):
        self.n_s = n_s
        self.n_a = n_a
        self.args = args
        self.MEMORY_SIZE = 1000
        self.BATCH_SIZE = 64

        self.all_cur_emb = None
        self.all_next_emb = None
        if args.is_train and args.is_irl:
            self.all_cur_emb = np.empty(shape=(self.MEMORY_SIZE, 32), dtype=np.float16)
            self.all_next_emb = np.empty(shape=(self.MEMORY_SIZE, 32), dtype=np.float16)
        
        self.all_s = np.empty(shape=(self.MEMORY_SIZE, self.args.num_of_node, 64), dtype=np.float16)
        self.all_a = np.random.randint(low=0, high=self.n_a, size=self.MEMORY_SIZE, dtype=np.int16)
        self.all_r = np.empty(self.MEMORY_SIZE, dtype=np.float64)

        self.all_s_ = np.empty(shape=(self.MEMORY_SIZE, self.args.num_of_node, 64), dtype=np.float16)
        self.count = 0
        self.t = 0

    def add(self, s, a, r, s_, cur_emb=None, next_emb=None):
        self.all_s[self.t] = s
        self.all_a[self.t] = a
        self.all_r[self.t] = r
        self.all_s_[self.t] = s_
        self.count = max(self.count, self.t + 1)

        if cur_emb and next_emb:
            self.all_cur_emb[self.t] = cur_emb
            self.all_next_emb[self.t] = next_emb

        self.t = (self.t + 1) % self.MEMORY_SIZE

    def can_sample(self):
        return self.count > 0

    def sample(self):
        if self.count < self.BATCH_SIZE:
            indexes = range(0, self.count)
        else:
            indexes = random.sample(range(0, self.count), self.BATCH_SIZE)

        batch_s = []
        batch_a = []
        batch_r = []
        batch_s_ = []
        batch_next_node_emb = []
        batch_cur_node_emb = []

        for idx in indexes:
            batch_s.append(self.all_s[idx])
            batch_a.append(self.all_a[idx])
            batch_r.append(self.all_r[idx])
            batch_s_.append(self.all_s_[idx])
            if self.all_cur_emb is not None and self.all_next_emb is not None:
                batch_cur_node_emb.append(self.all_cur_emb[idx])
                batch_next_node_emb.append(self.all_next_emb[idx])

        batch_s_tensor = torch.as_tensor(np.asarray(batch_s), dtype=torch.float32)
        batch_a_tensor = torch.as_tensor(np.asarray(batch_a), dtype=torch.int64).unsqueeze(-1)
        batch_r_tensor = torch.as_tensor(np.asarray(batch_r), dtype=torch.float32).unsqueeze(-1)
        batch_s__tensor = torch.as_tensor(np.asarray(batch_s_), dtype=torch.float32)

        return batch_s_tensor, batch_a_tensor, batch_r_tensor, batch_s__tensor, batch_cur_node_emb, batch_next_node_emb


class DQN(nn.Module):
    def __init__(self, n_input, n_output, device):
        super().__init__()  # Reuse the param of nn.Module
        self.device = device
        self.fc1 = nn.Linear(n_input, 32)
        self.fc2 = nn.Linear(32, n_output)

    def forward(self, x1):
        x1 = x1.to(self.device)
        out1 = self.fc1(x1)
        out2 = F.relu(out1)
        out = self.fc2(out2)

        return out

    def act(self, to_nodes, obs):
        q_values = self(obs)

        q_values_numpy = q_values.t().squeeze(0).detach().cpu().numpy()

        q_values_dict = {}
        to_nodes = np.array(to_nodes).astype(dtype=int).tolist()
        for to_vex in to_nodes:
            q_values_dict.update({to_vex: q_values_numpy[to_vex]})

        return q_values_dict


class Agent:
    def __init__(self, agent_id, n_input, n_output, device, args, mode="train"):
        self.device = device
        self.mode = mode
        self.n_input = n_input
        self.n_output = n_output

        self.ADD_GAMMA = 0.99
        self.DLT_GAMMA = 0.99

        self.learning_rate = 1e-3

        self.add_memo = ReplayMemory(n_s=self.n_input, n_a=self.n_output, args=args)
        self.dlt_memo = ReplayMemory(n_s=self.n_input, n_a=self.n_output, args=args)

        if self.mode == "train":
            self.add_online_net = DQN(self.n_input, self.n_output, self.device).to(device)
            self.add_target_net = DQN(self.n_input, self.n_output, self.device).to(device)

            self.add_target_net.load_state_dict(self.add_online_net.state_dict())

            self.add_optimizer = torch.optim.Adam(self.add_online_net.parameters(), lr=self.learning_rate)

            self.dlt_online_net = DQN(self.n_input, self.n_output, self.device).to(device)
            self.dlt_target_net = DQN(self.n_input, self.n_output, self.device).to(device)

            self.dlt_target_net.load_state_dict(self.add_online_net.state_dict())

            self.dlt_optimizer = torch.optim.Adam(self.add_online_net.parameters(), lr=self.learning_rate)
        else:
            self.add_online_net = DQN(self.n_input, self.n_output, self.device).to(device)
            self.dlt_online_net = DQN(self.n_input, self.n_output, self.device).to(device)

            self.add_online_net.load_state_dict(torch.load(os.path.join(args.dqn_save_dir, f"add_{agent_id % 300}.pth")))
            self.dlt_online_net.load_state_dict(torch.load(os.path.join(args.dqn_save_dir, f"dlt_{agent_id % 300}.pth")))

            self.add_online_net.eval()
            self.dlt_online_net.eval()

    def add_act(self, nodes, obs):
        return self.add_online_net.act(nodes, obs)

    def dlt_act(self, nodes, obs):
        return self.dlt_online_net.act(nodes, obs)

    def add_learn(self, reward_model=None):
        if not self.add_memo.can_sample():
            return
        batch_s, batch_a, batch_r, batch_s_, batch_node_emb, batch_node_next_emb = self.add_memo.sample()

        batch_s_tensor = torch.as_tensor(np.asarray(batch_s), dtype=torch.float32).to(self.device)
        batch_a_tensor = torch.as_tensor(np.asarray(batch_a), dtype=torch.int64).to(self.device)
        batch_s__tensor = torch.as_tensor(np.asarray(batch_s_), dtype=torch.float32).to(self.device)
        
        if reward_model:
            batch_emb_tensor = torch.as_tensor(np.asarray(batch_node_emb), dtype=torch.float32).to(self.device)
            batch_emb_next_tensor = torch.as_tensor(np.asarray(batch_node_next_emb), dtype=torch.float32).to(self.device)
            with torch.no_grad():
                batch_r_tensor = reward_model(batch_emb_next_tensor) - reward_model(batch_emb_tensor)
        else:
            batch_r_tensor = torch.as_tensor(np.asarray(batch_r), dtype=torch.float32).to(self.device)

        if batch_s_tensor.numel() != 0:
            target_q_values = self.add_target_net(batch_s__tensor).squeeze(2)
            max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
            targets = batch_r_tensor + self.ADD_GAMMA * max_target_q_values
            # Compute Q_values
            q_values = self.add_online_net(batch_s_tensor)
            a_q_values = torch.gather(input=q_values.squeeze(2), dim=1, index=batch_a_tensor)
            # Compute Loss
            loss = nn.functional.smooth_l1_loss(a_q_values, targets)
            # Gradient Descent
            self.add_optimizer.zero_grad()
            loss.backward()
            self.add_optimizer.step()

    def dlt_learn(self, reward_model=None):
        if not self.dlt_memo.can_sample():
            return
        batch_s, batch_a, batch_r, batch_s_, batch_node_emb, batch_node_next_emb = self.dlt_memo.sample()

        batch_s_tensor = torch.as_tensor(np.asarray(batch_s), dtype=torch.float32).to(self.device)
        batch_a_tensor = torch.as_tensor(np.asarray(batch_a), dtype=torch.int64).to(self.device)
        batch_s__tensor = torch.as_tensor(np.asarray(batch_s_), dtype=torch.float32).to(self.device)

        if reward_model:
            batch_emb_tensor = torch.as_tensor(np.asarray(batch_node_emb), dtype=torch.float32).to(self.device)
            batch_emb_next_tensor = torch.as_tensor(np.asarray(batch_node_next_emb), dtype=torch.float32).to(self.device)
            with torch.no_grad():
                batch_r_tensor = reward_model(batch_emb_next_tensor) - reward_model(batch_emb_tensor)
        else:
            batch_r_tensor = torch.as_tensor(np.asarray(batch_r), dtype=torch.float32).to(self.device)

        # Compute Targets
        if batch_s_tensor.numel() != 0:
            target_q_values = self.dlt_target_net(batch_s__tensor).squeeze(2)
            max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
            targets = batch_r_tensor + self.DLT_GAMMA * max_target_q_values
            # Compute Q_values
            q_values = self.dlt_online_net(batch_s_tensor)
            a_q_values = torch.gather(input=q_values.squeeze(2), dim=1, index=batch_a_tensor)
            # Compute Loss
            loss = nn.functional.smooth_l1_loss(a_q_values, targets)
            # Gradient Descent
            self.dlt_optimizer.zero_grad()
            loss.backward()
            self.dlt_optimizer.step()

    def update_target_model(self):
        self.add_target_net.load_state_dict(self.add_online_net.state_dict())
        self.dlt_target_net.load_state_dict(self.dlt_online_net.state_dict())

    def save_model(self, model_id, save_dir):
        torch.save(self.add_online_net.state_dict(), os.path.join(save_dir, f"add_{model_id}.pth"))
        torch.save(self.dlt_online_net.state_dict(), os.path.join(save_dir, f"dlt_{model_id}.pth"))


def add_sort(obj, args):
    sorted_keys = [key for key, _ in sorted(obj.items(), key=lambda x: x[1], reverse=True)]
    return sorted_keys[:min(len(sorted_keys), args.add_list)]


def dlt_sort(obj, args):
    sorted_keys = [key for key, _ in sorted(obj.items(), key=lambda x: x[1], reverse=True)]
    return sorted_keys[:min(len(sorted_keys), args.dlt_list)]


def my_remove_edge(next_nx_graph, agent_id, node):
    next_nx_graph.remove_edge(agent_id, node)
    if nx.is_connected(next_nx_graph):
        return True
    else:
        next_nx_graph.add_edge(agent_id, node)
        return False


class AgentManager:
    def __init__(self, agent_num, n_state, n_action, device, args):
        self.agents = [Agent(agent_id=i,
                             n_input=n_state,
                             n_output=n_action,
                             device=device,
                             mode='train' if args.is_train else "",
                             args=args) for i in range(agent_num)]

        self.agent_num = agent_num
        self.args = args
        self.device = device

        self.reward_model = None
        if args.is_irl:
            self.reward_model = self.create_irl_reward(is_train=args.is_train)

    def create_irl_reward(self, is_train=True):
        reward_model = RewardNet(self.args.node_emb_dims).to(self.device)
        if not is_train:
            reward_model.load_state_dict(torch.load(self.args.irl_save_path))
            reward_model.eval()
        else:
            self.reward_optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-3)
            self.agent_trajectories = []
            self.expert_trajectories = torch.from_numpy(load_expert_trajectories(self.args.irl_datasets_dir)).to(self.device)

        return reward_model
    
    def get_irl_reward(self, state):
        in_ = torch.as_tensor(state.copy(), dtype=torch.float32, device=self.device)
        return self.reward_model(in_).item()
    

    def negotiate(self, epsilon, cur_graph: MyGraph,
                  add_compare_func=None,
                  dlt_compare_func=None,
                  get_compare_vals=None,
                  get_reward_func=None):
        """
        ---------------------------------------------------------------
        |          args          |             instruction            |
        ---------------------------------------------------------------
        |        cur_graph       |             Environment            |
        |       compare_func     |    Threshold Comparison Method     |
        |     get_compare_vals   | Get the parameters for comparison  |
        |     get_reward_func    |       Agent reward function        |
        ---------------------------------------------------------------
        """
        assert callable(add_compare_func)
        assert callable(dlt_compare_func)
        assert callable(get_compare_vals)
        assert callable(get_reward_func)

        s_add, s_dlt, a_add_list, a_dlt_list= [], [], [], []
        add_q_values, dlt_q_values = {}, {}

        for agent_id, agent in enumerate(self.agents):
            """Get neighbors"""
            first_neighbors, second_neighbors = cur_graph.get_second_neighbor(agent_id)

            """Get add DQN input tensor, add agent must consider second-order neighbors"""
            second_obs = cur_graph.get_obs_tensor(node_id=agent_id, neighbor_list=second_neighbors, device=self.device)
            
            """Record state"""
            s_add.append(second_obs.cpu().numpy().tolist())

            """Sort and filter"""
            add_q_values[agent_id] = agent.add_act(second_neighbors, second_obs)
            a_add_list.append(add_sort(add_q_values[agent_id], self.args))

            """Similar to the above"""
            first_obs = cur_graph.get_obs_tensor(node_id=agent_id, neighbor_list=first_neighbors, device=self.device)

            s_dlt.append(first_obs.cpu().numpy().tolist())

            dlt_q_values[agent_id] = agent.dlt_act(first_neighbors, first_obs)
            a_dlt_list.append(dlt_sort(dlt_q_values[agent_id], self.args))
            
        a_add, a_dlt = [], []

        """Create a dictionary of candidate node lists"""
        candidate_add_dict = {}
        candidate_dlt_dict = {}
        is_random_add_list = [random.random() < epsilon for _ in range(self.agent_num)]
        is_random_dlt_list = [random.random() < epsilon for _ in range(self.agent_num)]
        for agent_id in range(self.agent_num):
            first_neighbors, second_neighbors = cur_graph.get_second_neighbor(agent_id)

            # add
            if is_random_add_list[agent_id]:
                candidate_add_list = random.sample(second_neighbors, min(len(second_neighbors), self.args.add_list))
            else:
                candidate_add_list = a_add_list[agent_id]

            # dlt
            if is_random_dlt_list[agent_id]:
                candidate_dlt_list = random.sample(first_neighbors, min(len(first_neighbors), self.args.dlt_list))
            else:
                candidate_dlt_list = a_dlt_list[agent_id]

            candidate_add_dict[agent_id] = candidate_add_list
            candidate_dlt_dict[agent_id] = candidate_dlt_list

        next_nx_graph = copy.deepcopy(cur_graph.get_nx_graph())

        add_node_vals, dlt_node_vals = get_compare_vals(graph=cur_graph.nx_graph, add_q_values=add_q_values, dlt_q_values=dlt_q_values)
        ALPHA, BETA = 1, 0
        for agent_id in range(self.agent_num):
            action_list_add = []

            for node in candidate_add_dict[agent_id]:
                if agent_id in candidate_add_dict[node]:
                    if (not is_random_add_list[agent_id] and
                            not add_compare_func(ALPHA * add_node_vals[node] + BETA * add_q_values[agent_id][node], self.args.add_thresh)):
                        # If not random and doesn't meet the threshold, prohibit
                        # otherwise, allow add action.
                        pass
                    else:
                        action_list_add.append(node)
                        if not next_nx_graph.has_edge(agent_id, node):
                            next_nx_graph.add_edge(agent_id, node)
            """Record action"""
            a_add.append(action_list_add)

            action_list_dlt = []
            for node in candidate_dlt_dict[agent_id]:
                if agent_id in candidate_dlt_dict[node]:
                    if (not is_random_dlt_list[agent_id] and
                            not dlt_compare_func(ALPHA * dlt_node_vals[node] + BETA * dlt_q_values[agent_id][node], self.args.dlt_thresh)):
                        pass
                    else:
                        action_list_dlt.append(node)
                        if (next_nx_graph.has_edge(agent_id, node) and my_remove_edge(next_nx_graph, agent_id, node)):
                            # my_remove_edge checks whether edge removal is valid.
                            # if valid, will remove the egde
                            pass
            a_dlt.append(action_list_dlt)
        
        next_graph = MyGraph(next_nx_graph, self.args)

        r_add = np.zeros(self.agent_num)
        r_dlt = np.zeros(self.agent_num)

        """Get Agent rewards"""
        cur_embs, next_embs = {}, {}
        if self.args.is_irl:
            cur_embs = cur_graph.get_node_emb_dict()
            next_embs = next_graph.get_node_emb_dict()
        add_rewards, dlt_rewards = get_reward_func(cur_graph=cur_graph.nx_graph, next_graph=next_graph.nx_graph, 
                                                   reward_model=self.reward_model, cur_embs=cur_embs, next_embs=next_embs)
        for agent_id in range(self.agent_num):
            r_add[agent_id] = add_rewards[agent_id]
            r_dlt[agent_id] = dlt_rewards[agent_id]

        """Get next state and fill the Agent's ReplayMemory"""
        for agent_id, agent in enumerate(self.agents):
            first_neighbors, second_neighbors = next_graph.get_second_neighbor(agent_id)

            second_obs = next_graph.get_obs_tensor(node_id=agent_id, neighbor_list=second_neighbors, device=self.device).cpu()
            first_obs = next_graph.get_obs_tensor(node_id=agent_id, neighbor_list=first_neighbors, device=self.device).cpu()

            if a_add[agent_id] and s_add[agent_id]:
                for i in a_add[agent_id]:
                    agent.add_memo.add(s_add[agent_id], i, r_add[agent_id], second_obs)

            if a_dlt[agent_id] and s_dlt[agent_id]:
                for i in a_dlt[agent_id]:
                    agent.dlt_memo.add(s_dlt[agent_id], i, r_dlt[agent_id], first_obs)

        return next_graph

    def agent_learn(self):
        for agent in self.agents:
            agent.add_learn(reward_model=self.reward_model)
            agent.dlt_learn(reward_model=self.reward_model)

    def update_target_model(self):
        for agent in self.agents:
            agent.update_target_model()

    def save_model(self, dqn_save_dir, irl_save_path=None):
        for agent_id, agent in enumerate(self.agents):
            agent.save_model(model_id=agent_id, save_dir=dqn_save_dir)
        if irl_save_path is not None and self.reward_model is not None:
            torch.save(self.reward_model.state_dict(), irl_save_path)

    def add_trajectory(self, cur_graph: MyGraph):
        for agent_id in range(self.agent_num):
            self.agent_trajectories[agent_id].append(cur_graph.get_node_emb(agent_id))

    def update_reward_model(self):
        expert_rewards = self.reward_model(self.expert_trajectories)

        # agent_trajectories shape is [N, times, emb_dims]
        agent_trajectories_array = np.array(self.agent_trajectories, dtype=np.float32)
        agent_traj_tensor = torch.tensor(agent_trajectories_array).to(self.device)
        
        agent_rewards = self.reward_model(agent_traj_tensor)

        loss = torch.mean(agent_rewards) - torch.mean(expert_rewards)

        l2_reg = torch.tensor(0.).to(self.device)
        for param in self.reward_model.parameters():
            l2_reg += torch.norm(param, p=2)

        loss += 0.01 * l2_reg

        self.reward_optimizer.zero_grad()
        loss.backward()
        self.reward_optimizer.step()

