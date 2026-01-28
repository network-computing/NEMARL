import torch
import networkx as nx
import copy
from tqdm import tqdm

from model.agent import AgentManager
from model.my_graph import MyGraph
from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(args, train_mode='train'):
    assert train_mode in ['train', 'sw', 'degree', 'dynamic_degree', 'dynamic_luq', 'dynamic_lcc_eff']
    """Read the init graph"""
    if train_mode in ['sw', 'degree']:
        node_num = args.num_of_node
        init_graph = nx.generators.random_graphs.watts_strogatz_graph(node_num, args.degree, 0)
    else:
        init_graph = nx.read_gexf(args.init_graph_path, node_type=int)
        node_num = init_graph.number_of_nodes()
        args.num_of_node = node_num

    init_graph = check_graph_node_id_from_zero(init_graph)

    """Get DQN Input and Output Dimensions"""
    node_emb_dims = args.node_emb_dims
    n_state = node_emb_dims * 2
    n_action = 1

    """AgentManager handles interactions between agents, as well as model training and saving."""
    agent_manager = AgentManager(agent_num=node_num, n_state=n_state, n_action=n_action, device=device, args=args)

    """Set RL-related parameters"""
    total_episodes = args.total_episodes
    steps_per_episode = args.steps_per_episode
    epsilon = args.epsilon_initial
    epsilon_final = args.epsilon_final
    epsilon_decay_rate = args.epsilon_decay_rate

    pbar = tqdm(total=total_episodes)

    cur_nx_graph = copy.deepcopy(init_graph)

    if train_mode == 'dynamic_lcc_eff':
        s_lcc_series = []
        e_glob_series = []
        n0 = cur_nx_graph.number_of_nodes()
        if n0 > 0:
            lcc_size = len(max(nx.connected_components(cur_nx_graph), key=len)) if not nx.is_connected(cur_nx_graph) else n0
            s_lcc_series.append(lcc_size / n0)
            e_glob_series.append(nx.global_efficiency(cur_nx_graph))
        else:
            s_lcc_series.append(0.0)
            e_glob_series.append(0.0)
    
    if train_mode == 'dynamic_luq':
        static_graphs = [cur_nx_graph]

    if args.is_train and args.is_irl:
        args.reward_method = "irl"
        cur_graph = MyGraph(cur_nx_graph, args)
        agent_init_trajectories = [ [cur_graph.get_node_emb(i)] for i in range(node_num)]

    os.makedirs(args.out_graph_dir, exist_ok=True)
    os.makedirs(args.dqn_save_dir, exist_ok=True)
    """Start training"""
    for episode_i in range(total_episodes):
        pbar.set_description('Process')

        if args.is_train and args.is_irl:
            agent_manager.agent_trajectories = copy.deepcopy(agent_init_trajectories)

        cur_graph = MyGraph(cur_nx_graph, args)

        for step_i in range(steps_per_episode):
            epsilon = max(epsilon_final, epsilon * epsilon_decay_rate) if args.is_train else 0

            cur_graph = agent_manager.negotiate(epsilon, cur_graph,
                                                add_compare_func=get_compare_function(args.add_compare_method),
                                                dlt_compare_func=get_compare_function(args.dlt_compare_method),
                                                get_compare_vals=get_compare_vals_function(args.compare_val_name),
                                                get_reward_func=get_reward_function(args.reward_method))

            if args.is_train and args.is_irl:
                agent_manager.add_trajectory(cur_graph)

            if train_mode == 'dynamic_luq':
                static_graphs.append(cur_graph.get_nx_graph())
            if train_mode == 'dynamic_lcc_eff':
                nxg = cur_graph.get_nx_graph()
                n = nxg.number_of_nodes()
                if n > 0:
                    lcc_size = len(max(nx.connected_components(nxg), key=len)) if not nx.is_connected(nxg) else n
                    s_lcc_series.append(lcc_size / n)
                    e_glob_series.append(nx.global_efficiency(nxg))
                else:
                    s_lcc_series.append(0.0)
                    e_glob_series.append(0.0)

            if (episode_i + 1) % 5 == 0:
                out_graph_path = os.path.join(args.out_graph_dir, f"graph_{step_i}.gexf")
                cur_graph.save_gexf(out_graph_path)

        if args.is_train:
            if args.is_irl:
                agent_manager.update_reward_model()
            agent_manager.agent_learn()

        """Update target DQN"""
        if args.is_train and (episode_i + 1) % 2 == 0:
            agent_manager.update_target_model()

        pbar.update()

        """Save Agent DQN model"""
        if args.is_train and (episode_i + 1) % 20 == 0:
            agent_manager.save_model(dqn_save_dir=args.dqn_save_dir, irl_save_path=args.irl_save_path)
            
        if not args.is_train:
            fit_info = cur_graph.get_fit_info()
            if train_mode == 'dynamic_luq':
                degree_diffs = compute_degree_diffs(static_graphs)
                fit_info.update({
                    'degree_diffs': degree_diffs
                })
            if train_mode == 'dynamic_degree':
                fit_info.update({
                    'degree_distribution': nx.degree_histogram(cur_graph.nx_graph)
                })
            if train_mode == 'dynamic_lcc_eff':
                fit_info.update({
                    'S_LCC_series': s_lcc_series,
                    'E_glob_series': e_glob_series
                })
                
            return fit_info
    
