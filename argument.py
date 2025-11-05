import argparse
from dataclasses import dataclass


def get_args():
    parser = argparse.ArgumentParser("nemarl")
    parser.add_argument('--init_graph_path', type=str, default="./init_graph/basic.gexf", help="Initial network used for training")
    parser.add_argument('--out_graph_dir', type=str, default="./output/graphs/", help="Save path for gexf files")
    parser.add_argument("--degree", type=int, default=5, help="Degree parameters for providing initial graphs for fitting static networks")
    parser.add_argument("--add_list", type=int, default=20, help="Scope of the ADD Consultative Mechanism ")
    parser.add_argument("--dlt_list", type=int, default=10, help="Scope of the Delete Consultative Mechanism ")
    parser.add_argument("--add_thresh", type=float, default=0, help="Threshold of the Add Consultative Mechanism ")
    parser.add_argument("--dlt_thresh", type=float, default=0.95, help="Threshold of the Delete Consultative Mechanism ")
    parser.add_argument("--steps_per_episode", type=int, default=10, help="Number of steps in episode")
    parser.add_argument("--total_episodes", type=int, default=100, help="Number of episodes")

    parser.add_argument("--epsilon_initial", type=float, default=0.01,
                        help="Initial value of epsilon for the epsilon-greedy exploration strategy. "
                         "Controls the starting probability of taking a random action.")

    parser.add_argument("--epsilon_final", type=float, default=0.01,
                        help="Final (minimum) value of epsilon after decay. "
                            "Determines the lowest exploration rate allowed during training.")

    parser.add_argument("--epsilon_decay_rate", type=float, default=0.995,
                        help="Decay rate applied to epsilon after each episode or step. "
                            "Defines how quickly the exploration rate decreases from epsilon_initial to epsilon_final.")

    parser.add_argument("--node_emb_dims", type=int, default=32,
                        help="Dimension of the node embedding vectors used to represent each node in the graph.")

    parser.add_argument("--emb_method", type=str, default="wavelet",
                        help="Graph embedding method to be used. Options include 'wavelet', 's2c(struct2vec)'")

    parser.add_argument("--add_compare_method", type=str, default="upper",
                        help="Method for comparing node or graph representations, e.g., 'upper' for upper-bound comparison or "
                            "other strategies depending on the experiment design.")
    
    parser.add_argument("--dlt_compare_method", type=str, default="upper",
                        help="Method for comparing node or graph representations, e.g., 'upper' for upper-bound comparison or "
                            "other strategies depending on the experiment design.")

    parser.add_argument("--reward_method", type=str, default="pg",
                        help="Reward function computation method. Common choices include 'pg' (policy gradient), "
                            "'mm' (maximum margin), or other reinforcement learning-based approaches.")

    parser.add_argument("--compare_val_name", type=str, default="pg",
                        help="Name or identifier for the reward function variant being used or saved (e.g., 'pg' for policy gradient).")

    parser.add_argument("--dqn_save_dir", type=str, default="./output/out_models/dqn_models/",
                        help="Directory path where trained DQN (Deep Q-Network) models will be saved.")

    parser.add_argument("--mode", type=str, default="static",
                        help="Running mode of the experiment. Options may include 'static' for fixed graphs "
                            "or 'dynamic' for time-evolving graphs.")

    parser.add_argument("--no_train", action="store_false", dest="is_train",
                       help="Disable training mode (enable evaluation mode)")
    
    parser.add_argument("--irl", action="store_true", dest="is_irl",
                       help="Enable irl mode")
    
    parser.add_argument("--irl_save_path", type=str, default="./output/out_models/reward_models/reward_net.pth",
                        help="Path to save the trained inverse reinforcement learning reward model")

    parser.add_argument("--irl_datasets_dir", type=str, default="./datasets/irl/",
                        help="Directory containing datasets for inverse reinforcement learning training")

    args = parser.parse_args()

    return args


@dataclass
class DegreeFitConfig:
    data_dir: str = "./datasets/static"
    summary_dir: str = "./output/fit_result/static/degree"


@dataclass
class SWFitConfig:
    data_dir: str = "./datasets/static"
    summary_dir: str = "./output/fit_result/static/sw"


@dataclass
class DynamicLuqConfig:
    data_dir: str = "./datasets/dynamic/link_update_quantity/PRB_1990-2000"
    summary_dir: str = "./output/fit_result/dynamic/link_update_quantity/PRB_1990-2000"


@dataclass
class DynamicDegreeConfig:
    data_dir: str = "./datasets/dynamic/degree"
    summary_dir: str = "./output/fit_result/dynamic/degree"

