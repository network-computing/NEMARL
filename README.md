## NEMARL

The existing models ignore the fact that network evolution is inherently a group decision-making process, leading to two primary limitations: nodes lack the ability of policy self-learning, and there is no coordination among node policies. To address these shortcomings, and considering the effectiveness of multi-agent reinforcement learning in solving group decision-making tasks, we proposes a complex network model based on multi-agent reinforcement learning (NEMARL). This model achieves excellent performance in terms of reproducing classic network characteristics and fitting real network data.

### Model features

- Reproducing typical network properties: power law degree distribution, log-normal degree distribution, small world, positive degree correlation, negative degree correlation, community structure

- Fitting real network data: degree distribution, clustering coefficient, and average path length on static networks; degree distribution, link update quantity distribution on dynamic networks

- Scenario testing: similar degree, low closeness centrality and Pagerank/Degree

- Diverse reward functions: energy-based reward and inverse reinforcement learning-based reward


### Repository layout (partial)

- `main.py`: entry point; training or batched fitting by mode.
- `train.py`: one evaluation/training rollout; uses `AgentManager` and `MyGraph`.
- `argument.py`: CLI arguments and configs (static/dynamic data and output dirs).
- `model/agent.py`: DQN agents and multi-agent negotiation.
- `model/my_graph.py`: graph wrapper, neighborhood queries, embeddings, observations, metrics.
- `model/Energy.py`: energy functions for degree, small-world, and dynamic diffs.
- `model/fitter.py`: three fitters and CSV result aggregation.
- `model/emb_methods/`: `wavelet` and `s2v` implementations.
- `utils.py`: threshold/reward functions, metrics, utilities.
- `datasets/`: static and dynamic datasets.
- `output/`: CSV summaries and intermediate artifacts.

---

### Basic Usage

Train with default parameters:

```bash
python main.py
```

Train with custom parameters:

```bash
python main.py --init_graph_path ./init_graph/basic.gexf --add_list 20 --dlt_list 10 --add_thresh 0 --dlt_thresh 0.95
```

### IRL Mode

Enable Inverse Reinforcement Learning (IRL) mode for training:

```bash
python main.py --irl
```

### Feature Reproduction

- Power Law degree distribution

```bash
python main.py --init_graph_path ./init_graph/basic.gexf --add_list 3 --dlt_list 2 --add_thresh 0.015 --dlt_thresh 0.96
```

- Log-normal degree distribution

```bash
python main.py --init_graph_path ./init_graph/basic.gexf --add_list 5 --dlt_list 3 --add_thresh 0.01 --dlt_thresh 0.98
```

- Small world effect

```bash
python main.py --init_graph_path ./init_graph/sw_300.gexf --add_list 30 --dlt_list 15 --add_thresh 0 --dlt_thresh 0.95
```

- Positive degree correlation

```bash
python main.py --init_graph_path ./init_graph/basic.gexf --add_list 25 --dlt_list 10 --add_thresh 0 --dlt_thresh 0.94
```

- Negative degree correlation

```bash
python main.py --init_graph_path ./init_graph/basic.gexf --add_list 10 --dlt_list 10 --add_thresh 0.06 --dlt_thresh 0.84
```

- Community structure

```bash
python main.py --init_graph_path ./init_graph/modularity_300.gexf --add_list 20 --dlt_list 10 --add_thresh 0 --dlt_thresh 0.95
```

### Outputs

- Trained DQN models are saved in `output/out_models/dqn_models/basic/` or `output/out_models/dqn_models/irl/` directories
- In IRL mode, the reward network model is saved in `output/out_models/rewrad_models/reward_net.pth`
- Graph snapshots during evolution are saved in `output/graphs/.../graph_*.gexf`

---

## Real-world data fitting

Fitting experiment mode is used to evaluate the model's fitting capability on target network statistics without training (`--no_train`).

- Degree distribution fitting (static)

```bash
python main.py --no_train --mode static_degree
```

- Clustering coefficient and average path length fitting (static)

```bash
python main.py --no_train --mode static_sw
```

- Degree distribution fitting (dyamic)

```bash
python main.py --no_train --mode dynamic_degree
```

- Link update quantity distribution (dyamic)

```bash
python main.py --no_train --mode dynamic_luq
```

### Real data fitting in IRL mode

Run fitting experiments in IRL mode (using IRL-trained models):

```bash
python main.py --irl --no_train --mode static_sw
python main.py --irl --no_train --mode static_degree
python main.py --irl --no_train --mode dynamic_degree
python main.py --irl --no_train --mode dynamic_luq
```

### Scenario testing

- Similar degree

```bash
python main.py --reward_method similar_degree --compare_val_name sd --add_compare_method lower --dlt_compare_method upper --add_thresh 1 --dlt_thresh 1
```

- Low closeness centrality

```bash
python main.py --reward_method low_cc --compare_val_name lcc --add_compare_method lower --dlt_compare_method upper --add_thresh 0.2 --dlt_thresh 0.7
```

- Pagerank/Degree

```bash
python main.py --reward_method pg_degree --compare_val_name pg
```

### Outputs

Fitting experiment results are saved in the following locations:

- Degree distribution fitting (static): `output/fit_result/static/degree/`
- Clustering coefficient and average path length fitting (static): `output/fit_result/static/sw/`
- Degree distribution fitting (dyamic): `output/fit_result/dynamic/degree/`
- Link update quantity distribution (dyamic): `output/fit_result/dynamic/link_update_quantity/`

---

## Key Arguments

### General Arguments

- `--init_graph_path`: Path to initial GEXF graph file for training/evaluation (default: `./init_graph/basic.gexf`)
- `--out_graph_dir`: Path to save GEXF graph file during training/evaluation (default: `./output/graphs/`)
- `--mode`: Running mode, options: `static` | `static_sw` | `static_degree` | `dynamic_luq` | `dynamic_degree`
- `--total_episodes`: Total number of training episodes (default: 100)
- `--steps_per_episode`: Number of steps per episode (default: 10)
- `--no_train`: Disable training mode (enable evaluation/fitting mode)
- `--irl`: Enable IRL mode

### Embedding Arguments

- `--node_emb_dims`: Dimension of node embedding vectors (default: 32)
- `--emb_method`: Graph embedding method, options: `wavelet` or `s2c` (default: `wavelet`)

### Negotiation Mechanism Arguments

- `--add_thresh`($\alpha_i^\mathrm{add}$)`: Upper limit to the number of new added new links (default: 20)
- `--dlt_list`($\alpha_i^\mathrm{delete}$): Upper limit to the number of deleted old links (default: 10)
- `--add_thresh`($\beta_i^\mathrm{add}$): Threshold of filtering low-quality links for link add action (default: 0)
- `--dlt_thresh`($\beta_i^\mathrm{delete}$): Threshold of filtering low-quality links for link delete action (default: 0.95)
- `--compare_method`: Threshold comparison method, options: upper or lower (default: upper)

### Reward Function Arguments

- `--reward_method`: Reward function computation method (default: `pg`)
- `--rf_val_name`: Reward function variant name (default: `pg`)

---

## Data & Outputs

### Datasets

- **Static datasets**: `datasets/static/*` (supports `.mtx`, `.edges`, `.gexf` formats, etc.)
- **Dynamic datasets**: Time-ordered `.gexf` files in `datasets/dynamic/link_update_quantity/<NAME>` directories (e.g., `PRB_1990-2000`) or `.txt` files in `datasets/dynamic/degree/`
- **IRL datasets**: Time-series data in `datasets/irl/` directory

### Output Files

- **Trained models**:
  - `output/out_models/dqn_models/basic/`: DQN models in basic training mode
  - `output/out_models/dqn_models/irl/`: DQN models in IRL mode
  - `output/out_models/rewrad_models/reward_net.pth`: IRL reward network model
- **Evolved graphs**: `output/graphs/.../graph_*.gexf`
- **Fitting result CSVs**:
  - `output/fit_result/static/sw/sw_summary_basic.csv` or `sw_summary_irl.csv`
  - `output/fit_result/static/degree/degree_summary_basic.csv` or `degree_summary_irl.csv`
  - `output/fit_result/dynamic/link_update_quantity/<NAME>/dynamic_luq_summary_basic.csv` or `dynamic_luq_summary_irl.csv`
  - `output/fit_result/dynamic/degree/dynamic_degree_summary_basic.csv` or `dynamic_degree_summary_irl.csv`

---

## Typical Workflow

1. **Load initial graph**: Read initial graph in GEXF format and normalize node IDs to consecutive integers starting from 0
2. **Compute embeddings**: Calculate node embeddings using `wavelet` or `s2v` methods and construct observation vectors (concatenation of node embedding with the sum of neighbor embeddings)
3. **Multi-agent negotiation**: Each node agent selects add/delete edge candidates via epsilon-greedy strategy, filters actions through mutual candidacy and thresholds while preserving graph connectivity
4. **Reward computation and learning**: Compute rewards (PageRank neighbor sum deltas), store transition experiences, and periodically update target networks
5. **Evaluation and fitting**: Return graph metrics for evaluation; energy functions compute objective values and use simulated annealing to search for optimal parameters

---

## FAQ

### Dynamic Data Preparation

Provide a time-ordered sequence of `.gexf` files and ensure node IDs are consecutive integers (the project automatically checks continuity).

### Custom Compare/Reward Functions

Extend `compare_func_map`, `reward_func_map`, and `compare_vals_func_map` in `utils.py` to customize comparison and reward functions.

---

## License & Citation

If you use this project in academic or industrial work, please acknowledge this repository (NEMARL) and the authors of the public datasets and libraries employed.

## Contact
If you have any questions about this project, please contact dongli@sdu.edu.cn or socialcomputing109@gmail.com.