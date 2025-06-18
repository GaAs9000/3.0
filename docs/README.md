# Power Grid Partition

```
power_grid_partition_project/
├── main.py                Main Entry: Runs the entire project workflow, connecting all modules.
├── data_processing.py     Data Processing: Loads and cleans raw power grid data, constructs graph format.
├── models.py              Core Models: Defines "Physics-Guided Graph Attention Network (GAT)" for learning node representations.
├── env.py                 Simulation Environment (Curriculum Learning): Implements curriculum learning, wraps base environment to dynamically adjust difficulty.
├── metrics.py             Simulation Environment (Base & Evaluation): Defines reinforcement learning base environment and partition evaluation metrics.
├── agent.py               Agent: Implements PPO reinforcement learning algorithm, including decision network (Actor) and evaluation network (Critic).
├── training.py            Training Script: Responsible for running reinforcement learning training loops.
├── baseline.py            Baseline Methods: Contains traditional partition methods (such as spectral clustering) for comparison with our AI methods.
├── visualization.py       # 🎨 Visualization: Used to display partition results and training process in graphical form.
└── utils.py               # 🛠️ Utility Library: Stores common auxiliary functions, such as setting random seeds, configuring environment, etc.
```

## Detailed File Functions:ition

*   **`main.py`**:
    *   **Core Function**: Project launcher and orchestrator.
    *   **Specific Functions**:
        *   Initialize global settings (such as selecting CPU or GPU).
        *   Call core functions of other modules in sequence, completing the entire workflow from data loading, model construction, environment setup, agent training, baseline comparison to result visualization.
        *   Unified handling of top-level errors and log output during program execution.

*   **`data_processing.py`**:
    *   **Core Function**: Power grid data preprocessing and graph structure construction.
    *   **Specific Functions**:
        *   `PowerGridDataProcessor` class:
            *   Read raw power grid data in standard formats like MATPOWER.
            *   Clean data, such as handling missing values.
            *   Extract node (bus) features: active/reactive load, voltage, node degree, node type (PQ, PV, Slack), etc.
            *   Extract edge (branch) features: resistance, reactance, admittance, line capacity, whether it's a transformer, etc.
            *   Extract generator features and associate them with corresponding nodes.
            *   Standardize extracted features (such as using `RobustScaler`).
            *   Convert processed data to `torch_geometric.data.Data` objects for graph neural network use.
        *   `generate_random_power_grid`: A helper function to generate random power grid data for testing convenience.

*   **`models.py`**:
    *   **Core Function**: Define graph neural network models, particularly Physics-Guided GAT for node embedding learning.
    *   **Specific Functions**:
        *   `PhysicsGuidedGATConv` class:
            *   Custom GAT convolution layer, inheriting from `torch_geometric.nn.GATConv`.
            *   **Innovation**: When calculating attention weights, incorporates physical characteristics of power grids (such as line impedance/admittance), enabling nodes with closer physical connections to have higher attention. Includes a learnable `temperature` parameter to adjust the influence of physical priors.
        *   `EnhancedGATEncoder` class:
            *   Constructs a multi-layer GAT encoder using `PhysicsGuidedGATConv` as the core layer.
            *   Integrates residual connections, layer normalization, adaptive Dropout and other techniques to improve model performance and training stability.
            *   Optionally adds positional encoding (based on graph Laplacian eigenvectors) to help the model understand the global structural role of nodes.

*   **`env.py`**:
    *   **Core Function**: Implement curriculum learning mechanism, wrapping the base RL environment.
    *   **Specific Functions**:
        *   `CurriculumLearningEnv` class:
            *   Wraps a base `PowerGridPartitionEnv` instance (defined in `metrics.py`).
            *   Dynamically adjusts task difficulty: starts with simpler tasks (such as preset partial node partitions), gradually increases difficulty as agent performance improves (such as tightening constraints, reducing presets).
            *   Adjusts difficulty levels by monitoring agent success rate, aiming to accelerate training convergence and improve final performance.

*   **`metrics.py`**:
    *   **Core Function**: Define reinforcement learning base environment (`PowerGridPartitionEnv`) and evaluation metrics for partition quality.
    *   **Specific Functions**:
        *   `PartitionMetrics` dataclass: Used to store a complete set of partition evaluation metrics.
        *   `PowerGridPartitionEnv` class:
            *   **State**: Represents current power grid partition status, may include node assignment vector `z`, boundary node information, regional embedding representations, global context information, etc.
            *   **Action**: Defines operations the agent can perform, typically assigning an unassigned boundary node to an existing partition.
            *   **Reward**: Designs multi-objective reward function, comprehensively considering load balance (such as CV coefficient, Gini coefficient), decoupling degree (electrical coupling of inter-regional connection lines), intra-regional connectivity, power balance, partitioning progress, etc.
            *   **Environment Dynamics**: Implements `reset` method (initialize environment, select seed nodes to start partitioning) and `step` method (execute action, update state, calculate reward, determine if finished).
            *   **Metric Calculation**: Contains methods to calculate various detailed partition metrics, such as:
                *   `load_cv`, `load_gini`: Measure load balance among partitions.
                *   `total_coupling`: Total coupling strength of inter-regional lines.
                *   `inter_region_lines`: Number of inter-regional tie lines.
                *   `connectivity`: Ensure each partition is internally connected.
                *   `power_balance`: Degree of generation-load matching within regions.
                *   `modularity`: Graph theory metric for measuring community division quality.

*   **`agent.py`**:
    *   **Core Function**: Implement PPO reinforcement learning agent.
    *   **Specific Functions**:
        *   `PPOMemory` class: Used to store experiences generated during agent-environment interaction (states, actions, rewards, probabilities, etc.), and supports GAE (Generalized Advantage Estimation) calculation.
        *   `HierarchicalActor` class (Policy Network):
            *   Decides which action to take based on current state (i.e., how to assign the next node).
            *   May adopt hierarchical decision-making to handle complex action spaces.
        *   `Critic` class (Value Network):
            *   Evaluates the value of the current state (i.e., expected cumulative reward obtainable from the current state).
        *   `PPOAgent` class:
            *   Integrates Actor and Critic networks.
            *   Implements `select_action` method, selects actions based on policy network and interacts with environment.
            *   Implements `update` method, uses collected experience data to update Actor and Critic network parameters according to PPO algorithm optimization objectives (including clipped probability ratio, value function loss, and entropy bonus).
            *   Includes model saving and loading functionality.

*   **`training.py`**:
    *   **Core Function**: Manage and execute reinforcement learning training process.
    *   **Specific Functions**:
        *   `train_ppo` function:
            *   Organizes main training loop: conducted over multiple episodes.
            *   In each episode, agent interacts with environment (`env.step()`), collecting experiences.
            *   Periodically calls `agent.update()` to train the agent.
            *   Records various data during training, such as episode rewards, partition metrics, loss function values, etc.
            *   Supports using TensorBoard for training process visualization and monitoring.
            *   Periodically saves trained models.

*   **`baseline.py`**:
    *   **Core Function**: Implement traditional, non-AI power grid partition methods as performance comparison baselines for AI methods.
    *   **Specific Functions**:
        *   `BaselinePartitioner` class:
            *   Implements multiple classic partition algorithms, such as:
                *   Spectral clustering (`_spectral_partition`): Based on graph Laplacian matrix eigenvectors for clustering.
                *   K-means clustering (`_kmeans_partition`): Based on node embedding vectors for clustering.
                *   Random partitioning (`_random_partition`): As the simplest comparison.
        *   `evaluate_partition_method`: Uses metrics defined in `metrics.py` to evaluate partition results generated by these baseline methods.
        *   `compare_methods`: Uniformly calls AI methods and various baseline methods for partitioning, collects their performance metrics, and generates comparison tables.

*   **`visualization.py`**:
    *   **Core Function**: Display abstract partition results and training data in intuitive graphical form.
    *   **Specific Functions**:
        *   `visualize_partition`:
            *   Draws power grid partition results as network graphs: nodes colored by their assigned partitions, inter-regional tie lines highlighted.
            *   Can simultaneously display related partition metrics, regional load distribution bar charts, inter-regional coupling matrix heatmaps, etc.
        *   `plot_training_curves`:
            *   Plots metric change curves during training, such as reward curves, loss function curves, load balance metric curves, etc., helping analyze training effectiveness and model convergence.
        *   `create_interactive_visualization` (optional): May use libraries like Plotly to create interactive visualization charts.

*   **`utils.py`**:
    *   **Core Function**: Provide common auxiliary functions and configurations for the project.
    *   **Specific Functions**:
        *   Centralized import of commonly used standard libraries and third-party libraries.
        *   `set_seed`: Set global random seed (Python built-in random, NumPy, PyTorch) to ensure experimental result reproducibility.
        *   Configure computing device (automatically select GPU or CPU).
        *   Create output directories needed for project runtime (such as `models/`, `figures/`, `logs/`).
        *   May include other common utility functions.


```
使用方法：

# 快速演示（默认）
python main.py

# 快速演示（明确指定）
python main.py --mode quick

# 完整训练
python main.py --mode full
```