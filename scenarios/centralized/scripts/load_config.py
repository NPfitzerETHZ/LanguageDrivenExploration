# Function to load kwargs
from vmas.simulator.utils import ScenarioUtils
def load_scenario_config(kwargs, env):
        
    # === Map & Scenario Layout ===
    env.x_semidim = kwargs.pop("x_semidim", 1.0)
    env.y_semidim = kwargs.pop("y_semidim", 1.0)
    env._covering_range = kwargs.pop("covering_range", 0.15)
    env._lidar_range = kwargs.pop("lidar_range", 0.15)
    env.agent_radius = kwargs.pop("agent_radius", 0.16)
    env.n_obstacles = kwargs.pop("n_obstacles", 10)

    # === Agent/Target Counts & Behavior ===
    env.n_agents = kwargs.pop("n_agents", 6)
    env._agents_per_target = kwargs.pop("agents_per_target", 1)
    env.n_targets_per_class = kwargs.pop("n_targets_per_class", 1)
    env.n_target_classes = kwargs.pop("n_target_classes", 2)
    env.n_targets = env.n_target_classes * env.n_targets_per_class
    env.done_at_termination = kwargs.pop("done_at_termination", True)

    # === Rewards ===
    env.shared_target_reward = kwargs.pop("shared_target_reward", True)
    env.shared_final_reward = kwargs.pop("shared_final_reward", True)
    env.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -0.5)
    env.obstacle_collision_penalty = kwargs.pop("obstacle_collision_penalty", -0.5)
    env.covering_rew_coeff = kwargs.pop("covering_rew_coeff", 5.0)
    env.false_covering_penalty_coeff = kwargs.pop("false_covering_penalty_coeff", -0.25)
    env.time_penalty = kwargs.pop("time_penalty", -0.05)
    env.terminal_rew_coeff = kwargs.pop("terminal_rew_coeff", 15.0)
    env.exponential_search_rew = kwargs.pop("exponential_search_rew_coeff", 1.5)
    env.termination_penalty_coeff = kwargs.pop("termination_penalty_coeff", -5.0)

    # === Exploration Rewards ===
    env.use_expo_search_rew = kwargs.pop("use_expo_search_rew", True)
    env.grid_visit_threshold = kwargs.pop("grid_visit_threshold", 3)
    env.exploration_rew_coeff = kwargs.pop("exploration_rew_coeff", -0.05)
    env.new_cell_rew_coeff = kwargs.pop("new_cell_rew_coeff", 0.0)
    env.heading_exploration_rew_coeff = kwargs.pop("heading_exploration_rew_coeff", 20.0)

    # === Lidar & Sensing ===
    env.use_lidar = kwargs.pop("use_lidar", False)
    env.use_target_lidar = kwargs.pop("use_target_lidar", False)
    env.use_agent_lidar = kwargs.pop("use_agent_lidar", False)
    env.use_obstacle_lidar = kwargs.pop("use_obstacle_lidar", False)
    env.n_lidar_rays_entities = kwargs.pop("n_lidar_rays_entities", 8)
    env.n_lidar_rays_agents = kwargs.pop("n_lidar_rays_agents", 12)
    env.use_velocity_controller = kwargs.pop("use_velocity_controller", True)
    env.max_agent_observation_radius = kwargs.pop("max_agent_observation_radius", 0.4)
    env.prediction_horizon_steps = kwargs.pop("prediction_horizon_steps", 1)

    # === Agent Communication & GNNs ===
    env.use_gnn = kwargs.pop("use_gnn", False)
    env.comm_dim = kwargs.pop("comm_dim", 1)
    env._comms_range = kwargs.pop("comms_radius", 0.35)

    # === Observation Settings ===
    env.observe_grid = kwargs.pop("observe_grid", True)
    env.observe_targets = kwargs.pop("observe_targets", True)
    env.observe_pos_history = kwargs.pop("observe_pos_history", True)
    env.observe_vel_history = kwargs.pop("observe_vel_history", False)
    env.use_grid_data = kwargs.pop("use_grid_data", True)
    env.use_class_data = kwargs.pop("use_class_data", True)
    env.use_max_targets_data = kwargs.pop("use_max_targets_data", True)
    env.use_confidence_data = kwargs.pop("use_confidence_data", False)

    # === Grid Settings ===
    env.num_grid_cells = kwargs.pop("num_grid_cells", 400)
    env.mini_grid_radius = kwargs.pop("mini_grid_radius", 1)
    env.plot_grid = True

    # === Movement & Dynamics ===
    env.agent_weight = kwargs.pop("agent_weight", 1.0)
    env.agent_v_range = kwargs.pop("agent_v_range", 1.0)
    env.agent_a_range = kwargs.pop("agent_a_range", 1.0)
    env.min_collision_distance = kwargs.pop("min_collision_distance", 0.1)
    env.linear_friction = kwargs.get("linear_friction", 0.1)
    env.agent_f_range = env.agent_a_range + env.linear_friction
    env.agent_u_range = env.agent_v_range if env.use_velocity_controller else env.agent_f_range

    # === Histories ===
    env.history_length = kwargs.pop("history_length", 2)
    env.pos_history_length = env.history_length
    env.pos_dim = 2
    env.vel_history_length = env.history_length
    env.vel_dim = 2

    # === Language & LLM Goals ===
    env.embedding_size = kwargs.pop("embedding_size", 1024)
    env.llm_activate = kwargs.pop("llm_activate", True)

    # === External Inputs ===
    env.data_json_path = kwargs.pop("data_json_path", "")
    env.decoder_model_path = kwargs.pop("decoder_model_path", "")
    env.use_decoder = kwargs.pop("use_decoder", False)

    # === Visuals ===
    env.viewer_zoom = 1
    
    # Final check
    ScenarioUtils.check_kwargs_consumed(kwargs)
    

def load_scenario_config_yaml(config, env):
    cfg = config["task_config"].value

    # === Map & Scenario Layout ===
    env.task_x_semidim = cfg.x_semidim
    env.task_y_semidim = cfg.y_semidim
    env._covering_range = cfg.covering_range
    env.agent_radius = cfg.agent_radius
    env.n_obstacles = cfg.n_obstacles

    # === Agent/Target Counts & Behavior ===
    env.n_agents = cfg.n_agents
    env._agents_per_target = cfg.agents_per_target
    env.n_targets_per_class = cfg.n_targets_per_class
    env.n_target_classes = cfg.n_target_classes
    env.n_targets = env.n_target_classes * env.n_targets_per_class
    env.done_at_termination = cfg.done_at_termination

    # === Rewards ===
    env.shared_target_reward = cfg.shared_target_reward
    env.shared_final_reward = cfg.shared_final_reward
    env.agent_collision_penalty = cfg.agent_collision_penalty
    env.obstacle_collision_penalty = cfg.obstacle_collision_penalty
    env.covering_rew_coeff = cfg.covering_rew_coeff
    env.false_covering_penalty_coeff = cfg.false_covering_penalty_coeff
    env.time_penalty = cfg.time_penalty
    env.terminal_rew_coeff = cfg.terminal_rew_coeff
    env.exponential_search_rew = cfg.exponential_search_rew_coeff
    env.termination_penalty_coeff = cfg.termination_penalty_coeff

    # === Exploration Rewards ===
    env.use_expo_search_rew = cfg.use_expo_search_rew
    env.grid_visit_threshold = cfg.grid_visit_threshold
    env.exploration_rew_coeff = cfg.exploration_rew_coeff
    env.new_cell_rew_coeff = cfg.new_cell_rew_coeff
    env.heading_exploration_rew_coeff = cfg.heading_exploration_rew_coeff

    # === Lidar & Sensing ===
    env.use_lidar = cfg.use_lidar
    env.n_lidar_rays_entities = cfg.n_lidar_rays_entities
    env.n_lidar_rays_agents = cfg.n_lidar_rays_agents
    env.use_velocity_controller = cfg.use_velocity_controller
    env.max_agent_observation_radius = cfg.max_agent_observation_radius
    env.prediction_horizon_steps = cfg.prediction_horizon_steps

    # === Agent Communication & GNNs ===
    env.use_gnn = cfg.use_gnn
    env.comm_dim = cfg.comm_dim
    env._comms_range = cfg.comms_radius

    # === Observation Settings ===
    env.observe_grid = cfg.observe_grid
    env.observe_targets = cfg.observe_targets
    env.observe_pos_history = cfg.observe_pos_history
    env.observe_vel_history = cfg.observe_vel_history
    env.use_grid_data = cfg.use_grid_data
    env.use_class_data = cfg.use_class_data
    env.use_max_targets_data = cfg.use_max_targets_data
    env.use_confidence_data = cfg.use_confidence_data

    # === Grid Settings ===
    env.num_grid_cells = cfg.num_grid_cells
    env.mini_grid_radius = cfg.mini_grid_radius
    env.plot_grid = True

    # === Movement & Dynamics ===
    env.agent_weight = cfg.agent_weight
    env.agent_v_range = cfg.agent_v_range
    env.agent_a_range = cfg.agent_a_range
    env.min_collision_distance = cfg.min_collision_distance
    env.linear_friction = cfg.linear_friction
    env.agent_f_range = env.agent_a_range + env.linear_friction
    env.agent_u_range = env.agent_v_range if env.use_velocity_controller else env.agent_f_range

    # === Histories ===
    env.history_length = cfg.history_length
    env.pos_history_length = cfg.history_length
    env.pos_dim = 2
    env.vel_history_length = cfg.history_length
    env.vel_dim = 2

    # === Language & LLM Goals ===
    env.embedding_size = cfg.embedding_size
    env.llm_activate = cfg.llm_activate

    # === External Inputs ===
    env.data_json_path = cfg.data_json_path
    env.decoder_model_path = cfg.decoder_model_path
    env.use_decoder = cfg.use_decoder

    # === Visuals ===
    env.viewer_zoom = 1
