# config_loader_condensed.py
from collections.abc import Mapping
from vmas.simulator.utils import ScenarioUtils

# --------------------------------------------------------------------------- #
# 1. PARAMS table
# --------------------------------------------------------------------------- #
PARAMS = [
    # --- Map & Layout -------------------------------------------------------
    ("x_semidim", 1.0), ("y_semidim", 1.0),
    ("lidar_range", 0.15),
    ("agent_radius", 0.16), ("n_obstacles", 10),

    # --- Agent / Targets ----------------------------------------------------
    ("n_agents", 6), ("_agents_per_target", 1),
    ("n_targets_per_class", 1), ("n_target_classes", 2),
    ("done_at_termination", True),

    # --- Rewards ------------------------------------------------------------
    ("reward_scale_factor", 0.1),
    ("shared_target_reward", True), ("shared_final_reward", True),
    ("agent_collision_penalty", -0.5), ("obstacle_collision_penalty", -0.5),
    ("covering_rew_coeff", 5.0), ("false_covering_penalty_coeff", -0.25),
    ("time_penalty", -0.05), ("terminal_rew_coeff", 15.0),
    ("exponential_search_rew", "exponential_search_rew_coeff", 1.5),
    ("termination_penalty_coeff", -5.0),

    # --- Exploration --------------------------------------------------------
    ("use_expo_search_rew", True), ("grid_visit_threshold", 3),
    ("exploration_rew_coeff", -0.05), ("new_cell_rew_coeff", 0.0),
    ("heading_exploration_rew_coeff", 20.0),

    # --- Lidar & Sensing ----------------------------------------------------
    ("use_lidar", False), ("use_target_lidar", False),
    ("use_agent_lidar", False), ("use_obstacle_lidar", False),
    ("n_lidar_rays_entities", 8), ("n_lidar_rays_agents", 12),
    ("max_agent_observation_radius", 0.4), ("prediction_horizon_steps", 1),

    # --- Communication / GNN -----------------------------------------------
    ("use_gnn", False), ("use_conv_2d", False),
    ("comm_dim", 1), ("_comms_range", "comms_radius", 0.35),

    # --- Observation --------------------------------------------------------
    ("observe_grid", True), ("observe_targets", True),
    ("observe_agents", False), ("observe_pos_history", True),
    ("observe_vel_history", False), ("use_grid_data", True),
    ("use_class_data", True), ("use_max_targets_data", True),
    ("use_confidence_data", False), ("use_team_level_target_count", True),

    # --- Grid ---------------------------------------------------------------
    ("num_grid_cells", 400), ("mini_grid_radius", 1),

    # --- Dynamics -----------------------------------------------------------
    ("use_velocity_controller", True), ("use_kinematic_model", True),
    ("agent_weight", 1.0), ("agent_v_range", 1.0), ("agent_a_range", 1.0),
    ("min_collision_distance", 0.1), ("linear_friction", 0.1),

    # --- Histories ----------------------------------------------------------
    ("history_length", 2),
    ("max_steps", 250),

    # --- Language / LLM -----------------------------------------------------
    ("embedding_size", 1024), ("use_embedding_ratio", 1.0), ("llm_activate", True),

    # --- External paths -----------------------------------------------------
    ("data_json_path", ""), ("decoder_model_path", ""), ("use_decoder", False),
    
]

# Expand every entry to canonical (dest, key, default) form
PARAMS = [
    (dest, key, default)
    for entry in PARAMS
    for dest, key, default in [(
        # ────────────────────────────────────────────────
        # 3-tuples stay as-is
        entry if isinstance(entry, tuple) and len(entry) == 3 else
        # 2-tuples → (name, name, default)
        (entry[0], entry[0], entry[1]) if isinstance(entry, tuple) else
        # bare-string → (name, name, None)
        (entry, entry, None)
    )]
]

# --------------------------------------------------------------------------- #
# 2.  Generic loader
# --------------------------------------------------------------------------- #
def load_scenario_config(source, env):
    """
    Parameters
    ----------
    source : dict-like (kwargs) | object with attributes
    env    : your VMAS scenario environment
    """
    is_mapping = isinstance(source, Mapping)

    for dest_attr, key, default in PARAMS:
        if is_mapping:
            value = source.get(key, default)
        else:
            value = getattr(source, key, default)
        setattr(env, dest_attr, value)

    # --- derived attributes -------------------------------------------------
    env.n_targets = env.n_target_classes * env.n_targets_per_class
    env.agent_f_range = env.agent_a_range + env.linear_friction
    env.agent_u_range = (
        env.agent_v_range if env.use_velocity_controller else env.agent_f_range
    )
    env.pos_history_length = env.vel_history_length = env.history_length
    env.pos_dim = env.vel_dim = 2
    env.viewer_zoom = 1
    env.plot_grid   = True
