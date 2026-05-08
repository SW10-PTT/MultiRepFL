PRINTS_SILENT = set()

PRINTS_EVEN_LESSER = {
    "round_boundary",
    "experiment_end",
}

PRINTS_LESS = {
    "round_boundary",
    "round_models",
    "round_rewards",
    "experiment_end",
}

PRINTS_ALL = {
    "setup_env",
    "setup_data",
    "setup_contracts",
    "round_boundary",
    "round_training",
    "round_models",
    "round_matrices",
    "round_scoring",
    "round_rewards",
    "agent_behavior",
    "experiment_end",
    "writer",
    "gas_report",
    "latex_output",
    "autorunner"
}

# PRINTS_MINIMAL = PRINTS_EVEN_LESSER
# PRINTS_NORMAL = PRINTS_LESS
# PRINTS_DEBUG = PRINTS_ALL

DEFAULT_ENABLED_PRINTS_CONFIG = PRINTS_ALL
