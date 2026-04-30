PRINTS_SILENT = set()

PRINTS_MINIMAL = {
    "round_start",
    "round_end",
    "experiment_end",
}

PRINTS_NORMAL = {
    "round_start",
    "round_end",
    "experiment_end",
    "slot_registration",
    "weights_submission",
    "contribution_score",
    "round_lifecycle",
}

PRINTS_DEBUG = {
    "env_info",
    "connection_info",
    "writer_info",
    "writer_debug",
    "experiment_summary",
    "account_init",
    "contract_deploy",
    "account_registration",
    "contract_info",
    "gpu_info",
    "pytorch_model_created",
    "data_loaded",
    "participant_added",
    "data_split",
    "data_split_labels",
    "round_start",
    "training_banner",
    "training_mode",
    "malicious_behavior",
    "attitude_switch",
    "freerider_behavior",
    "user_training",
    "merge_result",
    "model_exchange",
    "model_verify",
    "evaluation",
    "round_matrices",
    "slot_registration",
    "weights_submission",
    "feedback",
    "round_lifecycle",
    "contribution_score",
    "contribution_score_detail",
    "round_end",
    "rewards",
    "punishments",
    "round_reputation",
    "shapley_warnings",
    "model_termination",
    "experiment_end",
    "gas_report",
    "latex_output",
}

DEFAULT_ENABLED_PRINTS_CONFIG = PRINTS_MINIMAL
