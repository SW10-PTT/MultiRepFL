from datetime import datetime
import sys
import multiprocessing as mp
from pathlib import Path
import experiment.experiment_runner as ExperimentRunner
from experiment.experiment_configuration import ExperimentConfiguration
from openfl.utils.async_writer import AsyncWriter
from experiment.helper import getPath
from openfl.api import globals

# Add the repo root to sys.path so `analysis` package is importable from here
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis import ExperimentLogger

config = ExperimentConfiguration(
    min_buy_in=int(1e18),
    max_buy_in=int(1e18),
    contribution_score_strategy="loss_tolerance_aware", # Options: dotproduct, naive, accuracy_loss, accuracy_only, loss_only, loss_tolerance_aware, loss_tolerance_snap
    loss_tolerance_pct=0.15, # ε = pct * avg_prev_loss; only used by loss_tolerance_* strategies
    use_outlier_detection=True,
    minimum_rounds=25,
    epochs=1,
    number_of_good_contributors=5,
    number_of_bad_contributors=3,
    number_of_freerider_contributors=1,
    force_merge_all=False,
    freerider_noise_scale=0.1,
    malicious_noise_scale=1.0,
    punish_factor=3,
    punish_factor_contrib=3,
    freerider_start_round=1,
    malicious_start_round=1,
    number_of_participants=9,
    dataset="mnist",
    data_percentages=None,
    label_rules=None,
    seed=42,
    user_seeds=None,
    allow_overlap=False,
    replication_factor=1.0,
    partition_strategy="global", # Options: global, per_user
    per_user_partitions=None
    #data_percentages=[30, 10, 15, 15, 10, 20],
    # 0: {"only_labels": [0, 1, 2, 3, 4]}
    # 0: {"flip_map": {4: 9}}
    # 0: {"only_labels": [0, 1, 2, 3, 4], "flip_map": {4: 9}}
    # label_rules={
    #     0: {"only_labels": [0, 1, 2, 3, 4]},
    #     1: {"only_labels": [0, 1, 2, 3, 4]},
    #     2: {"only_labels": [0, 1, 2, 3, 4]},
    #     3: {"only_labels": [5, 6, 7, 8, 9]},
    #     4: {"only_labels": [5, 6, 7, 8, 9]},
    #     5: {"only_labels": [5, 6, 7, 8, 9]},
    # },
)

# OVERSKRIV variabler her for testing. eksempel: config = ExperimentConfiguration(minimum_rounds=1), hvis du kun vil køre een round#DATASET = "cifar-10"
RESULTDATAFOLDER = Path(__file__).resolve().parent.joinpath("data/sample")
DATASET = "mnist"

OUTPUTHEADERS = [
    "round",
    "time",
    "globalAcc",
    "globalLoss",
    "GRS",
    "accAvgPerUser",
    "lossAvgPerUser",
    "rewards",
    "conctractBalanceRewards",
    "punishments",
    "contributionScores",
    "feedbackMatrix",
    "disqualifiedUsers",
    "userStatuses",
    "GasTransactions",
    ]

WRITERBUFFERSIZE = 200

def main():
    run()


def run():
    startTime = datetime.now().strftime("%d-%m-%y--%H_%M_%S")
    path = getPath(config, startTime, DATASET, RESULTDATAFOLDER)
    writer = None
    logger = None
    metadata = {**vars(config), "dataset": DATASET, "timestamp": startTime}
    flags = globals.ReplayMode._actively_replaying | globals.ReplayMode.HardPlayBack
    if (globals.reuse_runs & flags) != flags:
        writer = AsyncWriter(path, OUTPUTHEADERS, WRITERBUFFERSIZE, config, "sample")
        logger = ExperimentLogger(experiment_id=path.stem, metadata=metadata)
    (experiment, filename) = ExperimentRunner.run_experiment(DATASET, config, writer, logger, path)
    writer.finish()
    logger.save(path.with_suffix(".pkl"))

    if (globals.reuse_runs & flags) != flags:
        experiment.model.visualize_simulation("figures")
        ExperimentRunner.print_transactions(experiment)

if __name__ == "__main__":
    if (False):
        mp.freeze_support()
    main()
    for p in mp.active_children():
        #print("Terminating:", p.pid)
        p.terminate()
    print("Done :)")
