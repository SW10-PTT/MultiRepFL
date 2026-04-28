from datetime import datetime
import sys
import multiprocessing as mp
from pathlib import Path
import experiment_runner as ExperimentRunner
from experiment_configuration import ExperimentConfiguration
from openfl.utils.async_writer import AsyncWriter
from helper import getPath

# Add the repo root to sys.path so `analysis` package is importable from here
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis import ExperimentLogger

config = ExperimentConfiguration(
    min_buy_in=int(1e18),
    max_buy_in=int(1e18),
    contribution_score_strategy="loss_only",
    use_outlier_detection=True,
    minimum_rounds=2,
    force_merge_all=False,
    freerider_noise_scale=0.5,
    malicious_noise_scale=0.5,
    punish_factor=3,
    punish_factor_contrib=3,
    freerider_start_round=1,
    malicious_start_round=1,
    #data_percentages=[30, 10, 15, 15, 10, 20],
    # 0: {"only_labels": [0, 1, 2, 3, 4]}
    # 0: {"flip_map": {4: 9}}
    # 0: {"only_labels": [0, 1, 2, 3, 4], "flip_map": {4: 9}}
    label_rules={
        0: {"only_labels": [0, 1, 2, 3, 4]},
        1: {"only_labels": [0, 1, 2, 3, 4]},
        2: {"only_labels": [0, 1, 2, 3, 4]},
        3: {"only_labels": [5, 6, 7, 8, 9]},
        4: {"only_labels": [5, 6, 7, 8, 9]},
        5: {"only_labels": [5, 6, 7, 8, 9]},
    },
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
    writer = AsyncWriter(path, OUTPUTHEADERS, WRITERBUFFERSIZE, config, "sample")
    metadata = {**vars(config), "dataset": DATASET, "timestamp": startTime}
    logger = ExperimentLogger(experiment_id=path.stem, metadata=metadata)
    experiment = ExperimentRunner.run_experiment(DATASET, config, writer, logger)
    writer.finish()
    logger.save(path.with_suffix(".pkl"))

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
