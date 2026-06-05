class MultirepRunConfig:
  def __init__(self,
        dataset="MNIST",
        reward=int(1e18),
        minimum_rounds=25,
        min_buy_in=int(1e18),
        max_buy_in=int(1e18),
        standard_buy_in=int(1e18),
        epochs=3,
        batch_size=32,
        punish_factor=3,
        punish_factor_contrib=3,
        first_round_fee=50,
        use_outlier_detection=True,
        contribution_score_strategy="loss_tolerance_snap",
        loss_tolerance_pct=0.1,
        freerider_noise_scale=0.1,
        freerider_start_round=3,
        malicious_noise_scale=1.0,
        malicious_start_round=3,
        number_of_participants=8,
        force_merge_all=False,
        enabled_prints=None,
        fork=True,
        seed=123,
        allow_overlap=True,
        replication_factor=4.0,
        training_mode=None,    # TrainingMode.LOCAL | REMOTE; None defaults to LOCAL
        remote_pool_size=0):   # REMOTE only: pool length; 0 = always submit fresh
    self.dataset = dataset.replace(".", "-").lower()
    self.reward = reward
    self.minimum_rounds = minimum_rounds

    self.min_buy_in = min_buy_in
    self.max_buy_in = max_buy_in
    self.standard_buy_in = standard_buy_in

    self.epochs = epochs
    self.batch_size = batch_size

    self.punish_factor = punish_factor
    self.punish_factor_contrib = punish_factor_contrib
    self.first_round_fee = first_round_fee

    self.use_outlier_detection = use_outlier_detection

    self.contribution_score_strategy = contribution_score_strategy
    self.loss_tolerance_pct = loss_tolerance_pct

    self.freerider_noise_scale = freerider_noise_scale
    self.freerider_start_round = freerider_start_round

    self.malicious_noise_scale = malicious_noise_scale
    self.malicious_start_round = malicious_start_round

    self.number_of_participants = number_of_participants

    self.force_merge_all = force_merge_all

    self.enabled_prints = enabled_prints if enabled_prints is not None else []

    self.fork = fork
    self.seed = seed
    self.allow_overlap = allow_overlap
    self.replication_factor = replication_factor

    from experiment.multirep.training_mode import TrainingMode
    self.training_mode: TrainingMode = (
        training_mode if training_mode is not None else TrainingMode.LOCAL
    )
    self.remote_pool_size: int = int(remote_pool_size)

  def _base_config_kwargs(self):
    return dict(
        dataset=self.dataset,
        reward=self.reward,
        minimum_rounds=self.minimum_rounds,
        min_buy_in=self.min_buy_in,
        max_buy_in=self.max_buy_in,
        standard_buy_in=self.standard_buy_in,
        epochs=self.epochs,
        batch_size=self.batch_size,
        punish_factor=self.punish_factor,
        punish_factor_contrib=self.punish_factor_contrib,
        first_round_fee=self.first_round_fee,
        use_outlier_detection=self.use_outlier_detection,
        contribution_score_strategy=self.contribution_score_strategy,
        loss_tolerance_pct=self.loss_tolerance_pct,
        freerider_noise_scale=self.freerider_noise_scale,
        freerider_start_round=self.freerider_start_round,
        malicious_noise_scale=self.malicious_noise_scale,
        malicious_start_round=self.malicious_start_round,
        number_of_participants=self.number_of_participants,
        force_merge_all=self.force_merge_all,
        enabled_prints=list(self.enabled_prints) if self.enabled_prints else None,
        partition_strategy="per_user",
        fork=self.fork,
        seed=self.seed,
        allow_overlap=self.allow_overlap,
        replication_factor=self.replication_factor,
    )

  @classmethod
  def from_dict(cls, d: dict) -> "MultirepRunConfig":
    from experiment.multirep.training_mode import TrainingMode
    d = dict(d)
    if "training_mode" in d and isinstance(d["training_mode"], str):
      d["training_mode"] = TrainingMode(d["training_mode"])
    return cls(**d)

  def to_experiment_config(self, partition_file):
    """Build ExperimentConfiguration using the given partition_file JSON."""
    from experiment.experiment_configuration import ExperimentConfiguration
    return ExperimentConfiguration(
        **self._base_config_kwargs(),
        per_user_partitions=partition_file,
    )

  def to_experiment_config_with_partitions(self, specs: dict):
    """Build ExperimentConfiguration from a flat {user_index: UserPartitionSpec} dict.

    Wraps *specs* under the normalised dataset key (e.g. "mnist", "cifar-10")
    so ExperimentConfiguration.get_partition_specs() can look them up by the
    active dataset name rather than falling back to the ANY_DATASET wildcard.
    """
    from experiment.experiment_configuration import ExperimentConfiguration
    from openfl.ml.partition_spec import normalize_dataset_name
    dataset_key = normalize_dataset_name(self.dataset)
    return ExperimentConfiguration(
        **self._base_config_kwargs(),
        per_user_partitions={dataset_key: specs},
    )
