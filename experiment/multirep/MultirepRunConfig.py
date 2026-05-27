class MultirepRunConfig:
  def __init__(self,
        partition_file,
        dataset="MNIST",
        reward=int(1e18),
        minimum_rounds=5,
        min_buy_in=int(1e18),
        max_buy_in=int(1e18),
        standard_buy_in=int(1e18),
        epochs=1,
        batch_size=32,
        punish_factor=3,
        punish_factor_contrib=3,
        first_round_fee=50,
        use_outlier_detection=True,
        contribution_score_strategy="loss_tolerance_snap",
        loss_tolerance_pct=0.05,
        freerider_noise_scale=0.1,
        freerider_start_round=3,
        malicious_noise_scale=1.0,
        malicious_start_round=3,
        number_of_participants=8,
        force_merge_all=False,
        enabled_prints=None,
        fork=True,
        seed=123,
        allow_overlap=False,
        replication_factor=1.0):
    self.partition_file = partition_file
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

  def to_experiment_config(self):
    """Build ExperimentConfiguration using the full partition_file JSON."""
    from experiment.experiment_configuration import ExperimentConfiguration
    return ExperimentConfiguration(
        **self._base_config_kwargs(),
        per_user_partitions=self.partition_file,
    )

  def to_experiment_config_with_partitions(self, partitions):
    """Build ExperimentConfiguration from a pre-parsed {dataset_key: {user_index: UserPartitionSpec}} dict.

    Used by multirep to create a per-run config containing only the selected
    participants' specs, so contributor counts and fingerprints are correct.
    """
    from experiment.experiment_configuration import ExperimentConfiguration
    return ExperimentConfiguration(
        **self._base_config_kwargs(),
        per_user_partitions=partitions,
    )
