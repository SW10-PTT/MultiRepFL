import math

from openfl.utils.types.TrainingSpecsJobListing import TrainingSpecsJobListing


class ExperimentConfiguration:
    def __init__(self,
                 number_of_good_contributors=4,
                 number_of_bad_contributors=1,
                 number_of_freerider_contributors=1,
                 number_of_inactive_contributors=0,
                 reward=int(1e18),
                 minimum_rounds=5,
                 min_buy_in=int(1e18),
                 max_buy_in=int(1e18),
                 standard_buy_in=int(1e18),
                 epochs=1,
                 batch_size=32,
                 punish_factor=3,
                 punish_factor_contrib=3,
                 first_round_fee=50, # Percentage of buy-in to charge as fee in first round
                 fork=True,
                 use_outlier_detection = True,
                 contribution_score_strategy="accuracy_only", # Options: dotproduct, naive, accuracy, None (defaults to dotproduct)
                 freerider_noise_scale=1.0,
                 freerider_start_round=3,
                 malicious_noise_scale=1.0,
                 malicious_start_round=3,
                 force_merge_all=False,
                 data_percentages=None,
                 label_rules=None): # Sets all entries in fbb to zeroes

        self.fork = fork

        # Apply scaling only if we’re on Sepolia (fork = False)
        if not fork:
            scale = 0.005  # scale down
            reward = int(reward * scale)
            min_buy_in = int(min_buy_in * scale)
            max_buy_in = int(max_buy_in * scale)
            standard_buy_in = int(standard_buy_in * scale)

        # Store everything
        self.number_of_good_contributors = number_of_good_contributors
        self.number_of_bad_contributors = number_of_bad_contributors
        self.number_of_freerider_contributors = number_of_freerider_contributors
        self.number_of_inactive_contributors = number_of_inactive_contributors
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
        self.contribution_score_strategy = contribution_score_strategy
        self.use_outlier_detection = use_outlier_detection
        self.freerider_noise_scale = freerider_noise_scale
        self.freerider_start_round = freerider_start_round
        self.malicious_start_round = malicious_start_round
        self.malicious_noise_scale = malicious_noise_scale
        self.force_merge_all = force_merge_all
        self.data_percentages = self._resolve_data_percentages(data_percentages)
        self.label_rules = self._resolve_label_rules(label_rules)

        class userConfig:
            number_of_good_contributors = "a"

    def get_training_specs(self, manager_address, model_hash) -> TrainingSpecsJobListing:
        return TrainingSpecsJobListing(model_hash, self.min_buy_in, self.max_buy_in, manager_address, self.reward, self.minimum_rounds, self.punish_factor, self.punish_factor_contrib, self.first_round_fee, 1) # Todo: Tasktype

    @property
    def number_of_contributors(self):
        return (self.number_of_good_contributors +
                self.number_of_bad_contributors +
                self.number_of_freerider_contributors +
                self.number_of_inactive_contributors)

    @property
    def number_of_data_users(self):
        return (self.number_of_good_contributors +
                self.number_of_bad_contributors +
                self.number_of_freerider_contributors)

    def _resolve_data_percentages(self, data_percentages):
        # make equal split 
        if data_percentages is None:
            equal_percent = 100.0 / self.number_of_data_users
            return [equal_percent] * self.number_of_data_users

        data_percentages = [float(percent) for percent in data_percentages]
        if len(data_percentages) != self.number_of_data_users:
            raise ValueError("data_percentages must match the number of configured users")
        if not math.isclose(sum(data_percentages), 100.0, abs_tol=1e-9):
            raise ValueError("data_percentages must sum to 100")

        return data_percentages

    def _resolve_label_rules(self, label_rules):
        # Example:
        # {
        #   2: {"only_labels": [4, 9], "flip_map": {4: 9, 9: 4}},
        #   3: {"flip_map": {2: 5}}
        # }
        if label_rules is None:
            return {}

        resolved_rules = {}
        for user_index, rule in label_rules.items():
            only_labels = rule.get("only_labels")
            flip_map = rule.get("flip_map", {})

            normalized_rule = {
                "only_labels": [int(label) for label in only_labels] if only_labels is not None else None,
                "flip_map": {int(src): int(dst) for src, dst in flip_map.items()},
            }
            resolved_rules[int(user_index)] = normalized_rule

        return resolved_rules
