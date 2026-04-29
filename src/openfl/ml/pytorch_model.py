from __future__ import annotations

import copy
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import os
import time
import math
from collections import Counter
from web3 import Web3
from typing import Tuple, List
from collections import OrderedDict
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader, Subset, random_split

from openfl.ml.data_partition import DataPartition

if TYPE_CHECKING:
    from experiment.experiment_configuration import ExperimentConfiguration
from openfl.ml.Participant import Participant
from openfl.utils.RunRepo import RunRepo
from openfl.utils.ITestAndTrainer import ITestAndTrainer, get_filename
from openfl.utils.PytorchTrainer import PyTorchTrainer
from openfl.utils.types.EvaluationData import EvaluationData
from openfl.api import globals
from openfl.utils.types.Attitude import Attitude
from openfl.utils.types.Colors import gb, rb, red, yellow, green, b
from openfl.utils.types.ReplayTrainingSpecs import ReplayTrainingSpecs
from openfl.utils.types.userDict import UserDict
torch._dynamo.config.cache_size_limit = 512
import logging
debugging = sys.gettrace() is not None
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = (DEVICE.type == "cuda")
PIN_MEMORY = USE_CUDA
NON_BLOCKING = USE_CUDA
NUM_WORKERS = min(4, os.cpu_count() // 2) if torch.cuda.is_available() else 0
PERSISTENT_WORKERS = USE_CUDA and NUM_WORKERS > 0
AMP = USE_CUDA # Optional: mixed precision on CUDA

# cuDNN autotune for fixed-size inputs (both MNIST 28x28 and CIFAR-10 32x32)
torch.backends.cudnn.benchmark = USE_CUDA
if DEVICE.type == "cuda":
    torch.set_float32_matmul_precision("high")

def model_to_device(net: nn.Module) -> nn.Module:
    # Move model once; keep it on the chosen device
    return net.to(DEVICE, non_blocking=NON_BLOCKING)

def cuda_safe_dataloader(ds, batch_size, shuffle=False):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT_WORKERS,
    )



class Net_CIFAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net_MNIST(nn.Module):
    def __init__(self):
        super(Net_MNIST, self).__init__()
        # input is 28x28
        # padding=2 for same padding
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        # feature map size is 14*14 by pooling
        # padding=2 for same padding
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        # feature map size is 7*7 by pooling
        self.fc1 = nn.Linear(64*7*7, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64*7*7)   # reshape Variable
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # return F.log_softmax(x)
        return x

class PytorchModel:
    def __init__(self, config: ExperimentConfiguration, DATASET, _goodParticipants, _totalParticipants, epochs, batchsize, default_collateral, max_collateral, freerider_noise_scale: float = 1.0, freerider_start_round: int = 3, malicious_start_round: int = 3, malicious_noise_scale: float = 1.0, force_merge_all: bool = False):
        self.replaying = None
        self.config: ExperimentConfiguration = config
        if config.dataset == "mnist":
            self.global_model = Net_MNIST().to(DEVICE)
        else:
            self.global_model = Net_CIFAR().to(DEVICE)


        self.NUMBER_OF_CONTRIBUTERS = _totalParticipants
        self.NUMBER_OF_BAD_CONTRIBUTORS = 0
        self.NUMBER_OF_FREERIDER_CONTRIBUTORS = 0
        self.NUMBER_OF_INACTIVE_CONTRIBUTORS = 0
        self.NUMBER_OF_HONEST_CONTRIBUTORS = 0
        self.DATA = None
        self.train_by_user_id = {}
        self.val_by_user_id = {}
        self.mnist_prepared_user_ids = tuple()
        self.participants = []
        self.disqualified = []
        self.runRepo: ITestAndTrainer = None
        # self.EPOCHS = epochs
        # self.BATCHSIZE = batchsize
        self.train, self.val, self.test = self.load_data(self.NUMBER_OF_CONTRIBUTERS, _print=True)
        # self.default_collateral = default_collateral
        # self.max_collateral = max_collateral
        # self.force_merge_all = force_merge_all
        # INTERFACE VARIABLES
        self.accuracy = []
        self.loss = []

        self.round = 0

        if freerider_noise_scale < 0:
            raise ValueError("freerider_noise_scale must be non-negative")
        self.freerider_noise_scale = freerider_noise_scale

        if freerider_start_round < 1:
            raise ValueError("freerider_start_round must be at least 1")
        self.freerider_start_round = freerider_start_round

        if malicious_start_round < 1:
            raise ValueError("malicious_start_round must be at least 1")
        self.malicious_start_round = malicious_start_round

        if malicious_noise_scale < 0:
            raise ValueError("malicious_noise_scale must be non-negative")
        self.malicious_noise_scale = malicious_noise_scale

        print("===================================================================================")
        print("Pytorch Model created:\n")
        print(str(self.global_model))
        print("\n===================================================================================")

    def add_participant(self, user):
        train_loader, val_loader = self.get_user_dataloaders(user)
        
        if self.config.dataset == "mnist":
            _model = Net_MNIST().to(DEVICE)
        else:
            _model = Net_CIFAR().to(DEVICE)

        optimizer = optim.SGD(_model.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        l = len(self.participants)
        self.participants.append(Participant.from_user(
            user,
            train_loader,
            val_loader,
            _model,
            optimizer,
            criterion
        ))

        print("Participant added: {:<9} {}".format(rb(user.attitude.name.upper()[0]+user.attitude.name[1:]), rb("User")))

    def prepare_data_for_users(self, users, dataset_name, seed=42, allow_overlap=False, replication_factor=1.0):
        users = list(users)

        if dataset_name == "mnist":
            trainset = MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
            testset = MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
        if dataset_name == "cifar-10":
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            trainset = CIFAR10("./data", train=True, download=True, transform=transform)
            testset = CIFAR10("./data", train=False, download=True, transform=transform_test)

        partitioner = DataPartition(
            validation_split=0.1,
            seed=seed,
            allow_overlap=allow_overlap,
            replication_factor=replication_factor,
        )
        user_splits = partitioner.split_by_label(users, trainset.targets)

        trainloaders = []
        valloaders = []
        self.train_by_user_id = {}
        self.val_by_user_id = {}
        actual_splits = {}

        for user in users:
            user_id = user.get_id_or_address()
            user_split = user_splits[user_id]
            train_ids = list(user_split["train_ids"])
            val_ids = list(user_split["val_ids"])

            if user.only_labels:
                train_ids = partitioner.filter_indices_by_label(train_ids, trainset.targets, user.only_labels)
                val_ids = partitioner.filter_indices_by_label(val_ids, trainset.targets, user.only_labels)

            actual_splits[user_id] = {
                "data_percent": user_split["data_percent"],
                "num_samples": len(train_ids) + len(val_ids),
                "train_ids": train_ids,
                "val_ids": val_ids,
                "train_samples": len(train_ids),
                "val_samples": len(val_ids),
            }

            train_dataset = Subset(trainset, train_ids)
            val_dataset = Subset(trainset, val_ids)

            if user.flip_map:
                train_dataset = partitioner.apply_flip_map(train_dataset, user)
                val_dataset = partitioner.apply_flip_map(val_dataset, user)

            train_loader = cuda_safe_dataloader(train_dataset, self.config.batch_size, shuffle=True)
            val_loader = cuda_safe_dataloader(val_dataset, self.config.batch_size, shuffle=False)
            trainloaders.append(train_loader)
            valloaders.append(val_loader)
            self.train_by_user_id[user_id] = train_loader
            self.val_by_user_id[user_id] = val_loader

        self.print_data_split_summary(users, actual_splits, trainset.targets, dataset_name)

        testloader = cuda_safe_dataloader(testset, self.config.batch_size, shuffle=False)
        self.DATA = (trainloaders, valloaders, testloader)
        self.train, self.val, self.test = self.DATA

    def get_user_dataloaders(self, user):
        if self.train_by_user_id:
            user_id = user.get_id_or_address()
            return self.train_by_user_id[user.address], self.val_by_user_id[user.address]

        trainloaders, valloaders, _test = self.load_data(self.NUMBER_OF_CONTRIBUTERS)
        index = len(self.participants)
        return trainloaders[index], valloaders[index]

    def print_data_split_summary(self, users, user_splits, labels, dataset_name):
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        else:
            labels = list(labels)

        dataset_size = len(labels)
        num_classes = len(set(labels))
        per_class = dataset_size // num_classes
        print(f"Dataset: {dataset_name} | {dataset_size:,} total samples | {num_classes} classes | ~{per_class:,} per class")
        print()
        print("Data split per user:")
        print(
            "{:<4} {:<16} {:>10} {:>10} {:>9}   {}".format(
                "Idx",
                "Address",
                "Config %",
                "Actual %",
                "Samples",
                "Rule",
            )
        )
        print("-" * 75)

        total_config_percent = 0.0
        total_actual_percent = 0.0
        total_samples = 0
        label_flip_rows = []
        label_dist_rows = []
        all_label_classes = sorted(set(labels))

        for user in users:
            user_id = user.get_id_or_address()
            split = user_splits[user_id]
            config_percent = float(user.data_percent)
            actual_percent = (100.0 * split["num_samples"] / dataset_size)
            total_config_percent += config_percent
            total_actual_percent += actual_percent
            total_samples += int(split["num_samples"])

            print(
                "{:<4} {:<16} {:>9.2f}% {:>9.2f}% {:>9,}   {}".format(
                    getattr(user, "number", user_id),
                    user.address[0:14] + "...",
                    config_percent,
                    actual_percent,
                    split["num_samples"],
                    self.get_label_method(user),
                )
            )

            all_ids = split["train_ids"] + split["val_ids"]
            label_counts = self.count_labels(labels, all_ids)
            label_dist_rows.append((getattr(user, "number", user_id), label_counts))

            if user.flip_map:
                train_flip_counts = self.count_flipped_labels(labels, split["train_ids"], user.flip_map)
                val_flip_counts = self.count_flipped_labels(labels, split["val_ids"], user.flip_map)
                label_flip_rows.append((
                    getattr(user, "number", user_id), "Train",
                    self.format_flip_counts(train_flip_counts),
                    sum(train_flip_counts.values()),
                    self.ratio_percent(sum(train_flip_counts.values()), split["train_samples"]),
                ))
                label_flip_rows.append((
                    getattr(user, "number", user_id), "Val",
                    self.format_flip_counts(val_flip_counts),
                    sum(val_flip_counts.values()),
                    self.ratio_percent(sum(val_flip_counts.values()), split["val_samples"]),
                ))

        lost_samples = dataset_size - total_samples
        lost_percent = 100.0 * lost_samples / dataset_size

        print("-" * 75)
        print("Configured total: {:.2f}%".format(total_config_percent))
        print("Assigned: {:>7,} / {:,} samples  ({:.2f}%)".format(total_samples, dataset_size, total_actual_percent))
        if lost_samples > 0:
            print("Lost:     {:>7,} / {:,} samples  ({:.2f}%)  <- dropped due to only_labels".format(lost_samples, dataset_size, lost_percent))
        print()

        col_w = 6
        header = "{:<4} " + " ".join(f"{'L' + str(l):>{col_w}}" for l in all_label_classes)
        print(header.format("Idx"))
        print("-" * (5 + col_w * len(all_label_classes)))
        for user_idx, label_counts in label_dist_rows:
            row = "{:<4} " + " ".join(f"{label_counts.get(l, 0):>{col_w},}" for l in all_label_classes)
            print(row.format(user_idx))

        if not label_flip_rows:
            return

        print()
        print("Label flips (samples changed per user):")
        print("{:<4} {:<5}   {:<24} {:>8} {:>11}".format("Idx", "Set", "Flip counts", "Changed", "% of set"))
        print("-" * 60)
        for user_idx, set_name, flip_counts, changed_total, changed_percent in label_flip_rows:
            print("{:<4} {:<5}   {:<24} {:>8} {:>10.2f}%".format(
                user_idx, set_name, flip_counts, changed_total, changed_percent,
            ))
        print("-" * 60)

    def count_labels(self, labels, ids):
        counts = Counter(labels[i] for i in ids)
        return dict(sorted(counts.items()))

    def count_flipped_labels(self, labels, ids, flip_map):
        before_counts = self.count_labels(labels, ids)
        return {
            f"{src}->{dst}": before_counts.get(src, 0)
            for src, dst in flip_map.items()
            if before_counts.get(src, 0) > 0
        }

    def format_flip_counts(self, flip_counts):
        if not flip_counts:
            return "none"
        return ", ".join(f"{flip}:{count}" for flip, count in flip_counts.items())

    def ratio_percent(self, part, whole):
        if whole == 0:
            return 0.0
        return 100.0 * part / whole

    def get_user_role(self, user):
        role = getattr(user, "futureAttitude", None)
        if role is None:
            role = getattr(user, "attitude", None)
        return role.name if role is not None else "Unknown"

    def get_label_method(self, user):
        if user.only_labels is not None and user.flip_map:
            return f"only={user.only_labels}, flip={user.flip_map}"
        if user.only_labels is not None:
            return f"only={user.only_labels}"
        if user.flip_map:
            return f"flip={user.flip_map}"
        return "none"


    def load_data(self, NUM_CLIENTS, _print=False):
        if self.DATA:
            return self.DATA

        if self.config.dataset == "cifar-10":
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            trainset = CIFAR10("./data", train=True, download=True, transform=transform)
            testset = CIFAR10("./data", train=False, download=True, transform=transform_test)
        else:
            trainset = MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
            testset = MNIST("./data", train=False, download=True, transform=transforms.ToTensor())


        if _print:
            print("Data Loaded:")
            print("Nr. of images for training: {:,.0f}".format(len(trainset)))
            print("Nr. of images for testing:  {:,.0f}\n".format(len(testset)))

        # Split training set into partitions to simulate the individual dataset
        partition_size = len(trainset) // NUM_CLIENTS
        lengths = [partition_size] * NUM_CLIENTS

        images_needed = partition_size * NUM_CLIENTS
        if images_needed < len(trainset):
            trainset,_ = random_split(trainset, [images_needed, len(trainset)-images_needed])

        datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

        # Split each partition into train/val and create DataLoader
        trainloaders = []
        valloaders = []
        for ds in datasets:
            len_val = len(ds) // 10  # 10 % validation set
            len_train = len(ds) - len_val
            lengths = [len_train, len_val]
            ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
            trainloaders.append(DataLoader(
                ds_train,
                batch_size=self.config.batch_size,
                shuffle=True,
                pin_memory=PIN_MEMORY,
                num_workers=NUM_WORKERS,
                persistent_workers=PERSISTENT_WORKERS,
            ))
            valloaders.append(DataLoader(
                ds_val,
                batch_size=self.config.batch_size,
                shuffle=False,
                pin_memory=PIN_MEMORY,
                num_workers=NUM_WORKERS,
                persistent_workers=PERSISTENT_WORKERS,
            ))
        testloader = DataLoader(
            testset,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=PIN_MEMORY,
            num_workers=NUM_WORKERS,
            persistent_workers=PERSISTENT_WORKERS,
        )
        self.DATA = (trainloaders, valloaders, testloader)
        return trainloaders, valloaders, testloader


    def federated_training(self):
        if debugging or globals.reuse_runs:
            print(b("\n================ SEQUENTIAL FEDERATED TRAINING START ================"))
            start_total = time.perf_counter()
            print(yellow(f"{'Debugging mode' if debugging else ''} {'and ' if debugging and globals.reuse_runs else ''}{'reuse_runs ' if globals.reuse_runs else ''}detected → running sequential training"))
            results = self.run_sequential()
        else:
            print(b("\n================ PARALLEL FEDERATED TRAINING START ================"))
            start_total = time.perf_counter()
            results = self.run_multi_processing()

        self.apply_training_results(results)
        total_time = time.perf_counter() - start_total

        print(b("=================== PARALLEL TRAINING END ===================\n"))
        print(green(f"Total federated training time: {total_time:.2f} seconds\n"))

    def finalize_paricipant_evaluation(self, participant): # Same as lines 294-296,306 in orgiginal code.
        loss, acc = self.runRepo.test(self.round,f"test-finalize_paricipant_evaluation-{participant.id}", participant.model, self.test, DEVICE) # TODO: Investigate if this should be user.val instead.
        participant._accuracy.append(acc) # Line 295 in original code # TODO: Investigate if this should be test and not validation accuracy.
        participant._loss.append(loss) # Line 296 in original code # TODO: Investigate if this should be test and not validation loss.
        participant.hashedModel = self.runRepo.get_hash(self.round, f"finalize_paricipant_evaluation-{participant.id}", participant.model.state_dict())


    def apply_training_results(self, results):
        # Apply results back to participants
        participant_map = {x.id: x for x in self.participants}
        for user_address, state_dict, val_loss, val_acc in results:
            user = participant_map[user_address]
            user.model.load_state_dict(state_dict)
            user.currentAcc = val_acc # Line 287 in original code
            user.currentLoss = val_loss
            self.finalize_paricipant_evaluation(user)


    def run_sequential(self):
        num_gpus = torch.cuda.device_count()
        print_training_mode(num_gpus, 1)

        results = []

        for idx, user in enumerate(self.participants):
            device_id = idx % max(1, num_gpus)
            sd_cpu = {k: v.cpu() for k, v in user.model.state_dict().items()}

            if user.attitude == Attitude.Honest:
                result = self.runRepo.train_user_proc(
                    self.round,
                     f"train_user_proc-run_sequential-{user.id}-{self.round}",
                    user.id,
                    sd_cpu,
                    user.train.dataset,
                    user.val.dataset,
                    self.config.epochs,
                    device_id,
                    self.config.dataset,
                    self.config.batch_size,
                    PIN_MEMORY,
                    False
                )
                results.append(result)
            else:
                self.finalize_paricipant_evaluation(user)
        return results

    def run_multi_processing(self):
        num_gpus = torch.cuda.device_count()
        ctx = mp.get_context("spawn")

        available_workers = num_gpus if num_gpus > 0 else (os.cpu_count() or 1)

        num_processes = max(
            1,
            min(len(self.participants), available_workers)
        )

        print_training_mode(num_gpus, num_processes)

        with ctx.Pool(processes=num_processes) as pool:
            start_pool = time.perf_counter()

            async_results = []
            for idx, user in enumerate(self.participants):
                device_id = idx % max(1, num_gpus)
                sd_cpu = {k: v.cpu() for k, v in user.model.state_dict().items()}  # safe copy

                if user.attitude == Attitude.Honest: # train
                    async_results.append(pool.apply_async(
                        train_user_proc,
                        (
                        user.address,
                        sd_cpu,
                        user.train.dataset,
                        user.val.dataset,
                        self.config.epochs,
                        device_id,
                        self.config.dataset,
                        self.config.batch_size,
                        PIN_MEMORY,
                        False)
                    ))
                else: # If user's behaviour !good, skip Training.
                    # Skips apply_training_results() - goes directly to evaluation. Corresponds to lines 261-277 in original code.
                    self.finalize_paricipant_evaluation(user)

            results = [r.get() for r in async_results] # Gather results from Multi-Processing
        print(green(f"Parallel execution time: {time.perf_counter() - start_pool:.2f} seconds"))
        return results


    def let_malicious_users_do_their_work(self):
        for i in range(len(self.participants)):
            if self.participants[i].attitude == Attitude.Malicious:
                print(red("Address {} going to provide random weights".format(self.participants[i].address[0:16]+"...")))
                manipulated_state_dict = manipulate(self.participants[i].model,scale=self.malicious_noise_scale,)
                self.participants[i].model.load_state_dict(manipulated_state_dict)
                self.participants[i].hashedModel = self.runRepo.get_hash(self.round, f"let_malicious_users_do_their_work-{self.participants[i].id}", self.participants[i].model.state_dict())
                loss, accuracy = self.runRepo.test(self.round,f"test-let_malicious_users_do_their_work-usermodel-{self.participants[i].id}", self.participants[i].model, self.test, DEVICE)
                print("{:<17} {} |  Testing  | Accuracy {:>3.0f} % | Loss ∞\n".format("Account testing:   ",
                                                                                self.participants[i].address[0:16]+"...",
                                                                                accuracy*100, loss))
                # TODO: Why is test_loss not used here?

    def update_users_attitude(self):
        for user in self.participants:
            if user.attitudeSwitch == self.round \
                and user.attitude != user.futureAttitude:
                print(rb("Address {} going to switch attitude to {}".format(user.address[0:16]+"...",
                                                                            user.futureAttitude)))
                user.attitude = user.futureAttitude
                user.update_color(None, user.attitude)


    def let_freerider_users_do_their_work(self):
        for participant in self.participants:
            if participant.attitude == Attitude.FreeRider:

                # # Freerider has no data and must therefore provide something random
                # # After first round freerider can copy other participants
                # if self.round == 1:
                #     print(red("Account {} going to provide ".format(user_idx[0:8]+"...") \
                #                   + "random weights; starts copycat-ing " \
                #                   + "next round"))
                #
                #     new_state_dict = manipulate(copy.deepcopy(participant.model))
                # else:
                #     foreign_model = copy.deepcopy(self.participants[0].previousModel)
                #     new_state_dict = foreign_model.state_dict()
                #
                # participant.model.load_state_dict(new_state_dict)
                #
                # if self.round > 1:
                #     print(red("Address {} going to add random noise to weights".format(user_idx[0:16]+"...")))
                #     participant.model.load_state_dict(add_noise(copy.deepcopy(participant.model)))
                if self.round < self.freerider_start_round:
                    print(yellow(
                        "Address {} waiting until round {} to start freeriding".format(
                            participant.address[0:16] + "...",
                            self.freerider_start_round,
                        )
                    ))
                    new_state_dict = manipulate(copy.deepcopy(participant.model))
                else:
                    new_state_dict = self._freerider_submit_with_noise(participant)


                participant.model.load_state_dict(new_state_dict)
                participant.hashedModel = self.runRepo.get_hash(self.round, f"get_hash-let_freerider_users_do_their_work-{participant.id}",participant.model.state_dict())
                loss, accuracy = self.runRepo.test(self.round,f"test-let_freerider_users_do_their_work-usermodel-{participant.id}", participant.model, self.test, DEVICE)
                print("{:<17} {} |  Testing  | Accuracy {:>3.0f} % | Loss ∞\n".format("Account testing:   ",
                                                                                participant.address[0:16]+"...",
                                                                                accuracy*100, loss))
                # TODO: Why is loss not used here?


    def _freerider_submit_with_noise(self, user):
        """Freerider reuses the global model with configurable noise."""

        if self.freerider_noise_scale < 0:
            raise ValueError("freerider_noise_scale must be non-negative")

        if self.freerider_noise_scale == 0: # Copy global model if noise is zero
            print(yellow("Address {} resubmitting original model".format(user.address[0:16]+"...")))
            return copy.deepcopy(user.model).state_dict()

        print(red(
            "Address {} adding noise (scale={}) to global weights".format(
                user.address[0:16]+"...",
                self.freerider_noise_scale,
            )
        ))
        return manipulate(copy.deepcopy(user.model), scale=self.freerider_noise_scale)


    def the_merge(self, _users):
        # No qualified users → skip merge this round
        if not _users:
            print("-----------------------------------------------------------------------------------")
            print(red("No participants qualified for merge this round – skipping aggregation"))
            print("-----------------------------------------------------------------------------------\n")
            return

        ids, client_models = [], []
        for u in _users:
            ids.append(u.address)
            client_models.append(u.model)
            print("Account {} participating in merge".format(u.address[0:16]+"..."))
            #print(test(c[1],self.test,DEVICE))

        with torch.no_grad():
            global_dict = self.global_model.state_dict()
            for k in global_dict.keys():
                stacked = torch.stack([
                    client_models[i].state_dict()[k].to(
                        device=global_dict[k].device,
                        dtype=global_dict[k].dtype
                    )
                    for i in range(len(client_models))
                ], dim=0)
                global_dict[k] = stacked.mean(0)
            self.global_model.load_state_dict(global_dict)

        loss, accuracy = self.runRepo.test(self.round,f"test-themerge-globalmodel", self.global_model,self.test,DEVICE)
        self.accuracy.append(accuracy)
        self.loss.append(loss)
        print("-----------------------------------------------------------------------------------")
        print(b("Merged Model: Accuracy {:>3.0f} % | Loss {:>6,.2f}".format(accuracy*100,loss)))

        for u in self.participants:
            u.previousModel = copy.deepcopy(u.model) #the model from this round
            u.model.load_state_dict(self.global_model.state_dict()) #the global model

        print("-----------------------------------------------------------------------------------\n")



    def exchange_models(self):
        print("Users exchanging models...")
        for user in self.participants:
            user.userToEvaluate = []
            for j in self.participants:
                if user.model == j.model:
                    continue
                if j.model in user.userToEvaluate:
                    continue
                user.userToEvaluate.append(j)
        print("-----------------------------------------------------------------------------------")


    def verify_models(self, on_chain_hashes):
        print("Users verifying models...")
        for _user in self.participants:
            _user.cheater = []
            for user in _user.userToEvaluate:
                if not self.runRepo.get_hash(self.round, f"get_hash-verify_models-{user.id}", user.model.state_dict()) == on_chain_hashes[user.id]:
                    print(red(f"Account {_user.number}: Account {user.address[0:16]}... could not provide the registered model"))
                    _user.cheater.append(user)

        print("-----------------------------------------------------------------------------------")

    def get_global_model_hash(self):
        return get_hash(self.global_model.state_dict())

    def evaluation(self):
        print("Users evaluating models...")

        scalar = 100 # Adds more decimals for precision (Adding 0 gives another decimal, vice versa)
        MAX_UINT16_SIZE = 65535
        count_dq = len(self.disqualified)\

        matrices = EvaluationData.new(self.participants + self.disqualified)

        # n = len(self.participants) + count_dq

        # feedback_matrix = np.zeros((n, n), dtype=np.int8)
        # accuracy_matrix = [[0 for _ in range(n)] for _ in range(n)]
        # loss_matrix = [[0 for _ in range(n)] for _ in range(n)]
        # prev_accs = [0 for _ in range(n)]
        # prev_losses = [0 for _ in range(n)]

        for feedbackGiver in self.participants:
            valloader = feedbackGiver.val
            bad_att = feedbackGiver.attitude == Attitude.Malicious
            free_att = feedbackGiver.attitude == Attitude.FreeRider
            accuracy_last_round = -1

            # Depending on the attitude of the feedback giver, the evaluation is done differently:

            # For each user, traverse its list of usersToEvaluate and fill the feedback, accuracy and loss matrices
            for ix, user in enumerate(feedbackGiver.userToEvaluate):
                giver_idx = feedbackGiver.id
                user_idx = user.id
                prev_loss, prev_acc = self.runRepo.test(self.round,f"test-feedback-globalmodel-{giver_idx}-{user_idx}", self.global_model, valloader, DEVICE)
                loss, accuracy = self.runRepo.test(self.round,f"test-feedback-usermodel-{giver_idx}-{user_idx}", user.model, valloader, DEVICE)
                if not bad_att and not free_att:
                    prev_acc = round(prev_acc * 100 * scalar)
                    prev_loss = safe_scale(prev_loss, scalar, MAX_UINT16_SIZE)

                if bad_att:
                    matrices.feedback_matrix[giver_idx, user_idx] = -1
                    matrices.accuracy_matrix[giver_idx, user_idx] = 0
                    matrices.loss_matrix(giver_idx, user_idx, 65535)
                    matrices.prev_accuracies[giver_idx] = round(prev_acc * 100 * scalar)
                    matrices.prev_losses[giver_idx] = safe_scale(prev_loss, scalar, MAX_UINT16_SIZE)


                elif free_att:
                    matrices.feedback_matrix[giver_idx, user_idx] = 0
                    if accuracy_last_round == -1:
                        loss_last_round, accuracy_last_round = self.runRepo.test(self.round,f"test-feedback-globalmodel-accuracy_last_round-{giver_idx}-{user_idx}",self.global_model, valloader, DEVICE)
                        accuracy_last_round = round(accuracy_last_round * 100 * scalar)
                        loss_last_round = safe_scale(loss_last_round, scalar, MAX_UINT16_SIZE)
                    matrices.accuracy_matrix[giver_idx, user_idx] = accuracy_last_round
                    matrices.loss_matrix[giver_idx, user_idx] = min(loss_last_round , MAX_UINT16_SIZE)
                    matrices.prev_accuracies[giver_idx] = accuracy_last_round
                    matrices.prev_losses[giver_idx] = loss_last_round

                elif len(feedbackGiver.cheater) > 0 and user in feedbackGiver.cheater:
                    matrices.feedback_matrix[giver_idx, user_idx] = -1
                    matrices.accuracy_matrix[giver_idx, user_idx] = round(accuracy * 100 * scalar)
                    matrices.loss_matrix[giver_idx, user_idx] = safe_scale(loss, scalar, MAX_UINT16_SIZE)
                    matrices.prev_accuracies[giver_idx] = prev_acc
                    matrices.prev_losses[giver_idx] = prev_loss

                elif accuracy > feedbackGiver.currentAcc - 0.07:  # 7% Worse
                    matrices.feedback_matrix[giver_idx, user_idx] = 1
                    matrices.accuracy_matrix[giver_idx, user_idx] = round(accuracy * 100 * scalar)
                    matrices.loss_matrix[giver_idx, user_idx] = safe_scale(loss, scalar, MAX_UINT16_SIZE)
                    matrices.prev_accuracies[giver_idx] = prev_acc
                    matrices.prev_losses[giver_idx] = prev_loss

                elif accuracy > feedbackGiver.currentAcc - 0.14:  # 14% Worse
                    matrices.feedback_matrix[giver_idx, user_idx] = 0
                    matrices.accuracy_matrix[giver_idx, user_idx] = round(accuracy * 100 * scalar)
                    matrices.loss_matrix[giver_idx, user_idx] = safe_scale(loss, scalar, MAX_UINT16_SIZE)
                    matrices.prev_accuracies[giver_idx] = prev_acc
                    matrices.prev_losses[giver_idx] = prev_loss

                else:
                    matrices.feedback_matrix[giver_idx, user_idx] = -1
                    matrices.accuracy_matrix[giver_idx, user_idx] = round(accuracy * 100 * scalar)
                    matrices.loss_matrix[giver_idx, user_idx] = safe_scale(loss, scalar, MAX_UINT16_SIZE)
                    matrices.prev_accuracies[giver_idx] = prev_acc
                    matrices.prev_losses[giver_idx] = prev_loss

                if self.config.force_merge_all:
                    matrices.feedback_matrix[giver_idx, user_idx] = 0

                # Reset
                feedbackGiver.userToEvaluate = []

        print("FEEDBACK MATRIX:")
        print(matrices.feedback_matrix)
        print("-----------------------------------------------------------------------------------")
        print("ACCURACY MATRIX:")
        print(matrices.accuracy_matrix)
        print("-----------------------------------------------------------------------------------")
        print("LOSS MATRIX:")
        print(matrices.loss_matrix)
        print("-----------------------------------------------------------------------------------")
        print("PREVIOUS ACCURACIES:")
        print(matrices.prev_accuracies)
        print("-----------------------------------------------------------------------------------")
        print("PREVIOUS LOSSES:")
        print(matrices.prev_losses)
        print("-----------------------------------------------------------------------------------")

        return matrices

        return feedback_matrix, accuracy_matrix, loss_matrix, prev_accs, prev_losses, addresses

    def get_participant(self, address_or_id):
        p = next((x for x in self.participants if x.id == address_or_id), None)
        if p:
            return p
        return next((x for x in self.participants if x.address == address_or_id), None)

    def setup_replay(self, experiment_finger_print, config: ExperimentConfiguration):

        fileName = get_filename(experiment_finger_print, config)

        if globals.reuse_runs and fileName.is_file():
            self.runRepo: ITestAndTrainer = RunRepo(config, fileName)  # Hash config to compare?
            self.replaying = True
        else:
            self.runRepo: ITestAndTrainer = PyTorchTrainer(config, fileName)  # Hash config to compare?

        loss, accuracy = self.runRepo.test(0, f"test-setup_replay-globalmodel", self.global_model, self.test, DEVICE)

        self.accuracy = [accuracy]
        self.loss = [loss]

        self.round = 1

def get_hash(_state_dict):
    if not isinstance(_state_dict, dict):
        _state_dict = dict(_state_dict)

    parts = []
    for k, v in sorted(_state_dict.items(), key=lambda x: x[0]):
        t = v.detach()
        if t.is_cuda:
            t = t.cpu()
        t = t.contiguous()
        parts.append(k.encode("utf-8"))
        parts.append(b"|")
        # include shape to avoid accidental collisions
        parts.append(np.asarray(t.shape, dtype=np.int64).tobytes())
        parts.append(b"|")
        parts.append(t.numpy().tobytes())
        parts.append(b"\n")
    blob = b"".join(parts)
    return Web3.keccak(blob)  #remove hex to match old, with improved algo.

# PYTORCH FUNCTIONS
def train(net, trainloader: torch.utils.data.DataLoader, epochs: int, device: torch.device) -> None:

    # Compile ONCE per process (not per batch)
    if device.type == "cuda":
        try:
            net = torch.compile(net)#, mode="reduce-overhead")
        except Exception:
            pass

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    net.train()

    for _ in range(epochs):
        for images, labels in trainloader:
            images = images.to(device, non_blocking=NON_BLOCKING)
            labels = labels.to(device, non_blocking=NON_BLOCKING)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = net(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

def test(net, testloader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[float, float]:
    """
    Evaluate model on test set: forward pass only (no gradients), with optional AMP on CUDA
    Accumulate total cross-entropy loss and count correct predictions for accuracy
    Returns (total_loss, accuracy) over the entire test dataset
    """
    criterion = nn.CrossEntropyLoss()
    net.eval()

    correct = 0
    total = 0
    loss = 0.0

    use_amp = device.type == "cuda"

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device, non_blocking=NON_BLOCKING)
            labels = labels.to(device, non_blocking=NON_BLOCKING)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    loss = min(sys.float_info.max, loss)

    return loss, accuracy


def manipulate(model, scale: float = 1.0) -> OrderedDict:
    sd = OrderedDict()
    with torch.no_grad():
        for k, v in model.state_dict().items():
            t = v.clone()
            if t.is_floating_point():
                # uniform noise in [-scale, scale]
                noise = torch.empty_like(t).uniform_(-scale, scale)
                t.add_(noise)
            sd[k] = t
    return sd


def add_noise(model, offset_from_end: int = 5) -> OrderedDict:
    """
    GPU-friendly: keep tensors on their original device/dtype and add a tiny scalar
    to the tensor at index len(state_dict)-offset_from_end.
    """
    items = list(model.state_dict().items())
    target_idx = max(0, len(items) - offset_from_end)

    new_sd = OrderedDict()
    with torch.no_grad():
        for idx, (k, v) in enumerate(items):
            t = v.clone()
            if t.is_floating_point() and idx == target_idx:
                # Match original magnitude: 9e-6 or 1e-5
                eps = 1e-5 if random.randint(9, 10) == 10 else 9e-6
                t.add_(eps)  # in-place scalar add on the same device (CPU/GPU)
            new_sd[k] = t
    return new_sd

def train_user_proc(user_addr, model_state, train_ds, val_ds, epochs, device_id, dataset, batchsize, pin_memory,
                    shuffle):
    # Multi-GPU Support
    # Select device
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{device_id}" if use_cuda else "cpu")

    # Recreate model based on dataset
    if dataset == "mnist":
        model = Net_MNIST()
    else:
        model = Net_CIFAR()

    model.load_state_dict(model_state)
    model.to(device)

    # Rebuild dataloaders inside the process
    train_loader = DataLoader(train_ds, batch_size=batchsize, shuffle=shuffle,
                              pin_memory=pin_memory)  # TODO: Investigate if this breaks something
    val_loader = DataLoader(val_ds, batch_size=batchsize, shuffle=False,
                            pin_memory=pin_memory)  # TODO: Investigate if this breaks something

    train(model, train_loader, epochs, device)  # Line 285 in original code
    val_loss, val_acc = test(model, val_loader, device)  # Line 286 in original code

    # del: Mark for GC
    del train_loader
    del val_loader

    print(f"[{device_label(device, device_id)}] User {user_addr} done | Acc: {val_acc:.3f}, Loss: {val_loss:.3f}")

    # Ensure all GPU work is complete before worker exits
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return user_addr, model.state_dict(), val_loss, val_acc

def print_training_mode(num_gpus: int, num_processes: int):
    """Prints a clean status message describing how training will run."""
    if num_gpus >= 2:
        print(green(f"Detected {num_gpus} GPU(s) → Parallel multi-GPU training"))

    elif num_gpus == 1:
        if num_processes > 1:
            print(yellow(
                f"Detected 1 GPU → Parallel training on one GPU (shared across {num_processes} workers)"
            ))
        else:
            print(green("Detected 1 GPU → Sequential GPU training"))

    else:  # CPU-only
        if num_processes > 1:
            print(yellow(
                f"Detected 0 GPU(s) → Parallel CPU training ({num_processes} workers)"
            ))
        else:
            print(red("Detected 0 GPU(s) → Sequential CPU mode"))



def device_label(device: torch.device, device_id: int = 0) -> str:
    if device.type == "cuda":
        return f"GPU {device_id}"
    else:
        return "CPU"

def safe_scale(value, scalar, max_val):
    if not math.isfinite(value):
        return max_val

    scaled = value * scalar

    if not math.isfinite(scaled):
        return max_val

    return min(round(scaled), max_val)