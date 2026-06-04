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
import platform
import time
import math
from collections import Counter
from web3 import Web3
from typing import Tuple, List
from collections import OrderedDict
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader, Subset, random_split

from openfl.api.globals import ReplayMode
from openfl.ml.data_partition import DataPartition

# Imported only for type hints; skipped at runtime to avoid import errors when not on sys.path.
from experiment.print_config import AGGRESSIVE_GC
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
from openfl.utils.printer import log
from openfl.utils.types.ReplayTrainingSpecs import ReplayTrainingSpecs
from openfl.utils.types.userDict import UserDict

torch._dynamo.config.cache_size_limit = 512
import logging
debugging = sys.gettrace() is not None
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = (DEVICE.type == "cuda")
# torch.version.hip is set on ROCm builds (AMD); absent or None on NVIDIA CUDA builds
_IS_AMD_GPU = USE_CUDA and (getattr(torch.version, "hip", None) is not None)
_IS_NVIDIA_GPU = USE_CUDA and not _IS_AMD_GPU
# On Windows, DataLoader worker processes carry high overhead with CUDA/ROCm and
# can exhaust system RAM (36+ background processes for 8 participants × 4 workers).
# Use 0 workers on Windows; forked workers on Linux are much cheaper.
_IS_WINDOWS = platform.system() == "Windows"
_IS_AMD_WINDOWS = _IS_AMD_GPU and _IS_WINDOWS
# Linux torch default ('file_descriptor') keeps a real FD per tensor passed
# from a worker until the receiver releases it; under a long sweep with many
# DataLoaders these accumulate and eventually exhaust the per-process FD
# limit. 'file_system' shares tensors via /dev/shm files instead and avoids
# the leak. No-op on Windows where workers use spawn.
if AGGRESSIVE_GC and not _IS_WINDOWS:
    mp.set_sharing_strategy("file_system")
# pin_memory and non_blocking benefit NVIDIA but add overhead with no gain on AMD Windows
PIN_MEMORY = USE_CUDA and not _IS_AMD_WINDOWS
NON_BLOCKING = USE_CUDA and not _IS_AMD_WINDOWS
NUM_WORKERS = 0 if _IS_WINDOWS else (min(4, os.cpu_count() // 2) if torch.cuda.is_available() else 0)
# Persistent workers keep DataLoader subprocesses alive across iterations for
# speed, but bunch all their teardown at the end of the experiment — a long
# stall between runs in the auto sweep, and they hold FDs the whole time.
# Short-lived workers spread that cost across the run with negligible overhead
# on small datasets (MNIST/CIFAR).
PERSISTENT_WORKERS = False if AGGRESSIVE_GC else (USE_CUDA and NUM_WORKERS > 0)
# AMP is well-supported on NVIDIA and AMD Linux (ROCm); skip on AMD Windows where support is unreliable
AMP = USE_CUDA and not _IS_AMD_WINDOWS
COMPILE = False

# cuDNN autotune is NVIDIA-specific; MIOpen (AMD ROCm) ignores this flag
torch.backends.cudnn.benchmark = _IS_NVIDIA_GPU
if _IS_NVIDIA_GPU:
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


# Torchvision datasets are read-only once built (.data/.targets never mutate;
# transforms apply lazily per __getitem__), so one instance is safe to share
# across every run in this process. Building MNIST/CIFAR from disk is the bulk
# of per-run data overhead — cache by dataset name so both load_data() and
# prepare_data_for_users() reuse one instance. Shared by multirep (many tasks)
# and auto_runner (many runs) in the same process.
_DATASET_CACHE: dict[str, tuple] = {}


def _build_dataset_transforms(dataset_name):
    if dataset_name == "mnist":
        return transforms.ToTensor(), transforms.ToTensor()
    if dataset_name == "cifar-10":
        train_t = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        return train_t, test_t
    raise ValueError(f"Unknown dataset {dataset_name!r}. Expected 'mnist' or 'cifar-10'.")


def get_cached_datasets(dataset_name):
    """Return a process-cached (trainset, testset) pair, built from disk once."""
    cached = _DATASET_CACHE.get(dataset_name)
    if cached is not None:
        return cached
    train_t, test_t = _build_dataset_transforms(dataset_name)
    cls = MNIST if dataset_name == "mnist" else CIFAR10
    trainset = cls("./data", train=True, download=True, transform=train_t)
    testset = cls("./data", train=False, download=True, transform=test_t)
    _DATASET_CACHE[dataset_name] = (trainset, testset)
    return trainset, testset


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
        # Deterministic init so the shared-init broadcast in add_participant reproduces.
        torch.manual_seed(42)
        if USE_CUDA:
            torch.cuda.manual_seed_all(42)
        if config.dataset == "mnist":
            self.global_model = Net_MNIST().to(DEVICE)
        else:
            self.global_model = Net_CIFAR().to(DEVICE)

        if USE_CUDA and COMPILE:
            self.global_model = torch.compile(self.global_model, mode="reduce-overhead")


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
        self.test_tensors = None  # GPU-preloaded (images, labels) for the global test set
        # self.EPOCHS = epochs
        # self.BATCHSIZE = batchsize
        # In per_user mode run_experiment calls prepare_data_for_users(), which
        # builds and overwrites self.DATA/test — so load_data() here is wasted.
        # Skip it, except in HardPlayBack replay where prepare_data_for_users is
        # not called and load_data supplies the test set for global evaluation.
        _skip_init_load = (
            config.partition_strategy == "per_user"
            and not (globals.reuse_runs & globals.ReplayMode.HardPlayBack)
        )
        if _skip_init_load:
            self.train, self.val, self.test = None, None, None
        else:
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

        log("setup_data", "===================================================================================")
        log("setup_data", "Pytorch Model created:\n")
        log("setup_data", str(self.global_model))
        log("setup_data", "\n===================================================================================")

    def add_participant(self, user):
        train_loader, val_loader = self.get_user_dataloaders(user)
        
        if self.config.dataset == "mnist":
            _model = Net_MNIST().to(DEVICE)
        else:
            _model = Net_CIFAR().to(DEVICE)

        if USE_CUDA and COMPILE:
            _model = torch.compile(_model, mode="reduce-overhead")

        # Shared init: all participants start from the same global weights so
        # coordinate-wise FedAvg in the_merge stays valid (neuron alignment).
        _model.load_state_dict(self.global_model.state_dict())

        lr = 0.001 if self.config.dataset == "mnist" else 0.05
        optimizer = optim.SGD(_model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        l = len(self.participants)
        p = Participant.from_user(user, train_loader, val_loader, _model, optimizer, criterion)
        p.val_tensors = preload_to_gpu(val_loader, DEVICE) if USE_CUDA else None
        self.participants.append(p)

        attitude = (user.futureAttitude.name[0].upper() + user.futureAttitude.name[1:]).ljust(9)
        log("setup_contracts", f"Participant added: [{rb(attitude)}] {rb(user.display_label())}")

    # seed/allow_overlap/replication_factor forward to DataPartition for reproducible, optionally overlapping splits.
    def prepare_data_for_users(self, users, dataset_name, seed=42, allow_overlap=False, replication_factor=1.0):
        users = list(users)

        trainset, testset = get_cached_datasets(dataset_name)

        per_user_specs = (
            self.config.get_partition_specs(dataset_name)
            if self.config.partition_strategy == "per_user"
            else None
        )
        partitioner = DataPartition(
            validation_split=0.1,
            seed=seed,
            allow_overlap=allow_overlap,
            replication_factor=replication_factor,
            per_user_specs=per_user_specs,
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
        if USE_CUDA:
            self.test_tensors = preload_to_gpu(testloader, DEVICE)

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
        rep_factor = float(getattr(self.config, "replication_factor", 1.0))
        allow_overlap = bool(getattr(self.config, "allow_overlap", False))
        overlap_active = allow_overlap and rep_factor > 1.0
        effective_pool = int(dataset_size * rep_factor) if overlap_active else dataset_size
        pool_suffix = f" | overlap ON, rep={rep_factor:g} -> effective pool {effective_pool:,}" if overlap_active else ""
        log("setup_data",f"Dataset: {dataset_name} | {dataset_size:,} total samples | {num_classes} classes | ~{per_class:,} per class{pool_suffix}")
        log("setup_data")
        log("setup_data","Data split per user:")
        log("setup_data",
            "{:<4} {:<14} {:<16} {:>10} {:>10} {:>9}   {}".format(
                "Idx",
                "Name",
                "Address",
                "Config %",
                "Actual %",
                "Samples",
                "Rule",
            )
        )
        log("setup_data","-" * 90)

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

            user_idx = getattr(user, "number", user_id)
            user_name = (getattr(user, "partition_name", None) or "-")[:14]

            log("setup_data",
                "{:<4} {:<14} {:<16} {:>9.2f}% {:>9.2f}% {:>9,}   {}".format(
                    user_idx,
                    user_name,
                    user.address[0:14] + "...",
                    config_percent,
                    actual_percent,
                    split["num_samples"],
                    self.get_label_method(user),
                )
            )

            all_ids = split["train_ids"] + split["val_ids"]
            label_counts = self.count_labels(labels, all_ids)
            label_dist_rows.append((user_idx, user_name, label_counts))

            if user.flip_map:
                train_flip_counts = self.count_flipped_labels(labels, split["train_ids"], user.flip_map)
                val_flip_counts = self.count_flipped_labels(labels, split["val_ids"], user.flip_map)
                label_flip_rows.append((
                    user_idx, user_name, "Train",
                    self.format_flip_counts(train_flip_counts),
                    sum(train_flip_counts.values()),
                    self.ratio_percent(sum(train_flip_counts.values()), split["train_samples"]),
                ))
                label_flip_rows.append((
                    user_idx, user_name, "Val",
                    self.format_flip_counts(val_flip_counts),
                    sum(val_flip_counts.values()),
                    self.ratio_percent(sum(val_flip_counts.values()), split["val_samples"]),
                ))

        unique_ids = set()
        for user in users:
            split = user_splits[user.get_id_or_address()]
            unique_ids.update(split["train_ids"])
            unique_ids.update(split["val_ids"])
        unique_count = len(unique_ids)
        unmapped_unique = dataset_size - unique_count
        unmapped_pct = 100.0 * unmapped_unique / dataset_size

        log("setup_data","-" * 90)
        log("setup_data","Configured total: {:.2f}%  (sum of Config % column)".format(total_config_percent))
        if overlap_active:
            pool_pct = 100.0 * total_samples / effective_pool
            avg_dup = (total_samples / unique_count) if unique_count else 0.0
            log("setup_data","Assigned: {:>7,} / {:,} effective slots  ({:.2f}% of inflated pool, {:.2f}% of base dataset)".format(
                total_samples, effective_pool, pool_pct, total_actual_percent))
            log("setup_data","Unique:   {:>7,} / {:,} base samples covered  (avg {:.2f}x replication per assigned sample)".format(
                unique_count, dataset_size, avg_dup))
        else:
            log("setup_data","Assigned: {:>7,} / {:,} samples  ({:.2f}%)".format(total_samples, dataset_size, total_actual_percent))
        if unmapped_unique > 0:
            log("setup_data", "Unused:   {:>7,} / {:,} base samples  ({:.2f}%)  <- not assigned to any user (only_labels / label_distribution retention / pct<100)".format(
                unmapped_unique, dataset_size, unmapped_pct))
        log("setup_data")

        col_w = 6
        header = "{:<4} {:<14} " + " ".join(f"{'L' + str(l):>{col_w}}" for l in all_label_classes)
        log("setup_data",header.format("Idx", "Name"))
        log("setup_data","-" * (20 + col_w * len(all_label_classes)))
        for user_idx, user_name, label_counts in label_dist_rows:
            row = "{:<4} {:<14} " + " ".join(f"{label_counts.get(l, 0):>{col_w},}" for l in all_label_classes)
            log("setup_data",row.format(user_idx, user_name))

        if not label_flip_rows:
            return

        log("setup_data")
        log("setup_data","Label flips (samples changed per user):")
        log("setup_data","{:<4} {:<14} {:<5}   {:<24} {:>8} {:>11}".format("Idx", "Name", "Set", "Flip counts", "Changed", "% of set"))
        log("setup_data","-" * 75)
        for user_idx, user_name, set_name, flip_counts, changed_total, changed_percent in label_flip_rows:
            log("setup_data","{:<4} {:<14} {:<5}   {:<24} {:>8} {:>10.2f}%".format(
                user_idx, user_name, set_name, flip_counts, changed_total, changed_percent,
            ))
        log("setup_data","-" * 75)

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

        trainset, testset = get_cached_datasets(self.config.dataset)

        if _print:
            log("setup_data", "Data Loaded:")
            log("setup_data", "Nr. of images for training: {:,.0f}".format(len(trainset)))
            log("setup_data", "Nr. of images for testing:  {:,.0f}\n".format(len(testset)))

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
        if USE_CUDA and self.test_tensors is None:
            self.test_tensors = preload_to_gpu(testloader, DEVICE)
        return trainloaders, valloaders, testloader


    def federated_training(self):
        if debugging or globals.reuse_runs:
            log("round_training", b("\n================ SEQUENTIAL FEDERATED TRAINING START ================"))
            start_total = time.perf_counter()
            log("round_training", yellow(f"{'Debugging mode' if debugging else ''} {'and ' if debugging and globals.reuse_runs else ''}{'reuse_runs ' if globals.reuse_runs else ''}detected → running sequential training"))
            results = self.run_sequential()
        else:
            log("round_training", b("\n================ PARALLEL FEDERATED TRAINING START ================"))
            start_total = time.perf_counter()
            results = self.run_multi_processing()

        self.apply_training_results(results)
        if USE_CUDA:
            torch.cuda.empty_cache()
        total_time = time.perf_counter() - start_total

        log("round_training", b("=================== PARALLEL TRAINING END ===================\n"))
        log("round_training", green(f"Total federated training time: {total_time:.2f} seconds\n"))

    def finalize_paricipant_evaluation(self, participant): # Same as lines 294-296,306 in orgiginal code.
        loss, acc = self.runRepo.test(self.round,f"test-finalize_paricipant_evaluation-{participant.id}", participant.model, self._test_data, DEVICE) # TODO: Investigate if this should be user.val instead.
        participant._accuracy.append(acc) # Line 295 in original code # TODO: Investigate if this should be test and not validation accuracy.
        participant._loss.append(loss) # Line 296 in original code # TODO: Investigate if this should be test and not validation loss.
        participant.hashedModel = self.runRepo.get_hash(self.round, f"get_hash-finalize_paricipant_evaluation-{participant.id}", participant.model.state_dict())


    def apply_training_results(self, results):
        # Apply results back to participants
        participant_map = {x.id: x for x in self.participants}
        for user_id, state_dict, val_loss, val_acc in results:
            user = participant_map[user_id]
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
                    user.display_label(),
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

        # With only 1 worker there's no parallelism benefit from a Pool; the spawn
        # overhead just wastes time and memory (especially on Windows/ROCm).
        if num_processes == 1:
            return self.run_sequential()

        pool = ctx.Pool(processes=num_processes)
        start_pool = time.perf_counter()
        try:
            async_results = []
            for idx, user in enumerate(self.participants):
                device_id = idx % max(1, num_gpus)
                sd_cpu = {k: v.cpu() for k, v in user.model.state_dict().items()}  # safe copy

                if user.attitude == Attitude.Honest: # train
                    async_results.append(pool.apply_async(
                        train_user_proc,
                        (
                        user.id,
                        user.display_label(),
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
        finally:
            pool.close()
            pool.join()
        log("round_training", green(f"Parallel execution time: {time.perf_counter() - start_pool:.2f} seconds"))
        return results


    def let_malicious_users_do_their_work(self):
        for i in range(len(self.participants)):
            if self.participants[i].attitude == Attitude.Malicious:
                log("agent_behavior",red("{} ({}) going to provide random weights".format(
                    self.participants[i].display_label(),
                    self.participants[i].address[0:16]+"...",
                )))
                # noise_scale is set per-user (from the spec in per_user mode,
                # or from experiment_config.malicious_noise_scale in global mode).
                scale = self.participants[i].noise_scale
                if scale is None:
                    raise ValueError(
                        f"malicious participant {self.participants[i].display_label()} "
                        f"has no noise_scale set"
                    )
                manipulated_state_dict = manipulate(self.participants[i].model, scale=scale)
                self.participants[i].model.load_state_dict(manipulated_state_dict)
                self.participants[i].hashedModel = self.runRepo.get_hash(self.round, f"let_malicious_users_do_their_work-{self.participants[i].id}", self.participants[i].model.state_dict())
                loss, accuracy = self.runRepo.test(self.round,f"test-let_malicious_users_do_their_work-usermodel-{self.participants[i].id}", self.participants[i].model, self._test_data, DEVICE)
                log("agent_behavior","{:<17} {} ({}) |  Testing  | Accuracy {:>3.0f} % | Loss ∞\n".format(
                    "Account testing:   ",
                    self.participants[i].display_label(),
                    self.participants[i].address[0:16]+"...",
                    accuracy*100, loss))
                # TODO: Why is test_loss not used here?

    def update_users_attitude(self):
        for user in self.participants:
            if user.attitudeSwitch == self.round \
                and user.attitude != user.futureAttitude:
                log("agent_behavior",rb("{} ({}) going to switch attitude to {}".format(
                    user.display_label(),
                    user.address[0:16]+"...",
                    user.futureAttitude,
                )))
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
                # Per-user start_round drives when this freerider activates.
                start_round = participant.start_round
                if start_round is None:
                    raise ValueError(
                        f"freerider participant {participant.display_label()} "
                        f"has no start_round set"
                    )
                if self.round < start_round:
                    log("agent_behavior",yellow(
                        "{} ({}) waiting until round {} to start freeriding".format(
                            participant.display_label(),
                            participant.address[0:16] + "...",
                            start_round,
                        )
                    ))
                    new_state_dict = manipulate(participant.model)
                else:
                    new_state_dict = self._freerider_submit_with_noise(participant)


                participant.model.load_state_dict(new_state_dict)
                participant.hashedModel = self.runRepo.get_hash(self.round, f"get_hash-let_freerider_users_do_their_work-{participant.id}",participant.model.state_dict())
                loss, accuracy = self.runRepo.test(self.round,f"test-let_freerider_users_do_their_work-usermodel-{participant.id}", participant.model, self._test_data, DEVICE)
                log("agent_behavior","{:<17} {} ({}) |  Testing  | Accuracy {:>3.0f} % | Loss ∞\n".format(
                    "Account testing:   ",
                    participant.display_label(),
                    participant.address[0:16]+"...",
                    accuracy*100, loss))
                # TODO: Why is loss not used here?


    def _freerider_submit_with_noise(self, user):
        """Freerider reuses the global model with configurable noise."""

        # Per-user noise_scale: set from spec in per_user mode, from
        # experiment_config.freerider_noise_scale in global mode.
        scale = user.noise_scale
        if scale is None:
            raise ValueError(
                f"freerider participant {user.display_label()} has no noise_scale set"
            )
        if scale < 0:
            raise ValueError(
                f"freerider participant {user.display_label()}: noise_scale must be "
                f"non-negative, got {scale}"
            )

        if scale == 0: # Copy global model if noise is zero
            log("agent_behavior",yellow("{} ({}) resubmitting original model".format(
                user.display_label(),
                user.address[0:16]+"...",
            )))
            return {k: v.clone() for k, v in user.model.state_dict().items()}

        log("agent_behavior",red(
            "{} ({}) adding noise (scale={}) to global weights".format(
                user.display_label(),
                user.address[0:16]+"...",
                scale,
            )
        ))
        return manipulate(user.model, scale=scale)


    def the_merge(self, _users):
        # No qualified users → skip merge this round
        if not _users:
            log("round_models", "-----------------------------------------------------------------------------------")
            log("round_models", red("No participants qualified for merge this round – skipping aggregation"))
            log("round_models", "-----------------------------------------------------------------------------------\n")
            return

        ids, client_models = [], []
        for u in _users:
            ids.append(u.address)
            client_models.append(u.model)
            log("agent_behavior","{} ({}) participating in merge".format(u.display_label(), u.address[0:16]+"..."))
            #print(test(c[1],self.test,DEVICE))

        with torch.no_grad():
            global_dict = self.global_model.state_dict()
            client_state_dicts = [m.state_dict() for m in client_models]
            for k in global_dict.keys():
                stacked = torch.stack([
                    client_state_dicts[i][k].to(
                        device=global_dict[k].device,
                        dtype=global_dict[k].dtype
                    )
                    for i in range(len(client_models))
                ], dim=0)
                global_dict[k] = stacked.mean(0)
            self.global_model.load_state_dict(global_dict)

        loss, accuracy = self.runRepo.test(self.round,f"test-themerge-globalmodel", self.global_model, self._test_data, DEVICE)
        self.accuracy.append(accuracy)
        self.loss.append(loss)
        log("round_models", "-----------------------------------------------------------------------------------")
        log("round_models", b("Merged Model: Accuracy {:>3.0f} % | Loss {:>6,.2f}".format(accuracy*100,loss)))

        for u in self.participants:
            u.previousModel.load_state_dict(u.model.state_dict())
            u.model.load_state_dict(self.global_model.state_dict())

        log("round_models", "-----------------------------------------------------------------------------------\n")




    def exchange_models(self):
        log("round_models", "Users exchanging models...")
        for user in self.participants:
            user.userToEvaluate = []
            for j in self.participants:
                if user.model == j.model:
                    continue
                if j.model in user.userToEvaluate:
                    continue
                user.userToEvaluate.append(j)
        log("round_models", "-----------------------------------------------------------------------------------")



    def verify_models(self, on_chain_hashes):
        log("round_models", "Users verifying models...")
        for _user in self.participants:
            _user.cheater = []
            for user in _user.userToEvaluate:
                if not self.runRepo.get_hash(self.round, f"get_hash-verify_models-{user.id}", user.model.state_dict()) == on_chain_hashes[user.id]:
                    log("round_models",red(f"Account {_user.display_label()} (#{_user.number}): Account {user.display_label()} ({user.address[0:16]}...) could not provide the registered model"))
                    _user.cheater.append(user)

        log("round_models", "-----------------------------------------------------------------------------------")

    def get_global_model_hash(self):
        return get_hash(self.global_model.state_dict())

    def evaluation(self):
        log("round_models", "Users evaluating models...")
        start_total = time.perf_counter()

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
            val_tensors = getattr(feedbackGiver, 'val_tensors', None)
            valloader = val_tensors if val_tensors is not None else feedbackGiver.val
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

                # Vote threshold reference (raw fraction). prev_acc gets scaled below for honest givers,
                # so capture baseline first.
                if self.config.vote_baseline == "prev_global":
                    baseline_acc = prev_acc
                else:
                    baseline_acc = feedbackGiver.currentAcc

                if not bad_att and not free_att:
                    prev_acc = round(prev_acc * 100 * scalar)
                    prev_loss = safe_scale(prev_loss, scalar, MAX_UINT16_SIZE)

                if bad_att:
                    matrices.feedback_matrix[giver_idx, user_idx] = -1
                    matrices.accuracy_matrix[giver_idx, user_idx] = 0
                    matrices.loss_matrix[giver_idx, user_idx] = 65535
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

                elif accuracy > baseline_acc - 0.07:  # 7% Worse: 0.07
                    matrices.feedback_matrix[giver_idx, user_idx] = 1
                    matrices.accuracy_matrix[giver_idx, user_idx] = round(accuracy * 100 * scalar)
                    matrices.loss_matrix[giver_idx, user_idx] = safe_scale(loss, scalar, MAX_UINT16_SIZE)
                    matrices.prev_accuracies[giver_idx] = prev_acc
                    matrices.prev_losses[giver_idx] = prev_loss

                elif accuracy > baseline_acc - 0.14:  # 14% Worse: 0.14
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

        log("round_matrices", "FEEDBACK MATRIX:")
        log("round_matrices", matrices.feedback_matrix)
        log("round_matrices", "-----------------------------------------------------------------------------------")
        log("round_matrices", "ACCURACY MATRIX:")
        log("round_matrices", matrices.accuracy_matrix)
        log("round_matrices", "-----------------------------------------------------------------------------------")
        log("round_matrices", "LOSS MATRIX:")
        log("round_matrices", matrices.loss_matrix)
        log("round_matrices", "-----------------------------------------------------------------------------------")
        log("round_matrices", "PREVIOUS ACCURACIES:")
        log("round_matrices", matrices.prev_accuracies)
        log("round_matrices", "-----------------------------------------------------------------------------------")
        log("round_matrices", "PREVIOUS LOSSES:")
        log("round_matrices", matrices.prev_losses)
        log("round_matrices", "-----------------------------------------------------------------------------------")
        log("round_matrices", f"TOTAL evaluation time: {time.perf_counter() - start_total:.3f}s")
        return matrices

    def get_participant(self, address_or_id, participants = None):
        if participants is None:
            participants = self.participants
        p = next((x for x in participants if x.id == address_or_id), None)
        if p:
            return p
        p = next((x for x in participants if x.address == address_or_id), None)
        if p:
            return p
        as_str = str(address_or_id)
        return next((x for x in participants if x.guid is not None and x.guid == as_str), None)

    @property
    def _test_data(self):
        """Return pre-loaded GPU tensors when available, otherwise the DataLoader."""
        return self.test_tensors if self.test_tensors is not None else self.test

    def setup_replay(self, filename, config: ExperimentConfiguration, path):

        if ReplayMode.PlayBack in globals.reuse_runs and filename.is_file():
            self.runRepo: ITestAndTrainer = RunRepo(config, filename)
            self.replaying = True
        else:
            self.runRepo: ITestAndTrainer = PyTorchTrainer(config, filename)

        loss, accuracy = self.runRepo.test(0, f"test-setup_replay-globalmodel", self.global_model, self._test_data, DEVICE)

        self.accuracy = [accuracy]
        self.loss = [loss]

        self.round = 1
        self.runRepo.save(-1, f"save-setup_replay-path", path)

    def cleanup(self):
        """Shut down DataLoader worker processes (pt_data_worker / forked python).

        Call this after each experiment run. With persistent_workers=True the
        workers live as long as the DataLoader object does; without an explicit
        shutdown they accumulate across loop iterations until GC finally fires.
        """
        import gc

        all_loaders: list = (
            list(self.train_by_user_id.values())
            + list(self.val_by_user_id.values())
        )
        if isinstance(self.DATA, tuple):
            for group in self.DATA:
                if isinstance(group, list):
                    all_loaders.extend(group)
                elif group is not None:
                    all_loaders.append(group)
        for p in self.participants:
            if p.train is not None:
                all_loaders.append(p.train)
            if p.val is not None:
                all_loaders.append(p.val)

        seen: set = set()
        for loader in all_loaders:
            if loader is None or id(loader) in seen:
                continue
            seen.add(id(loader))
            it = getattr(loader, '_iterator', None)
            if it is not None:
                if hasattr(it, '_shutdown_workers'):
                    try:
                        it._shutdown_workers()
                    except Exception:
                        logerror("Failed to shut down workers")
                        pass
                try:
                    loader._iterator = None
                    del loader
                except Exception:
                    logerror("Failed to shutdown loader")
                    pass

        self.train_by_user_id.clear()
        del self.train_by_user_id
        self.val_by_user_id.clear()
        del self.val_by_user_id
        
        for p in self.participants:
            p.train = None
            del p.train
            p.val = None
            del p.val
        self.DATA = None
        del self.DATA
        self.train = None
        del self.train
        self.val = None
        del self.val
        self.test = None
        del self.test
        self.test_tensors = None
        del self.test_tensors

        gc.collect()

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

def preload_to_gpu(dataloader, device):
    """Load an entire DataLoader into GPU memory as a single (images, labels) tensor pair."""
    images_list, labels_list = [], []
    with torch.no_grad():
        for imgs, lbls in dataloader:
            images_list.append(imgs)
            labels_list.append(lbls)
    return (
        torch.cat(images_list).to(device, non_blocking=NON_BLOCKING),
        torch.cat(labels_list).to(device, non_blocking=NON_BLOCKING),
    )


# PYTORCH FUNCTIONS
def train(net, trainloader: torch.utils.data.DataLoader, epochs: int, device: torch.device, lr: float = 0.001) -> None:

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    scaler = torch.amp.GradScaler(enabled=AMP)

    net.train()

    for _ in range(epochs):
        for images, labels in trainloader:
            images = images.to(device, non_blocking=NON_BLOCKING)
            labels = labels.to(device, non_blocking=NON_BLOCKING)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=AMP):
                outputs = net(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

def test(net, testloader, device):
    criterion = nn.CrossEntropyLoss()
    net.eval()

    with torch.no_grad():
        # Fast path: pre-loaded GPU tensors — one forward pass, no per-batch CPU↔GPU transfers
        if isinstance(testloader, tuple):
            images, labels = testloader
            with torch.autocast(device_type=device.type, enabled=AMP):
                outputs = net(images)
            loss = criterion(outputs, labels).item()
            accuracy = (outputs.argmax(dim=1) == labels).float().mean().item()
            return loss, accuracy

        correct = 0
        total = 0
        loss = 0.0
        for images, labels in testloader:
            images = images.to(device, non_blocking=NON_BLOCKING)
            labels = labels.to(device, non_blocking=NON_BLOCKING)

            with torch.autocast(device_type=device.type, enabled=AMP):
                outputs = net(images)

            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item() * labels.size(0)

            predicted = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss /= total
    accuracy = correct / total

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

def train_user_proc(user_addr, user_label, model_state, train_ds, val_ds, epochs, device_id, dataset, batchsize, pin_memory,
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

    if USE_CUDA and COMPILE:
            model = torch.compile(model, mode="reduce-overhead")

    model.load_state_dict(model_state)
    model.to(device)

    # Rebuild dataloaders inside the process
    train_loader = DataLoader(train_ds, batch_size=batchsize, shuffle=shuffle,
                              pin_memory=pin_memory)  # TODO: Investigate if this breaks something
    val_loader = DataLoader(val_ds, batch_size=batchsize, shuffle=False,
                            pin_memory=pin_memory)  # TODO: Investigate if this breaks something

    lr = 0.001 if dataset == "mnist" else 0.01
    train(model, train_loader, epochs, device, lr=lr)  # Line 285 in original code
    val_loss, val_acc = test(model, val_loader, device)  # Line 286 in original code

    # del: Mark for GC
    del train_loader
    del val_loader

    log("round_training", f"[{device_label(device, device_id)}] {user_label:<32} ({user_addr})  done | Acc: {val_acc:.3f}  Loss: {val_loss:.3f}")

    # Ensure all GPU work is complete before worker exits
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return user_addr, model.state_dict(), val_loss, val_acc

def print_training_mode(num_gpus: int, num_processes: int):
    """Prints a clean status message describing how training will run."""
    if num_gpus >= 2:
        log("round_training", green(f"Detected {num_gpus} GPU(s) → Parallel multi-GPU training"))

    elif num_gpus == 1:
        if num_processes > 1:
            log("round_training", yellow(
                f"Detected 1 GPU → Parallel training on one GPU (shared across {num_processes} workers)"
            ))
        else:
            log("round_training", green("Detected 1 GPU → Sequential GPU training"))

    else:  # CPU-only
        if num_processes > 1:
            log("round_training", yellow(
                f"Detected 0 GPU(s) → Parallel CPU training ({num_processes} workers)"
            ))
        else:
            log("round_training", red("Detected 0 GPU(s) → Sequential CPU mode"))



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