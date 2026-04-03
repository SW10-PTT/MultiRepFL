import random
from torch.utils.data import Subset
from torchvision import datasets, transforms


class DataPartition:
    def __init__(self, participants_config, seed=42):
        self.participants_config = participants_config
        self.seed = seed

    def get_num_participants(self):
        return len(self.participants_config)
    
    def validate_config(self):
        total = sum(cfg["dataSplit"] for cfg in self.participants_config.values())
        if total > 100:
            raise ValueError(f"Total dataSplit is {total}, must not exceed 100")
        
    def split_mnist(self):
        self.validate_config()

        dataset = dataset.MNIST(
            root="./data", 
            train=True, 
            download=True,
            transform=transforms.ToTensor()
        )

        indices = list(range(len(dataset)))
        random.Random(self.seed).shuffle(indices)

        result = {}
        start = 0

        for address, cfg in self.participants_config.items():
            amount = int(len(dataset) * (cfg["dataSplit"] / 100))
            part_indices = indices[start:start + amount]
            start += amount

            result[address] = Subset(dataset, part_indices)

        return result