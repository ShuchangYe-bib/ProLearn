import torch
from torch.utils.data import DataLoader


class MMSegDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.custom_collate_fn  # Use the custom collate function
        )

    @staticmethod
    def custom_collate_fn(batch):
        # Custom collate function to handle batches with None values
        collated_batch = {}
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            # Check if the first element is None; if so, keep all as None
            if values[0] is None:
                collated_batch[key] = None  # Keeps [None, None, None, ...] format
            else:
                collated_batch[key] = torch.utils.data.default_collate(values)
        return collated_batch

