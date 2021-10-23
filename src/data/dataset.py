from typing import List

from torch.utils.data import Dataset

from src.data.movement import ClassifiedMovement


class MovementDataset(Dataset):
    def __init__(self, movements: List[ClassifiedMovement]):
        super()
        self.movements = movements

    def __len__(self) -> int:
        return len(self.movements)

    def __getitem__(self, idx: int) -> ClassifiedMovement:
        return self.movements[idx]
