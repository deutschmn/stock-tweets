from typing import List

from torch.utils.data import Dataset

from src.data.movement import Movement


class MovementDataset(Dataset):
    def __init__(self, movements: List[Movement]):
        super()
        self.movements = movements

    def __len__(self) -> int:
        return len(self.movements)

    def __getitem__(self, idx: int) -> Movement:
        movement = self.movements[idx]

        return (
            list(movement.tweets["text"]),
            list(movement.tweets["user_followers"]),
            movement.price["movement percent"],
        )