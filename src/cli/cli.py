from pytorch_lightning.utilities.cli import LightningCLI

from src.cli.wandb import ConfigWandbMixin


class MovementCLI(ConfigWandbMixin, LightningCLI):
    pass
