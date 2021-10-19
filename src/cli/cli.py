from pytorch_lightning.utilities.cli import LightningCLI

from src.cli.wandb import ConfigWandbMixin


class MovementCLI(ConfigWandbMixin, LightningCLI):
    def fit(self):
        super().fit()

        # always test after fit
        self.trainer.test(**self.fit_kwargs)
