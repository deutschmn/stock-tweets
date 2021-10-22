from pytorch_lightning.utilities.cli import LightningCLI

from src.cli.wandb import ConfigWandbMixin


class MovementCLI(ConfigWandbMixin, LightningCLI):
    def fit(self):
        super().fit()

        # always test after fit
        self.trainer.test(**self.fit_kwargs)

    def prepare_fit_kwargs(self):
        super().prepare_fit_kwargs()

        # if configured, pass test loader as second val loader
        if self.model.test_as_second_val_loader:
            self.datamodule.prepare_data()
            self.datamodule.setup()

            self.fit_kwargs["datamodule"] = None
            self.fit_kwargs["train_dataloaders"] = self.datamodule.train_dataloader()
            self.fit_kwargs["val_dataloaders"] = [
                self.datamodule.val_dataloader(),
                self.datamodule.test_dataloader(),
            ]
