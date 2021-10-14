from pytorch_lightning.loggers.wandb import WandbLogger


class ConfigWandbMixin:
    def before_fit(self):
        super().before_fit()
        for logger in filter(lambda l: isinstance(l, WandbLogger), self.trainer.logger):
            logger.log_hyperparams({
                "model": type(self.model).__name__,
                "datamodule": type(self.datamodule).__name__,
            })
            logger.log_hyperparams(self.config)
            logger.watch(self.model)
