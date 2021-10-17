from src.cli.cli import MovementCLI
from src.data.datamodule import MovementDataModule
from src.model import AttentionClassifier


if __name__ == "__main__":
    cli = MovementCLI(
        model_class=AttentionClassifier,
        datamodule_class=MovementDataModule,
        save_config_callback=None,
    )
