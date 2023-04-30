from lightning.pytorch.tuner import Tuner

from lightning_template.utils import LightningCLI as _LightningCLI


class LightningCLI(_LightningCLI):
    def before_instantiate_classes(self) -> None:
        """Implement to run some code before instantiating the classes."""
        super().before_instantiate_classes()
        config = (
            self.config
            if "subcommand" not in self.config
            else self.config[self.config["subcommand"]]
        )
        if config.get("trainer") is not None:
            config["trainer"]["strategy"] = "auto"
            config["trainer"]["devices"] = 1


def main():
    cli = LightningCLI(run=False)
    # Create a tuner for the trainer
    tuner = Tuner(cli.trainer)

    # Auto-scale batch size with binary search
    tuner.scale_batch_size(
        cli.model, datamodule=cli.datamodule, mode="binsearch", steps_per_trial=10
    )


if __name__ == "__main__":
    main()
