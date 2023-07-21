class SplitNameMixin:
    def __init__(self) -> None:
        self.split_names = ["train", "val", "test", "predict"]

    def _get_split_names(self, stage=None):
        if self.trainer.overfit_batches > 0:
            split_names = ["train"]
        elif stage is None:
            split_names = self.split_names
        elif stage == "fit":
            split_names = ["train", "val"]
        elif stage == "validate":
            split_names = ["val"]
        else:
            split_names = [stage.lower()]
        return split_names

    def setup(self, stage=None):
        self.split_names = self._get_split_names(stage)
