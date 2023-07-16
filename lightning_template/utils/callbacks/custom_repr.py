from lightning.pytorch import Callback


def custom_repr(original_repr):
    def wrapper(self):
        shape_str = ""
        if hasattr(self, "shape"):
            shape_str = f"{tuple(self.shape)}"
        elif hasattr(self, "__len__"):
            shape_str = f"{len(self)}"
        return f"{shape_str}{original_repr(self)}"

    return wrapper


class CustomReprCallback(Callback):
    def __init__(self, classes=None) -> None:
        super().__init__()
        if classes is None:
            classes = ["torch.Tensor"]
        self.classes = classes

    def setup(self, *args, **kwargs) -> None:
        for class_path in self.classes:
            class_module, class_name = class_path.rsplit(".", 1)
            module = __import__(class_module, fromlist=[class_name])
            args_class = getattr(module, class_name)
            args_class.__repr__ = custom_repr(args_class.__repr__)
