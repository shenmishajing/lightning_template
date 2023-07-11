import torch


class Timer:
    duration_ms = None

    def __enter__(self):
        torch.cuda.synchronize()
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, *args):
        self.end.record()
        torch.cuda.synchronize()
        self.duration_ms = self.start.elapsed_time(self.end)

    @property
    def duration(self):
        return self.duration / 1000

    def iters_per_second(self, batch_size=1):
        return 1000 / self.duration_ms * batch_size
