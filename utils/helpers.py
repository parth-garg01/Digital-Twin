import time


class ProgressTracker:
    def __init__(self, total_steps):
        self.total_steps = max(1, total_steps)
        self.current_step = 0

    def advance(self, stage, detail=""):
        self.current_step += 1
        percent = min(100, int(round((self.current_step / self.total_steps) * 100)))
        suffix = f" {detail}" if detail else ""
        print(f"[{percent}%] {stage}...{suffix}")


class StageTimer:
    def __init__(self):
        self.timings = {}

    def time_block(self, name):
        return _TimedBlock(self.timings, name)


class _TimedBlock:
    def __init__(self, timings, name):
        self.timings = timings
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start
        self.timings[self.name] = self.timings.get(self.name, 0.0) + elapsed
