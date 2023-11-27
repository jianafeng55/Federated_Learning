class ConstantLrScheduler:
    def __init__(self, inital_lr) -> None:
        self.initial_lr = inital_lr

    def get_lr(self, round_idx):
        return self.initial_lr
