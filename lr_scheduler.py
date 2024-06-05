import numpy as np
from torch.optim.lr_scheduler import LambdaLR

class LinearWarmUpCosineAnnealingLR(LambdaLR):
    def __init__(self, optimizer, *, peak, final, warm_up_steps, max_steps, init=1e-8, offset=0, epoch_size=0, **kwargs):
        assert peak >= final >= init >= 0
        assert max_steps >= warm_up_steps
        self.init = init
        self.peak = peak
        self.final = final
        self.warm_up_steps = warm_up_steps
        self.max_steps = max_steps
        self.offset = offset
        self.epoch_size = epoch_size
        kwargs['optimizer'] = optimizer
        kwargs['lr_lambda'] = self._step_inner
        super().__init__(**kwargs)

    def _step_inner(self, steps):
        steps += self.offset
        if self.epoch_size > 0:
            steps %= self.epoch_size
        # print(f"Steps: {steps}, Offset: {self.offset}, Epoch size: {self.epoch_size}")

        if self.warm_up_steps > 0 and steps < self.warm_up_steps:
            lr = self.init + (self.peak - self.init) / self.warm_up_steps * steps
            # print(f"Warm-up phase: LR={lr}")
            return lr

        if steps < self.max_steps:
            cos_steps = steps - self.warm_up_steps
            cos_max_steps = self.max_steps - self.warm_up_steps
            lr = self.final + 0.5 * (self.peak - self.final) * (1 + np.cos(cos_steps / cos_max_steps * np.pi))
            # print(f"Cosine annealing phase: LR={lr}, cos_steps={cos_steps}, cos_max_steps={cos_max_steps}")
            return lr
        self.final = 2e-6
        # print(f"Final phase: LR={self.final}")
        return self.final
