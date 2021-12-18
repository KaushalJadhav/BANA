from torch.optim.lr_scheduler import _LRScheduler


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, step_size, iter_max, power, last_epoch=-1):
        super(PolynomialLR, self).__init__(optimizer, last_epoch)
        self.optimizer=optimizer 

    def polynomial_decay(self, lr,last_epoch,iter_max,power):
        return lr * (1 - float(self.last_epoch) / self.iter_max) ** power

    def __call__(self,last_epoch,step_size,iter_max,power):
        if (
            (last_epoch == 0)
            or (last_epoch % step_size != 0)
            or (last_epoch > iter_max)
        ):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [self.polynomial_decay(lr,last_epoch,iter_max,power) for lr in self.base_lrs]