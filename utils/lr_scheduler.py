from torch.optim.lr_scheduler import _LRScheduler


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer,iter_max=-1,last_epoch=-1):
        super(PolynomialLR, self).__init__(optimizer, last_epoch) 
        self.iter_max=iter_max

    def polynomial_decay(self, lr,power):
        return lr * (1 - float(self.last_epoch) / self.iter_max) ** power

    def __call__(self,step_size,power):
        if (
            (self.last_epoch == 0)
            or (self.last_epoch % step_size != 0)
            or (self.last_epoch > self.iter_max)
        ):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [self.polynomial_decay(lr,self.iter_max,power) for lr in self.base_lrs]