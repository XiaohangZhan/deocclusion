import torch
from bisect import bisect_right

class _LRScheduler(object):
    def __init__(self, optimizer, last_iter=-1):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_iter == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_iter = last_iter

    def _get_new_lr(self):
        raise NotImplementedError

    def get_lr(self):
        return list(map(lambda group: group['lr'], self.optimizer.param_groups))

    def step(self, this_iter=None):
        if this_iter is None:
            this_iter = self.last_iter + 1
        self.last_iter = this_iter
        for param_group, lr in zip(self.optimizer.param_groups, self._get_new_lr()):
            param_group['lr'] = lr

class _WarmUpLRSchedulerOld(_LRScheduler):

    def __init__(self, optimizer, base_lr, warmup_lr, warmup_steps, last_iter=-1):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        if warmup_steps == 0:
            self.warmup_lr = base_lr
        else:
            self.warmup_lr = warmup_lr
        super(_WarmUpLRSchedulerOld, self).__init__(optimizer, last_iter)
    
    def _get_warmup_lr(self):
        if self.warmup_steps > 0 and self.last_iter < self.warmup_steps:
            # first compute relative scale for self.base_lr, then multiply to base_lr
            scale = ((self.last_iter/self.warmup_steps)*(self.warmup_lr - self.base_lr) + self.base_lr)/self.base_lr
            #print('last_iter: {}, warmup_lr: {}, base_lr: {}, scale: {}'.format(self.last_iter, self.warmup_lr, self.base_lr, scale))
            return [scale * base_lr for base_lr in self.base_lrs]
        else:
            return None

class _WarmUpLRScheduler(_LRScheduler):

    def __init__(self, optimizer, base_lr, warmup_lr, warmup_steps, last_iter=-1):
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        self.warmup_steps = warmup_steps
        assert isinstance(warmup_lr, list)
        assert isinstance(warmup_steps, list)
        assert len(warmup_lr) == len(warmup_steps)
        super(_WarmUpLRScheduler, self).__init__(optimizer, last_iter)
    
    def _get_warmup_lr(self):
        pos = bisect_right(self.warmup_steps, self.last_iter)
        if pos >= len(self.warmup_steps):
            return None
        else:
            if pos == 0:
                curr_lr = self.base_lr + self.last_iter * (self.warmup_lr[pos] - self.base_lr) / self.warmup_steps[pos]
            else:
                curr_lr = self.warmup_lr[pos - 1] + (self.last_iter - self.warmup_steps[pos - 1]) * (self.warmup_lr[pos] - self.warmup_lr[pos - 1]) / (self.warmup_steps[pos] - self.warmup_steps[pos - 1])
        scale = curr_lr / self.base_lr
        return [scale * base_lr for base_lr in self.base_lrs]

class StepLRScheduler(_WarmUpLRScheduler):
    def __init__(self, optimizer, milestones, lr_mults, base_lr, warmup_lr, warmup_steps, last_iter=-1):
        super(StepLRScheduler, self).__init__(optimizer, base_lr, warmup_lr, warmup_steps, last_iter)

        assert len(milestones) == len(lr_mults), "{} vs {}".format(milestones, lr_mults)
        for x in milestones:
            assert isinstance(x, int)
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.lr_mults = [1.0]
        for x in lr_mults:
            self.lr_mults.append(self.lr_mults[-1]*x)
    
    def _get_new_lr(self):
        warmup_lrs = self._get_warmup_lr()
        if warmup_lrs is not None:
            return warmup_lrs

        pos = bisect_right(self.milestones, self.last_iter)
        if len(self.warmup_lr) == 0:
            scale = self.lr_mults[pos]
        else:
            scale = self.warmup_lr[-1] * self.lr_mults[pos] / self.base_lr
        return [base_lr * scale for base_lr in self.base_lrs]
