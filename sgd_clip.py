import torch
from torch.optim import Optimizer


class SGDClipGrad(Optimizer):
    """Implements stochastic gradient descent with clipped gradient. """

    def __init__(
        self, params, lr, weight_decay=0, clipping_param=0, algorithm='local_clip'
    ):
        if lr and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if clipping_param < 0.0:
            raise ValueError("Invalid clipping_param value: {}".format(clipping_param))

        defaults = dict(lr=lr, weight_decay=weight_decay, clipping_param=clipping_param)
        super(SGDClipGrad, self).__init__(params, defaults)

        self.algorithm = algorithm

    @torch.no_grad()
    def step(
        self,
        local_correction=None,
        global_correction=None,
        closure=None,
    ):
        """Performs a single optimization step.
        Arguments:
            local_correction (List[torch.Tensor]): Subtracted from gradient for
                SCAFFOLD-style corrections.
            global_correction (List[torch.Tensor]): Added to gradient for SCAFFOLD-style
                corrections.
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Compute clipping coefficient and update.
        local_update_l2_norm_sq = 0.0
        i = 0
        for group in self.param_groups:
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                # Correct gradient, if necessary.
                correct = (self.algorithm == "episode")
                correct = correct or (self.algorithm == "scaffold" and local_correction is not None)
                correct = correct or (self.algorithm == "episode_mem" and local_correction is not None)
                if correct:
                    d_p = d_p.add(global_correction[i] - local_correction[i])

                param_state['update'] = torch.clone(d_p).detach()
                local_update_l2_norm_sq += torch.sum(d_p.data * d_p.data)
                i += 1

        local_update_l2_norm = torch.sqrt(local_update_l2_norm_sq).item()
        episode_like = self.algorithm == "episode" or (self.algorithm == "episode_mem" and local_correction is not None)
        if episode_like:
            global_update_l2_norm = torch.sqrt(
                torch.sum(torch.cat([g.view(-1) for g in global_correction]) ** 2)
            )

        # Compute update size.
        clipping_coeff = group['clipping_param'] / (1e-10 + local_update_l2_norm)
        indicator = global_update_l2_norm if episode_like else local_update_l2_norm
        clip = indicator > group['clipping_param'] / group['lr']
        lr = clipping_coeff if clip else group['lr']

        # Apply update.
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                p.add_(param_state["update"], alpha=-lr)

        return loss, int(clip)
