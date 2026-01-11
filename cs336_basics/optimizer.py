import math
from typing import Iterable, Dict, Any
import torch


class AdamW(torch.optim.Optimizer):
    """
    Minimal AdamW implementation compatible with CS336 tests.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                # Adam update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1

                # Weight decay (AdamW style)
                if wd != 0:
                    p.add_(p, alpha=-lr * wd)

                p.addcdiv_(exp_avg, denom, value=-step_size)


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    CS336 / GPT-style cosine learning rate schedule with linear warmup.
    """

    # 1. Linear warmup
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters

    # 2. Cosine decay
    t = it - warmup_iters
    T = cosine_cycle_iters - warmup_iters

    if t <= T:
        cosine = math.cos(math.pi * t / T)
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + cosine)

    # 3. After cosine decay
    return min_learning_rate


def save_checkpoint(model, optimizer, out, **kwargs):
    """
    Save the model and optimizer state to the given path.
    Accepts extra kwargs like 'iteration' to be compatible with test.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    # 保存 iteration（可选）
    if "iteration" in kwargs:
        checkpoint["iteration"] = kwargs["iteration"]

    torch.save(checkpoint, out)


def load_checkpoint(model, optimizer, src, **kwargs):
    """
    Load model and optimizer state from checkpoint.
    Returns checkpoint dict (can contain 'iteration').
    
    Args:
        model: torch.nn.Module
        optimizer: torch optimizer
        src: path to checkpoint file
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("iteration", None)


