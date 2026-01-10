import torch

def cross_entropy(logits, targets):
    """
    logits: (B, T, Vocab)
    targets: (B, T)
    """
    # 展平
    logits = logits.view(-1, logits.size(-1))
    targets = targets.view(-1)
    
    # 手动计算 LogSoftmax 并提取对应目标的 log-prob
    # 用 max 防止数值溢出
    logits_max = torch.max(logits, dim=-1, keepdim=True)[0]
    log_sum_exp = torch.log(torch.sum(torch.exp(logits - logits_max), dim=-1, keepdim=True))
    log_probs = (logits - logits_max) - log_sum_exp
    
    # 提取 target 对应的 log-probs
    loss = -log_probs[torch.arange(targets.size(0)), targets].mean()
    return loss

# AdamW 优化器基架 (继承自 torch.optim.Optimizer)
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                
                # 权重衰减 (Weight Decay)
                p.mul_(1 - group['lr'] * group['weight_decay'])
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                
                m, v = state['m'], state['v']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # 更新一阶和二阶矩
                m.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                
                # 偏差修正
                m_hat = m / (1 - beta1 ** state['step'])
                v_hat = v / (1 - beta2 ** state['step'])
                
                p.addcdiv_(m_hat, torch.sqrt(v_hat) + group['eps'], value=-group['lr'])