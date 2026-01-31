from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
    
from torch import Tensor

def calc_cross_entropy_loss(inputs: Tensor, targets: Tensor):
    max_logits = torch.max(inputs, dim=-1, keepdim=True).values
    shifted_logits = inputs - max_logits

    log_accumulate_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1, keepdim=True)) + max_logits

    target_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1))
    loss = log_accumulate_exp - target_logits

    return torch.mean(loss)

def learning_rate_schedule(alpha_max, alpha_min, t, t_w, t_c):
    if t < t_w:
        alpha_t = t * alpha_max / t_w
    elif t > t_c:
        alpha_t = alpha_min
    else:
        alpha_t = alpha_min + 0.5 * (1 + math.cos(math.pi * (t - t_w) / (t_c - t_w))) * (alpha_max - alpha_min)

    return alpha_t

class SGD(torch.optim.Optimizer):
    # 随机梯度下降
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"无效的学习率：{lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]  # 获取学习率
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # 获取与参数p相关的状态
                t = state.get("t", 0)  # 从状态中获取迭代次数，若不存在则初始化为0
                grad = p.grad.data  # 获取损失相对于参数p的梯度
                # 原地更新权重张量
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1  # 迭代次数加1
        return loss

class AdamW(torch.optim.Optimizer):
    # SGD + RMSProp + 权重衰减
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for elem in group["params"]:
                if elem.grad is None:
                    continue

                grad = elem.grad.data
                state = self.state[elem]

                if len(state) == 0:
                    state["step"] = 0 # 记录训练到了第几步
                    state["m"] = torch.zeros_like(elem.data) # m 初始全是 0
                    state["v"] = torch.zeros_like(elem.data) # v 初始全是 0      

                m = state["m"]
                v = state["v"]

                state["step"] += 1
                times_recorder = state["step"]

                # 一次矩 m 算法：m = beta1 * m + (1 - beta1) * g
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # 二次矩 v 算法：v = beta2 * v + (1 - beta2) * g^2
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                correction1 = 1 - beta1**times_recorder
                correction2 = 1 - beta2**times_recorder

                # 结合步长调整公式：alpha_t = alpha * sqrt(1-beta2^t) / (1-beta1^t)
                step_size = lr * math.sqrt(correction2) / correction1

                # p = p - step_size * m / (sqrt(v) + epsilon)
                elem.data.addcdiv_(m, v.sqrt().add_(eps), value=-step_size)

                # adam-W里面W的来历：减去权重
                if wd != 0:
                    elem.data.add_(elem.data, alpha=-lr * wd)

        return loss
"""
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=100) 
for t in range(10):
    opt.zero_grad()
    loss = (weights**2).mean()
    print(f"Iteration {t}, Loss: {loss.item()}")
    loss.backward()
    opt.step()

    # lr = 1：龟速衰减
        # Iteration 0, Loss: 23.095903396606445
        # Iteration 1, Loss: 22.181303024291992
        # Iteration 2, Loss: 21.558359146118164
        # Iteration 3, Loss: 21.063364028930664
        # Iteration 4, Loss: 20.644203186035156
        # Iteration 5, Loss: 20.276559829711914
        # Iteration 6, Loss: 19.946796417236328
        # Iteration 7, Loss: 19.64636993408203
        # Iteration 8, Loss: 19.3695125579834
        # Iteration 9, Loss: 19.112112045288086
    # lr = 10：衰减太慢
        # Iteration 0, Loss: 26.985790252685547
        # Iteration 1, Loss: 17.270906448364258
        # Iteration 2, Loss: 12.731374740600586
        # Iteration 3, Loss: 9.96094036102295
        # Iteration 4, Loss: 8.068361282348633
        # Iteration 5, Loss: 6.689596652984619
        # Iteration 6, Loss: 5.641787528991699
        # Iteration 7, Loss: 4.821068286895752
        # Iteration 8, Loss: 4.163372039794922
        # Iteration 9, Loss: 3.6267592906951904
    # lr = 100：😆 很不戳
        # Iteration 0, Loss: 25.894155502319336
        # Iteration 1, Loss: 25.894149780273438
        # Iteration 2, Loss: 4.442733287811279
        # Iteration 3, Loss: 0.10632462799549103
        # Iteration 4, Loss: 6.605827314157037e-17
        # Iteration 5, Loss: 7.362604351388022e-19
        # Iteration 6, Loss: 2.479247865487669e-20
        # Iteration 7, Loss: 1.4769050333450928e-21
        # Iteration 8, Loss: 1.2669845773834185e-22
        # Iteration 9, Loss: 1.4077606240068893e-23
    # lr = 1000: 步子太大 跳过最小值点 直接发散了
        # Iteration 0, Loss: 21.652320861816406
        # Iteration 1, Loss: 7816.4873046875
        # Iteration 2, Loss: 1350030.375
        # Iteration 3, Loss: 150176464.0
        # Iteration 4, Loss: 12164292608.0
        # Iteration 5, Loss: 767706202112.0
        # Iteration 6, Loss: 39411525877760.0
        # Iteration 7, Loss: 1695652417372160.0
        # Iteration 8, Loss: 6.249813070839808e+16
        # Iteration 9, Loss: 2.0068844232318648e+18
"""