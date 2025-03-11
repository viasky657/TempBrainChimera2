import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

class StableMax(nn.Module):
    """
    StableMax: A numerically stable alternative to Softmax that prevents Softmax Collapse.
    
    As described in the paper "Grokking at the Edge of Numerical Stability", StableMax
    uses a function s(x) instead of exp(x) that grows linearly for x >= 0 and approaches
    zero more slowly for x < 0, reducing the risk of numerical instability.
    """
    def __init__(self):
        super(StableMax, self).__init__()
    
    def forward(self, x):
        # For x >= 0: s(x) = x + 1
        # For x < 0: s(x) = 1/(1-x)
        positive_mask = (x >= 0).float()
        negative_mask = (x < 0).float()
        
        s_x = positive_mask * (x + 1) + negative_mask * (1.0 / (1.0 - x))
        
        # Compute StableMax similar to Softmax: s(xi) / sum(s(xj))
        sum_s_x = torch.sum(s_x, dim=-1, keepdim=True)
        return s_x / sum_s_x

class StableCrossEntropyLoss(nn.Module):
    """
    StableCrossEntropyLoss: A numerically stable alternative to CrossEntropyLoss
    that uses StableMax instead of Softmax to prevent Softmax Collapse.
    """
    def __init__(self, reduction='mean'):
        super(StableCrossEntropyLoss, self).__init__()
        self.stablemax = StableMax()
        self.reduction = reduction
    
    def forward(self, logits, targets):
        # Apply StableMax to get probabilities
        probs = self.stablemax(logits)
        
        # Compute cross-entropy loss
        if targets.dim() == logits.dim() - 1:
            # If targets are class indices
            loss = -torch.log(probs.gather(1, targets.unsqueeze(1)).squeeze(1) + 1e-10)
        else:
            # If targets are one-hot encoded
            loss = -torch.sum(targets * torch.log(probs + 1e-10), dim=-1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class OrthoGrad(Optimizer):
    """
    ⊥Grad (Ortho-Grad): An optimizer that prevents Naïve Loss Minimization (NLM)
    by only applying the component of the gradient that is orthogonal to the weights.
    
    As described in the paper "Grokking at the Edge of Numerical Stability", this
    prevents weights from scaling in their current direction, which can lead to
    numerical instability and delayed generalization.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        """
        Initialize OrthoGrad optimizer with Adam-like parameters.
        
        Args:
            params (iterable): Iterable of parameters to optimize
            lr (float, optional): Learning rate. Default: 1e-3
            betas (Tuple[float, float], optional): Coefficients for computing
                running averages of gradient and its square. Default: (0.9, 0.999)
            eps (float, optional): Term added to denominator for numerical stability. Default: 1e-8
            weight_decay (float, optional): Weight decay coefficient. Default: 0
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(OrthoGrad, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """
        Perform a single optimization step, applying only the component of the
        gradient that is orthogonal to the weights.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss. Default: None
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get gradient
                grad = p.grad.data
                
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # Compute orthogonal component of gradient
                if p.dim() > 0:  # Skip scalar parameters (e.g., biases)
                    # Flatten parameter and gradient
                    p_flat = p.data.view(-1)
                    grad_flat = grad.view(-1)
                    
                    # Compute the projection of grad onto p
                    # proj = (p·grad / p·p) * p
                    p_norm_sq = torch.dot(p_flat, p_flat)
                    
                    if p_norm_sq > 0:  # Avoid division by zero
                        proj_coeff = torch.dot(p_flat, grad_flat) / p_norm_sq
                        
                        # Compute orthogonal component: grad_orth = grad - proj
                        grad_proj = proj_coeff * p_flat
                        grad_orth = grad_flat - grad_proj
                        
                        # Reshape back to original shape
                        grad = grad_orth.view(grad.shape)
                
                # Apply Adam-like update with the orthogonal gradient
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Update biased first moment estimate (momentum)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute adaptive learning rate
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                
                # Apply update
                step_size = group['lr'] / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss


class OrthoAdamW(Optimizer):
    """
    ⊥AdamW (Ortho-AdamW): A variant of AdamW that only applies the component of the
    gradient that is orthogonal to the weights.
    
    This combines the benefits of AdamW with the orthogonal gradient approach
    of ⊥Grad to prevent Naïve Loss Minimization.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
                 amsgrad=False):
        """
        Initialize OrthoAdamW optimizer with AdamW-like parameters.
        
        Args:
            params (iterable): Iterable of parameters to optimize
            lr (float, optional): Learning rate. Default: 1e-3
            betas (Tuple[float, float], optional): Coefficients for computing
                running averages of gradient and its square. Default: (0.9, 0.999)
            eps (float, optional): Term added to denominator for numerical stability. Default: 1e-8
            weight_decay (float, optional): Weight decay coefficient. Default: 0.01
            amsgrad (bool, optional): Whether to use the AMSGrad variant. Default: False
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(OrthoAdamW, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(OrthoAdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
    
    def step(self, closure=None):
        """
        Perform a single optimization step, applying only the component of the
        gradient that is orthogonal to the weights.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss. Default: None
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get gradient
                grad = p.grad.data
                
                # Compute orthogonal component of gradient
                if p.dim() > 0:  # Skip scalar parameters (e.g., biases)
                    # Flatten parameter and gradient
                    p_flat = p.data.view(-1)
                    grad_flat = grad.view(-1)
                    
                    # Compute the projection of grad onto p
                    p_norm_sq = torch.dot(p_flat, p_flat)
                    
                    if p_norm_sq > 0:  # Avoid division by zero
                        proj_coeff = torch.dot(p_flat, grad_flat) / p_norm_sq
                        
                        # Compute orthogonal component: grad_orth = grad - proj
                        grad_proj = proj_coeff * p_flat
                        grad_orth = grad_flat - grad_proj
                        
                        # Reshape back to original shape
                        grad = grad_orth.view(grad.shape)
                
                if group['weight_decay'] != 0:
                    # Unlike traditional AdamW which modifies grad, we apply weight decay directly to parameters
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Apply AdamW-like update with the orthogonal gradient
                amsgrad = group['amsgrad']
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Update biased first moment estimate (momentum)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1
                
                # Apply update
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss


class OrthoSGD(Optimizer):
    """
    OrthoSGD: A variant of SGD that only applies the component of the
    gradient that is orthogonal to the weights.
    """
    def __init__(self, params, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        """
        Initialize OrthoSGD optimizer with SGD-like parameters.
        
        Args:
            params (iterable): Iterable of parameters to optimize
            lr (float): Learning rate. Default: 0.01
            momentum (float, optional): Momentum factor. Default: 0
            dampening (float, optional): Dampening for momentum. Default: 0
            weight_decay (float, optional): Weight decay coefficient. Default: 0
            nesterov (bool, optional): Enables Nesterov momentum. Default: False
        """
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                       weight_decay=weight_decay, nesterov=nesterov)
        
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        
        super(OrthoSGD, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """
        Perform a single optimization step, applying only the component of the
        gradient that is orthogonal to the weights.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss. Default: None
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get gradient
                grad = p.grad.data
                
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                # Compute orthogonal component of gradient
                if p.dim() > 0:  # Skip scalar parameters (e.g., biases)
                    # Flatten parameter and gradient
                    p_flat = p.data.view(-1)
                    grad_flat = grad.view(-1)
                    
                    # Compute the projection of grad onto p
                    p_norm_sq = torch.dot(p_flat, p_flat)
                    
                    if p_norm_sq > 0:  # Avoid division by zero
                        proj_coeff = torch.dot(p_flat, grad_flat) / p_norm_sq
                        
                        # Compute orthogonal component: grad_orth = grad - proj
                        grad_proj = proj_coeff * p_flat
                        grad_orth = grad_flat - grad_proj
                        
                        # Reshape back to original shape
                        grad = grad_orth.view(grad.shape)
                
                # Apply SGD-like update with the orthogonal gradient
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(grad)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                    
                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    else:
                        grad = buf
                
                # Apply update
                p.data.add_(grad, alpha=-group['lr'])
        
        return loss


def replace_optimizer(model, optimizer_type="OrthoAdamW", **kwargs):
    """
    Replace the optimizer in a model with an orthogonal gradient optimizer.
    
    Args:
        model: The model whose optimizer to replace
        optimizer_type (str): "OrthoGrad", "OrthoAdamW", or "OrthoSGD". Default: "OrthoAdamW"
        **kwargs: Optimizer parameters like lr, weight_decay, etc.
    
    Returns:
        The new optimizer instance
    """
    # Get parameters requiring gradients
    params = [p for p in model.parameters() if p.requires_grad]
    
    # Set default learning rate if not provided
    if 'lr' not in kwargs:
        kwargs['lr'] = 1e-3
    
    # Create the specified optimizer
    if optimizer_type.lower() == "orthograd":
        optimizer = OrthoGrad(params, **kwargs)
    elif optimizer_type.lower() == "orthoadamw":
        optimizer = OrthoAdamW(params, **kwargs)
    elif optimizer_type.lower() == "orthosgd":
        optimizer = OrthoSGD(params, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    # If the model has an attribute 'optimizer', replace it
    if hasattr(model, 'optimizer'):
        model.optimizer = optimizer
    
    return optimizer


def use_stablemax_loss(model, **kwargs):
    """
    Replace the standard CrossEntropyLoss with StableCrossEntropyLoss
    to prevent Softmax Collapse.
    
    Args:
        model: The model whose loss function to replace
        **kwargs: Loss function parameters
    
    Returns:
        The new StableCrossEntropyLoss instance
    """
    loss_fn = StableCrossEntropyLoss(**kwargs)
    
    # If the model has a 'criterion' or 'loss_fn' attribute, replace it
    if hasattr(model, 'criterion'):
        model.criterion = loss_fn
    elif hasattr(model, 'loss_fn'):
        model.loss_fn = loss_fn
    
    return loss_fn


def use_grokking_optimizations(model, loss=True, optimizer=True, optimizer_type="OrthoAdamW", 
                             optim_kwargs=None, loss_kwargs=None):
    """
    Apply the optimizations from the paper "Grokking at the Edge of Numerical Stability"
    to help the model grok faster.
    
    Args:
        model: The model to optimize
        loss (bool): Whether to replace the loss function. Default: True
        optimizer (bool): Whether to replace the optimizer. Default: True
        optimizer_type (str): "OrthoGrad", "OrthoAdamW" or "OrthoSGD". Default: "OrthoAdamW"
        optim_kwargs (dict): Optimizer parameters. Default: None
        loss_kwargs (dict): Loss function parameters. Default: None
    
    Returns:
        tuple: (new_loss_fn, new_optimizer) or just the one that was replaced
    """
    optim_kwargs = optim_kwargs or {}
    loss_kwargs = loss_kwargs or {}
    
    new_loss_fn = None
    new_optimizer = None
    
    if loss:
        new_loss_fn = use_stablemax_loss(model, **loss_kwargs)
    
    if optimizer:
        new_optimizer = replace_optimizer(model, optimizer_type, **optim_kwargs)
    
    if loss and optimizer:
        return new_loss_fn, new_optimizer
    elif loss:
        return new_loss_fn
    elif optimizer:
        return new_optimizer
    else:
        return None