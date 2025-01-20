import torch
import torch.nn as nn
from typing import List, Optional
from collections import defaultdict

class GradientVarianceEstimator:
    """
    Estimates the variance of gradients during training.
    
    This implementation uses an online algorithm to compute variance of gradients
    across multiple batches/iterations.
    """
    def __init__(self, model: nn.Module, window_size: int = 100):
        """
        Args:
            model: PyTorch model to monitor
            window_size: Number of recent gradients to use for variance estimation
        """
        self.model = model
        self.window_size = window_size
        self.grad_history = defaultdict(list)
        
    def update(self):
        """
        Captures current gradients and updates running statistics.
        Should be called after loss.backward() but before optimizer.step()
        """
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Store gradient
                grad_copy = param.grad.detach().clone()
                self.grad_history[name].append(grad_copy)
                
                # Keep only recent gradients
                if len(self.grad_history[name]) > self.window_size:
                    self.grad_history[name].pop(0)
    
    def compute_variance(self) -> dict:
        """
        Computes variance of gradients for each parameter.
        
        Returns:
            Dictionary mapping parameter names to their gradient variances
        """
        variances = {}
        
        for name, grads in self.grad_history.items():
            if len(grads) < 2:  # Need at least 2 samples for variance
                continue
                
            # Stack gradients along new dimension
            grads_tensor = torch.stack(grads)
            
            # Compute variance along batch dimension
            variance = torch.var(grads_tensor, dim=0)
            
            # Average variance across all dimensions
            avg_variance = variance.mean().item()
            variances[name] = avg_variance
            
        return variances
    
    def get_summary(self) -> dict:
        """
        Returns summary statistics about gradient variance.
        """
        variances = self.compute_variance()
        
        return {
            'max_variance': max(variances.values()) if variances else 0,
            'min_variance': min(variances.values()) if variances else 0,
            'mean_variance': sum(variances.values()) / len(variances) if variances else 0,
            'num_params_tracked': len(variances),
            'window_size': self.window_size
        }