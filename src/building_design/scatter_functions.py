"""
Custom functions to replace the torch-scatter functions
"""
import torch

# Custom scatter replacement function
def scatter_sum(src, index, dim_size=None):
    """
    Replacement for torch_scatter.scatter with 'sum' reduction
    
    Args:
        src: Source tensor to scatter
        index: Index tensor that defines where to scatter
        dim_size: Size of the output tensor
    
    Returns:
        Scattered tensor
    """
    if dim_size is None:
        dim_size = index.max().item() + 1
    
    output = torch.zeros(dim_size, *src.shape[1:], device=src.device, dtype=src.dtype)
    return output.index_add_(0, index, src)

def scatter_mean(src, index, dim_size=None):
    """
    Replacement for torch_scatter.scatter with 'mean' reduction
    
    Args:
        src: Source tensor to scatter
        index: Index tensor that defines where to scatter
        dim_size: Size of the output tensor
    
    Returns:
        Scattered tensor
    """
    if dim_size is None:
        dim_size = index.max().item() + 1
    
    output = torch.zeros(dim_size, *src.shape[1:], device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, device=src.device, dtype=torch.float)
    
    output = output.index_add_(0, index, src)
    count = count.index_add_(0, index, torch.ones_like(index, dtype=torch.float))
    
    # Avoid division by zero
    count = torch.clamp(count, min=1.0).unsqueeze(-1)
    return output / count