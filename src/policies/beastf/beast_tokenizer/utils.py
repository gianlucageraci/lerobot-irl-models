import torch
import einops
from typing import Union

def continuous_to_discrete(tensor, min_val=None, max_val=None, num_bins=256):
    """
    Convert a continuous PyTorch tensor to discrete tokens in the range [0, 255].

    Args:
        tensor (torch.Tensor): Input tensor with continuous values.
        min_val (float, optional): Minimum value for normalization. If None, use tensor.min().
        max_val (float, optional): Maximum value for normalization. If None, use tensor.max().

    Returns:
        torch.Tensor: Discretized tensor with values in the range [0, 255].
    """

    if min_val is None:
        min_val = tensor.min()
    if max_val is None:
        max_val = tensor.max()

    # Normalize the tensor to [0, 1]
    assert torch.all(tensor >= min_val - 1e-3), "Input tensor has values below min_val"
    assert torch.all(tensor <= max_val + 1e-3), "Input tensor has values above max_val"
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    normalized_tensor = torch.clamp(normalized_tensor, 0, 1)

    # Ensure no out-of-bound values
    # Scale to [0, 255] and quantize to integers
    discrete_tensor = torch.round(normalized_tensor * (num_bins-1)).to(torch.long)
    return discrete_tensor


def discrete_to_continuous(discrete_tensor, min_val=0, max_val=1, num_bins=256):
    """
    Convert a discrete PyTorch tensor with values in the range [0, 255]
    back to continuous values in the range [min_val, max_val].

    Args:
        discrete_tensor (torch.Tensor): Input tensor with discrete values (0 to 255).
        min_val (float): Minimum value of the original continuous range.
        max_val (float): Maximum value of the original continuous range.

    Returns:
        torch.Tensor: Continuous tensor with values in the range [min_val, max_val].
    """
    # Map discrete tokens to [0, 1]
    # Normalize the tensor to [0, 1]
    normalized_tensor = discrete_tensor.float() / (num_bins-1)

    # Map normalized values to [min_val, max_val]
    continuous_tensor = normalized_tensor * (max_val - min_val) + min_val

    # Ensure no out-of-bound values
    continuous_tensor = torch.clamp(continuous_tensor, min_val, max_val)
    return continuous_tensor


def normalize_tensor(tensor, w_min, w_max, norm_min=-1.0, norm_max=1.0):
    """
    Normalize a tensor from its original range [w_min, w_max] to a new range [norm_min, norm_max].
    
    Args:
        tensor (torch.Tensor): Input tensor to be normalized
        w_min (float): Minimum value bound of the original tensor
        w_max (float): Maximum value bound of the original tensor
        norm_min (float, optional): Minimum value of the normalized range. Defaults to 0.0.
        norm_max (float, optional): Maximum value of the normalized range. Defaults to 1.0.
    
    Returns:
        torch.Tensor: Normalized tensor with values in range [norm_min, norm_max]
    """
    
    # Clip the input tensor to be within [w_min, w_max]
    clipped_tensor = torch.clamp(tensor, w_min, w_max)
    
    # Normalize to [0, 1] range first
    normalized = (clipped_tensor - w_min) / (w_max - w_min)
    
    # Scale to the desired [norm_min, norm_max] range
    normalized = normalized * (norm_max - norm_min) + norm_min
    
    return normalized

def denormalize_tensor(normalized_tensor, w_min, w_max, norm_min=-1.0, norm_max=1.0):
    """
    Denormalize a tensor from the normalized range [norm_min, norm_max] back to the original range [w_min, w_max].
    
    Args:
        normalized_tensor (torch.Tensor): Normalized input tensor
        w_min (float): Minimum value bound of the original range
        w_max (float): Maximum value bound of the original range
        norm_min (float, optional): Minimum value of the normalized range. Defaults to 0.0.
        norm_max (float, optional): Maximum value of the normalized range. Defaults to 1.0.
    
    Returns:
        torch.Tensor: Denormalized tensor with values in range [w_min, w_max]
    """
    
    # Clip the normalized tensor to be within [norm_min, norm_max]
    clipped_tensor = torch.clamp(normalized_tensor, norm_min, norm_max)
    
    # Scale from [norm_min, norm_max] to [0, 1] first
    denormalized = (clipped_tensor - norm_min) / (norm_max - norm_min)
    
    # Scale to the original [w_min, w_max] range
    denormalized = denormalized * (w_max - w_min) + w_min
    
    return denormalized


def tensor_linspace(start: Union[float, int, torch.Tensor],
                    end: Union[float, int, torch.Tensor],
                    steps: int) -> torch.Tensor:
    """
    Vectorized version of torch.linspace.
    Modified from:
    https://github.com/zhaobozb/layout2im/blob/master/models/bilinear.py#L246

    Args:
        start: start value, scalar or tensor
        end: end value, scalar or tensor
        steps: num of steps

    Returns:
        linspace tensor
    """
    # Shape of start:
    # [*add_dim, dim_data] or a scalar
    #
    # Shape of end:
    # [*add_dim, dim_data] or a scalar
    #
    # Shape of out:
    # [*add_dim, steps, dim_data]

    # - out: Tensor of shape start.size() + (steps,), such that
    #   out.select(-1, 0) == start, out.select(-1, -1) == end,
    #   and the other elements of out linearly interpolate between
    #   start and end.

    if isinstance(start, torch.Tensor) and not isinstance(end, torch.Tensor):
        end += torch.zeros_like(start)
    elif not isinstance(start, torch.Tensor) and isinstance(end, torch.Tensor):
        start += torch.zeros_like(end)
    elif isinstance(start, torch.Tensor) and isinstance(end, torch.Tensor):
        assert start.size() == end.size()
    else:
        return torch.linspace(start, end, steps)

    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    out = torch.einsum('...ji->...ij', out)
    return out