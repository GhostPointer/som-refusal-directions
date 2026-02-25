import os
import torch 

def get_ablated_matrix(matrix: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    # Save the original dtype because FP8 (float8_e4m3fn) math is highly restricted and imprecise
    original_dtype = matrix.dtype
    
    # Cast matrix to higher precision for math
    if matrix.dtype in [torch.float8_e4m3fn, torch.float8_e5m2, torch.float16, torch.bfloat16]:
        math_matrix = matrix.to(torch.float32)
    else:
        math_matrix = matrix

    vec = vec / torch.norm(vec)
    vec = vec.to(dtype=math_matrix.dtype, device=math_matrix.device)
    
    # Perform projection in fp32
    proj = torch.einsum('...d,d->...', math_matrix, vec)  # shape: [...]
    ablated_matrix = math_matrix - proj.unsqueeze(-1) * vec  # shape: [..., d_model]
    
    # Cast back to the original format (e.g. float8_e4m3fn)
    return ablated_matrix.to(original_dtype)

def ablate_weights(model, direction: torch.Tensor):
    model.ablate_weights(direction)

def clear_ablation(model):
    """Reset model to pre-ablation state (clear hooks or restore weights)."""
    model.clear_ablation()
