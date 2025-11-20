import torch

def create_bidirectional_mask(batch_size, seq_length, device):
    """
    In a bidirectional mask, every token can attend to every other token,
    allowing full visibility in both directions.
    
    Args:
        batch_size (int): Batch size
        seq_length (int): Sequence length (both target and source length for self-attention)
        device: Device to create tensor on
        
    Returns:
        torch.FloatTensor: Bidirectional mask with shape (batch_size, 1, seq_length, seq_length)
    """
    # For bidirectional attention, we want all positions to be visible
    # This means the mask should be all zeros (allowing attention everywhere)
    
    # Create a tensor with shape (batch_size, 1, seq_length, seq_length) filled with zeros
    # In attention masks, 0.0 means "attend to this position"
    bidirectional_mask = torch.zeros((batch_size, 1, seq_length, seq_length), device=device)
    
    return bidirectional_mask