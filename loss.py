"""
NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for contrastive learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent loss for contrastive self-supervised learning.
    
    Given a batch of graph pairs (x_i, x_i'), the loss encourages
    positive pairs (from same graph) to be similar and negative pairs
    (from different graphs) to be dissimilar.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize NT-Xent loss.
        
        Args:
            temperature: Temperature parameter for scaling logits.
        """
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss.
        
        Args:
            z1: Embeddings of first augmented graphs [batch_size, hidden_dim].
            z2: Embeddings of second augmented graphs [batch_size, hidden_dim].
            
        Returns:
            Scalar loss value.
        """
        batch_size = z1.size(0)
        
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Concatenate all embeddings
        z = torch.cat([z1, z2], dim=0)  # [2*batch_size, hidden_dim]
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(z, z.T) / self.temperature  # [2*batch_size, 2*batch_size]
        
        # Create labels for positive pairs
        # For batch_size=3: [0,1,2] -> positive pairs are (0,3), (1,4), (2,5)
        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)  # [2*batch_size]
        
        # Create mask to exclude self-similarity
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss

