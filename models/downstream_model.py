"""
Downstream molecular property prediction model.
Uses a pretrained GIN-E encoder, a shallow MLP, and prediction head(s).
Default: single prediction head for binding energy prediction.
"""
import os
import torch
import torch.nn as nn
from typing import Optional
from models.gin_e import GINEEncoder


class DownstreamModel(nn.Module):
    """
    Downstream model for molecular property prediction.
    Components:
    1. Pretrained GIN-E encoder
    2. Shallow MLP to refine embeddings
    3. Prediction MLP head(s) (default: 1 for binding energy prediction)
    """
    
    def __init__(
        self,
        # GIN-E encoder
        gin_e_encoder: GINEEncoder,
        gin_e_checkpoint_path: Optional[str] = None,
        freeze_gin_e: bool = True,
        
        # MLP parameters
        mlp_hidden_dim: int = 512,
        mlp_dropout: float = 0.1,
        
        # Prediction head parameters
        num_tasks: int = 3,
        task_hidden_dim: int = 256,
        task_dropout: float = 0.1
    ):
        """
        Initialize downstream model.
        
        Args:
            gin_e_encoder: GIN-E encoder model.
            gin_e_checkpoint_path: Path to pretrained GIN-E checkpoint.
            freeze_gin_e: Whether to freeze GIN-E encoder weights.
            
            mlp_hidden_dim: Hidden dimension for the combining MLP.
            mlp_dropout: Dropout rate for the combining MLP.
            
            num_tasks: Number of prediction tasks (default: 3).
            task_hidden_dim: Hidden dimension for each prediction head.
            task_dropout: Dropout rate for prediction heads.
        """
        super(DownstreamModel, self).__init__()
        
        # GIN-E encoder
        self.gin_e_encoder = gin_e_encoder
        if gin_e_checkpoint_path is not None:
            self._load_checkpoint(self.gin_e_encoder, gin_e_checkpoint_path)
        if freeze_gin_e:
            for param in self.gin_e_encoder.parameters():
                param.requires_grad = False
            self.gin_e_encoder.eval()
        
        # Get embedding dimensions
        gin_e_dim = gin_e_encoder.hidden_dim
        combined_dim = gin_e_dim
        
        # Shallow MLP to combine embeddings
        self.combining_mlp = nn.Sequential(
            nn.Linear(combined_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout)
        )
        
        # Three separate prediction MLP heads
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(mlp_hidden_dim, task_hidden_dim),
                nn.ReLU(),
                nn.Dropout(task_dropout),
                nn.Linear(task_hidden_dim, task_hidden_dim),
                nn.ReLU(),
                nn.Dropout(task_dropout),
                nn.Linear(task_hidden_dim, 1)  # Single output for regression
            ) for _ in range(num_tasks)
        ])
        
        self.num_tasks = num_tasks
    
    def _load_checkpoint(self, model: nn.Module, checkpoint_path: str):
        """Load pretrained checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        print(f"Loading GIN-E checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', 'unknown')
            loss = checkpoint.get('loss', 'unknown')
            print(f"  Checkpoint info: epoch={epoch}, loss={loss}")
        else:
            state_dict = checkpoint
        
        # Load state dict with error handling
        try:
            model.load_state_dict(state_dict, strict=True)
            print(f"  Successfully loaded GIN-E encoder weights")
        except RuntimeError as e:
            # If strict loading fails, try partial loading
            print(f"  Warning: Strict loading failed: {e}")
            print(f"  Attempting partial loading...")
            model.load_state_dict(state_dict, strict=False)
            print(f"  Partially loaded GIN-E encoder weights (some layers may be missing)")
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, node_feature_dim].
            edge_index: Edge connectivity [2, num_edges].
            edge_attr: Edge features [num_edges, edge_feature_dim].
            batch: Batch assignment [num_nodes] (for batched graphs).
            
        Returns:
            Predictions for each task [batch_size, num_tasks] or [num_tasks] for single graph.
        """
        # Get embeddings from GIN-E encoder
        gin_e_frozen = all(not p.requires_grad for p in self.gin_e_encoder.parameters())
        
        with torch.set_grad_enabled(not gin_e_frozen):
            gin_e_emb = self.gin_e_encoder(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch
            )  # [batch_size, gin_e_dim] or [gin_e_dim]
        
        if batch is None:
            gin_e_emb = gin_e_emb.unsqueeze(0)  # [1, gin_e_dim]
        
        # Pass through combining MLP
        mlp_output = self.combining_mlp(gin_e_emb)  # [batch_size, mlp_hidden_dim] or [1, mlp_hidden_dim]
        
        # Get predictions from each head
        predictions = []
        for head in self.prediction_heads:
            pred = head(mlp_output)  # [batch_size, 1] or [1, 1]
            predictions.append(pred)
        
        # Stack predictions: [batch_size, num_tasks] or [1, num_tasks]
        predictions = torch.cat(predictions, dim=1)
        
        return predictions.squeeze(0) if batch is None else predictions

