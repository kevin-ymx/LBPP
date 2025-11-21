"""
GIN-E (Graph Isomorphism Network with Edge features) encoder for molecular graphs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool
from torch_geometric.nn import MessagePassing
from typing import Optional


class NodeFeatureEncoder(nn.Module):
    """
    Encodes raw node features into node embeddings.
    """
    def __init__(self, input_dim: int, embedding_dim: int):
        """
        Args:
            input_dim: Dimension of input node features (7).
            embedding_dim: Dimension of node embeddings.
        """
        super(NodeFeatureEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class EdgeFeatureEncoder(nn.Module):
    """
    Encodes raw edge features into edge embeddings.
    """
    def __init__(self, input_dim: int, embedding_dim: int):
        """
        Args:
            input_dim: Dimension of input edge features (3).
            embedding_dim: Dimension of edge embeddings.
        """
        super(EdgeFeatureEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.encoder(edge_attr)


class GINEEncoder(nn.Module):
    """
    Multi-layer GIN-E encoder for converting molecular graphs to embeddings.
    """
    def __init__(
        self,
        node_feature_dim: int = 7,
        edge_feature_dim: int = 3,
        node_embedding_dim: int = 128,
        edge_embedding_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 5,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        pooling: str = "mean"  # "mean" or "add"
    ):
        """
        Initialize GIN-E encoder.
        
        Args:
            node_feature_dim: Dimension of input node features.
            edge_feature_dim: Dimension of input edge features.
            node_embedding_dim: Dimension of node embeddings.
            edge_embedding_dim: Dimension of edge embeddings.
            hidden_dim: Hidden dimension for GIN layers.
            num_layers: Number of GIN layers.
            dropout: Dropout rate.
            use_batch_norm: Whether to use batch normalization.
            pooling: Graph-level pooling method ("mean" or "add").
        """
        super(GINEEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        
        # Node and edge feature encoders
        self.node_encoder = NodeFeatureEncoder(node_feature_dim, node_embedding_dim)
        self.edge_encoder = EdgeFeatureEncoder(edge_feature_dim, edge_embedding_dim)
        
        # GIN-E layers
        self.gin_layers = nn.ModuleList()
        
        # First layer: node_embedding_dim -> hidden_dim
        self.gin_layers.append(
            GINEConv(
                nn.Sequential(
                    nn.Linear(node_embedding_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
                    nn.ReLU()
                ),
                edge_dim=edge_embedding_dim,
                train_eps=True
            )
        )
        
        # Intermediate layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.gin_layers.append(
                GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
                        nn.ReLU()
                    ),
                    edge_dim=edge_embedding_dim,
                    train_eps=True
                )
            )
        
        # Last layer: hidden_dim -> hidden_dim (if more than 1 layer)
        if num_layers > 1:
            self.gin_layers.append(
                GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
                        nn.ReLU()
                    ),
                    edge_dim=edge_embedding_dim,
                    train_eps=True
                )
            )
        
        # Final projection layer
        self.final_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
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
            Graph-level embeddings [batch_size, hidden_dim] or [hidden_dim] for single graph.
        """
        # Encode node and edge features
        x = self.node_encoder(x)  # [num_nodes, node_embedding_dim]
        edge_attr = self.edge_encoder(edge_attr)  # [num_edges, edge_embedding_dim]
        
        # Apply GIN-E layers
        for i, gin_layer in enumerate(self.gin_layers):
            x = gin_layer(x, edge_index, edge_attr)
            if i < len(self.gin_layers) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final projection
        x = self.final_projection(x)
        
        # Graph-level pooling
        if batch is not None:
            # Batched graphs
            if self.pooling == "mean":
                graph_emb = global_mean_pool(x, batch)
            else:
                graph_emb = global_add_pool(x, batch)
        else:
            # Single graph
            if self.pooling == "mean":
                graph_emb = x.mean(dim=0, keepdim=True)
            else:
                graph_emb = x.sum(dim=0, keepdim=True)
        
        return graph_emb

