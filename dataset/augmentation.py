"""
Graph augmentation using connected subgraph removal for contrastive learning.
"""
import torch
import numpy as np
from torch_geometric.data import Data
from typing import Tuple, List, Optional, Set
import random
from collections import deque


class SubgraphRemovalAugmentation:
    """
    Augmentation strategy that removes a connected subgraph.
    Removed nodes are replaced with masked feature vectors, and removed bonds are deleted.
    Used to generate positive pairs for contrastive learning.
    """
    
    def __init__(self, removal_ratio: float = 0.25, seed: Optional[int] = None, mask_value: float = 0.0):
        """
        Initialize the augmentation.
        
        Args:
            removal_ratio: Ratio of nodes to remove (0.0 - 1.0).
            seed: Random seed for reproducibility.
            mask_value: Value to use for masking removed nodes (default: 0.0).
        """
        if not 0.0 <= removal_ratio < 1.0:
            raise ValueError(f"removal_ratio must be in [0.0, 1.0), got {removal_ratio}")
        
        self.removal_ratio = removal_ratio
        self.seed = seed
        self.mask_value = mask_value
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    def _find_connected_subgraph(
        self, 
        edge_index: torch.Tensor, 
        num_nodes: int, 
        target_size: int
    ) -> Set[int]:
        """
        Find a connected subgraph using BFS starting from a random node.
        
        Args:
            edge_index: Edge connectivity [2, num_edges].
            num_nodes: Total number of nodes.
            target_size: Target number of nodes in the subgraph.
            
        Returns:
            Set of node indices in the connected subgraph.
        """
        # Build adjacency list
        adj_list = [[] for _ in range(num_nodes)]
        edge_index_np = edge_index.cpu().numpy()
        for i in range(edge_index_np.shape[1]):
            src, dst = edge_index_np[0, i], edge_index_np[1, i]
            adj_list[src].append(dst)
            adj_list[dst].append(src)  # Undirected graph
        
        # Start BFS from a random node
        start_node = random.randint(0, num_nodes - 1)
        visited = set()
        queue = deque([start_node])
        visited.add(start_node)
        
        # BFS until we reach target size or run out of nodes
        while queue and len(visited) < target_size:
            current = queue.popleft()
            
            # Add unvisited neighbors
            for neighbor in adj_list[current]:
                if neighbor not in visited and len(visited) < target_size:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return visited
    
    def _create_mask_vector(self, feature_dim: int, device: torch.device = None) -> torch.Tensor:
        """
        Create a mask feature vector for removed nodes.
        
        Args:
            feature_dim: Dimension of node features.
            device: Device to create tensor on (if None, uses CPU).
            
        Returns:
            Mask vector with all values set to mask_value.
        """
        return torch.full((feature_dim,), self.mask_value, dtype=torch.float, device=device)
    
    def augment(self, graph: Data) -> Data:
        """
        Create an augmented version of the graph by removing a connected subgraph.
        Removed nodes are replaced with masked feature vectors, and removed bonds are deleted.
        
        Args:
            graph: Input graph (Data object).
            
        Returns:
            Augmented graph with connected subgraph masked.
        """
        num_nodes = graph.num_nodes
        num_to_remove = max(1, int(num_nodes * self.removal_ratio))
        
        # Ensure we don't remove all nodes
        num_to_remove = min(num_to_remove, num_nodes - 1)
        
        # Find connected subgraph to remove
        nodes_to_mask = self._find_connected_subgraph(
            graph.edge_index,
            num_nodes,
            num_to_remove
        )
        
        # Create masked node features
        x = graph.x.clone()
        mask_vector = self._create_mask_vector(x.shape[1], device=x.device)
        
        for node_idx in nodes_to_mask:
            x[node_idx] = mask_vector
        
        # Remove edges connected to masked nodes
        edge_index = graph.edge_index.clone()
        edge_attr = graph.edge_attr.clone() if graph.edge_attr is not None else None
        
        # Filter edges: keep only edges where both endpoints are NOT masked
        edge_index_np = edge_index.cpu().numpy()
        valid_edges = []
        
        for i in range(edge_index_np.shape[1]):
            src, dst = edge_index_np[0, i], edge_index_np[1, i]
            # Keep edge only if both nodes are not masked
            if src not in nodes_to_mask and dst not in nodes_to_mask:
                valid_edges.append(i)
        
        if len(valid_edges) > 0:
            valid_edges = torch.tensor(valid_edges, dtype=torch.long, device=edge_index.device)
            edge_index = edge_index[:, valid_edges]
            if edge_attr is not None:
                edge_attr = edge_attr[valid_edges]
        else:
            # If no valid edges, create empty edge tensors on the same device
            edge_index = torch.empty((2, 0), dtype=torch.long, device=graph.edge_index.device)
            if edge_attr is not None:
                edge_attr = torch.empty((0, edge_attr.shape[1]), dtype=torch.float, device=edge_attr.device)
        
        # Create augmented graph (keeping all nodes but with masked features)
        augmented_graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes  # Keep original number of nodes
        )
        
        return augmented_graph
    
    def create_pair(self, graph: Data) -> Tuple[Data, Data]:
        """
        Create a pair of augmented graphs from the same original graph.
        Each augmentation uses different random connected subgraph removal.
        
        Args:
            graph: Input graph.
            
        Returns:
            Tuple of two augmented graphs.
        """
        # Create two independent augmentations
        graph1 = self.augment(graph)
        graph2 = self.augment(graph)
        
        return graph1, graph2
    
    def __call__(self, graph: Data) -> Tuple[Data, Data]:
        """Alias for create_pair."""
        return self.create_pair(graph)

