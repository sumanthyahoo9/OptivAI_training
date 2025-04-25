"""
This module gives the edges for our building design Foundation Model
"""
import torch
import torch.nn as nn

class BuildingEdgeEmbedding(nn.Module):
    """Embeds different types of building edges (structural, pipes, wiring, airflow)"""
    
    def __init__(self, input_dims, node_emb_size, edge_emb_size):
        super().__init__()
        self.edge_emb_size = edge_emb_size
        
        # Create embeddings for each edge type
        self.embeddings = nn.ModuleDict({
            edge_type: nn.Sequential(
                nn.Linear(dim, edge_emb_size // 2),
                nn.ReLU(),
                nn.Linear(edge_emb_size // 2, edge_emb_size)
            ) for edge_type, dim in input_dims.items()
        })
        
        # Edge update based on connected nodes
        self.edge_update = nn.Sequential(
            nn.Linear(edge_emb_size + 2 * node_emb_size, edge_emb_size),
            nn.ReLU(),
            nn.Linear(edge_emb_size, edge_emb_size)
        )
        
    def forward(self, edge_features, edge_types, node_embeddings, edge_index):
        """
        Args:
            edge_features: Dictionary of features for each edge type
            edge_types: Type of each edge [0=structural, 1=pipes, 2=electrical, 3=airflow]
            node_embeddings: Embeddings of nodes [num_nodes, emb_size]
            edge_index: Edge connectivity [2, num_edges]
            
        Returns:
            Edge embeddings [num_edges, edge_emb_size]
        """
        embeddings = torch.zeros(
            len(edge_types), 
            self.edge_emb_size, 
            device=edge_types.device
        )
        
        # Map edge type integers to string keys
        type_mapping = {0: 'structural', 1: 'pipes', 2: 'electrical', 3: 'airflow'}
        
        # Embed each edge type
        for type_idx, type_name in type_mapping.items():
            mask = edge_types == type_idx
            if mask.any():
                embeddings[mask] = self.embeddings[type_name](edge_features[type_name])
        
        # Update edge embeddings based on connected nodes
        src, dst = edge_index
        node_src = node_embeddings[src]
        node_dst = node_embeddings[dst]
        
        edge_node_features = torch.cat([embeddings, node_src, node_dst], dim=1)
        embeddings = self.edge_update(edge_node_features)
        
        return embeddings
