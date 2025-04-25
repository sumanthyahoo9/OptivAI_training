"""
Predicts the structural integrity from node and edge features
"""
import torch
import torch.nn as nn
from scatter_functions import scatter_mean

class StructuralIntegrityOutput(nn.Module):
    """Predicts structural integrity from node and edge features"""
    
    def __init__(self, node_emb_size, edge_emb_size):
        super().__init__()
        
        # Process node features for structural integrity
        self.node_mlp = nn.Sequential(
            nn.Linear(node_emb_size, node_emb_size // 2),
            nn.ReLU(),
            nn.Linear(node_emb_size // 2, 1)
        )
        
        # Process edge features for structural integrity
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_emb_size, edge_emb_size // 2),
            nn.ReLU(),
            nn.Linear(edge_emb_size // 2, 1)
        )
        
        # Final prediction layer
        self.output = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    
    def forward(self, x, edge_attr, edge_index, batch):
        """
        Predict structural integrity
        
        Args:
            x: Node embeddings [num_nodes, node_emb_size]
            edge_attr: Edge embeddings [num_edges, edge_emb_size]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment for nodes
            
        Returns:
            Structural integrity prediction per building [num_buildings, 1]
        """
        # Get node-level predictions
        node_pred = self.node_mlp(x)
    
        # Aggregate node predictions per building
        building_node_pred = scatter_mean(node_pred, batch, dim_size=batch.max().item() + 1)
    
        # Get edge-level predictions
        edge_pred = self.edge_mlp(edge_attr)
    
        # Aggregate edge predictions per building
        edge_batch = batch[edge_index[0]]  # Use source node's batch
        building_edge_pred = scatter_mean(edge_pred, edge_batch, dim_size=batch.max().item() + 1)
        
        # Combine node and edge predictions
        combined = torch.cat([building_node_pred, building_edge_pred], dim=1)
        
        # Final prediction
        return self.output(combined)