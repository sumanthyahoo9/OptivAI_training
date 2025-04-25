"""
Predicts the energy consumption from the node and edge features
"""
import torch
import torch.nn as nn
from scatter_functions import scatter_mean, scatter_sum

class EnergyConsumptionOutput(nn.Module):
    """Predicts energy consumption from node and edge features"""
    
    def __init__(self, node_emb_size, edge_emb_size):
        super().__init__()
        
        # Different weights for different node types
        self.room_mlp = nn.Sequential(
            nn.Linear(node_emb_size, node_emb_size // 2),
            nn.ReLU(),
            nn.Linear(node_emb_size // 2, 1)
        )
        
        self.hvac_mlp = nn.Sequential(
            nn.Linear(node_emb_size, node_emb_size // 2),
            nn.ReLU(),
            nn.Linear(node_emb_size // 2, 1)
        )
        
        self.electrical_mlp = nn.Sequential(
            nn.Linear(node_emb_size, node_emb_size // 2),
            nn.ReLU(),
            nn.Linear(node_emb_size // 2, 1)
        )
        
        # Process edge features for energy consumption
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_emb_size, edge_emb_size // 2),
            nn.ReLU(),
            nn.Linear(edge_emb_size // 2, 1)
        )
        
        # Final prediction layer
        self.output = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x, edge_attr, edge_index, batch, node_types):
        """
        Predict energy consumption
        
        Args:
            x: Node embeddings [num_nodes, node_emb_size]
            edge_attr: Edge embeddings [num_edges, edge_emb_size]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment for nodes
            node_types: Type of each node
            
        Returns:
            Energy consumption prediction per building [num_buildings, 1]
        """
        # Get node-level predictions by type
        room_mask = node_types == 0
        hvac_mask = node_types == 1
        electrical_mask = node_types == 2
        
        # Initialize predictions
        node_pred = torch.zeros(x.shape[0], 1, device=x.device)
        
        # Predict for each node type
        if room_mask.any():
            node_pred[room_mask] = self.room_mlp(x[room_mask])
        
        if hvac_mask.any():
            node_pred[hvac_mask] = self.hvac_mlp(x[hvac_mask])
            
        if electrical_mask.any():
            node_pred[electrical_mask] = self.electrical_mlp(x[electrical_mask])

        room_pred = scatter_sum(
            node_pred[room_mask],
            batch[room_mask], 
            dim_size=batch.max().item() + 1
        )
        
        hvac_pred = scatter_mean(
            node_pred[hvac_mask],
            batch[hvac_mask],
            dim_size=batch.max().item() + 1
        )
        electrical_pred = scatter_mean(
            node_pred[electrical_mask],
            batch[electrical_mask],
            dim=0, 
            reduce='mean'
        )
        
        # Get edge-level predictions
        edge_pred = self.edge_mlp(edge_attr)
        
        # Aggregate edge predictions per building
        edge_batch = batch[edge_index[0]]
        building_edge_pred = scatter_sum(edge_pred, edge_batch, dim=0, reduce='sum')
        
        # Combine predictions
        combined = torch.cat([
            room_pred, 
            hvac_pred, 
            electrical_pred, 
            building_edge_pred
        ], dim=1)
        
        # Final prediction
        return self.output(combined)