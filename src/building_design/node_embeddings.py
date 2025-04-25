"""
This module deals with the nodes for our building design Foundation Model
Torch-scatter module needs to be installed properly since no version is available for Python 3.12
"""
import torch
import torch.nn as nn


class BuildingNodeEmbedding(nn.Module):
    """Embeds different types of building nodes (rooms, HVAC, electrical panels)"""
    def __init__(self, emb_size):
        """

        Args:
            emb_size (int): Embedding vector size
        """
        super().__init__()
        self.emb_size = emb_size
        
        # Room embedding
        self.room_embedding = nn.Sequential(
            nn.Linear(10, emb_size // 2),  # [dimensions, type, occupancy, etc.]
            nn.ReLU(),
            nn.Linear(emb_size // 2, emb_size)
        )
        
        # HVAC embedding
        self.hvac_embedding = nn.Sequential(
            nn.Linear(8, emb_size // 2),  # [capacity, efficiency, age, etc.]
            nn.ReLU(),
            nn.Linear(emb_size // 2, emb_size)
        )
        
        # Electrical panel embedding
        self.electrical_embedding = nn.Sequential(
            nn.Linear(6, emb_size // 2),  # [capacity, distribution features, etc.]
            nn.ReLU(),
            nn.Linear(emb_size // 2, emb_size)
        )
    
    def forward(self, node_features, node_types):
        """
        Args:
            node_features: Dictionary of features for each node type
            node_types: Type of each node [0=room, 1=hvac, 2=electrical]
            
        Returns:
            Node embeddings [num_nodes, emb_size]
        """
        embeddings = torch.zeros(len(node_types), self.emb_size, device=node_features['room'].device)
        
        # Embed each node type
        room_mask = node_types == 0
        if room_mask.any():
            embeddings[room_mask] = self.room_embedding(node_features['room'])
            
        hvac_mask = node_types == 1
        if hvac_mask.any():
            embeddings[hvac_mask] = self.hvac_embedding(node_features['hvac'])
            
        electrical_mask = node_types == 2
        if electrical_mask.any():
            embeddings[electrical_mask] = self.electrical_embedding(node_features['electrical'])
            
        return embeddings