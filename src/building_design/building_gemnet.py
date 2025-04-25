"""
GemNet inspired building design NN Architecture
"""
import torch.nn as nn
from node_embeddings import BuildingNodeEmbedding
from edge_embeddings import BuildingEdgeEmbedding
from basis_functions import BuildingBasisFunctions
from interaction import BuildingInteractionBlock
from structural_integrity import StructuralIntegrityOutput
from energy_consumption import EnergyConsumptionOutput

class BuildingGemNet(nn.Module):
    """GemNet-inspired model for building analysis and prediction"""
    
    def __init__(
        self,
        node_input_dims={'room': 10, 'hvac': 8, 'electrical': 6},
        edge_input_dims={'structural': 9, 'pipes': 7, 'electrical': 6, 'airflow': 5},
        node_emb_size=256,
        edge_emb_size=256,
        num_interaction_blocks=3,
        num_load_basis=8,
        num_thermal_basis=16,
        num_flow_basis=12,
    ):
        super().__init__()
        
        # Initialize embeddings
        self.node_embedding = BuildingNodeEmbedding(
            input_dims=node_input_dims,
            emb_size=node_emb_size
        )
        
        self.edge_embedding = BuildingEdgeEmbedding(
            input_dims=edge_input_dims, 
            node_emb_size=node_emb_size, 
            edge_emb_size=edge_emb_size
        )
        
        # Initialize basis functions
        self.basis_functions = BuildingBasisFunctions(
            num_load_basis=num_load_basis,
            num_thermal_basis=num_thermal_basis,
            num_flow_basis=num_flow_basis
        )
        
        # Basis dimensions dictionary for the interaction blocks
        basis_dims = {
            'structural': num_load_basis,
            'thermal': num_thermal_basis,
            'flow': num_flow_basis
        }
        
        # Initialize interaction blocks
        self.interaction_blocks = nn.ModuleList([
            BuildingInteractionBlock(
                node_dim=node_emb_size,
                edge_dim=edge_emb_size,
                basis_dim_dict=basis_dims
            ) for _ in range(num_interaction_blocks)
        ])
        
        # Initialize output blocks for different predictions
        self.structural_integrity_output = StructuralIntegrityOutput(
            node_emb_size=node_emb_size,
            edge_emb_size=edge_emb_size
        )
        
        self.energy_consumption_output = EnergyConsumptionOutput(
            node_emb_size=node_emb_size,
            edge_emb_size=edge_emb_size
        )
    
    def forward(self, data):
        """
        Forward pass through the building GemNet model
        
        Args:
            data: A data object containing:
                - node_features: Dict of node features by type
                - node_types: Type of each node
                - edge_features: Dict of edge features by type
                - edge_types: Type of each edge
                - edge_index: Connectivity [2, num_edges]
                - edge_values: Physical values for basis functions
                - edge_capacities: Capacity values for flow edges
                - batch: Batch assignment for nodes
                
        Returns:
            Dict containing predicted structural integrity and energy consumption
        """
        # Get node embeddings
        x = self.node_embedding(
            data.node_features, 
            data.node_types
        )
        
        # Get edge embeddings
        edge_attr = self.edge_embedding(
            data.edge_features,
            data.edge_types,
            x,
            data.edge_index
        )
        
        # Calculate basis functions
        basis = self.basis_functions(
            data.edge_types,
            data.edge_values,
            data.edge_capacities
        )
        
        # Initial predictions (to be used in residual connections)
        structural_pred = self.structural_integrity_output(
            x, edge_attr, data.edge_index, data.batch
        )
        
        energy_pred = self.energy_consumption_output(
            x, edge_attr, data.edge_index, data.batch,
            data.node_types
        )
        
        # Interaction blocks (message passing)
        for interaction_block in self.interaction_blocks:
            x, edge_attr = interaction_block(
                x=x,
                edge_index=data.edge_index,
                edge_attr=edge_attr,
                basis_functions=basis,
                edge_types=data.edge_types
            )
            
            # Residual predictions from each layer
            structural_block_pred = self.structural_integrity_output(
                x, edge_attr, data.edge_index, data.batch
            )
            structural_pred += structural_block_pred
            
            energy_block_pred = self.energy_consumption_output(
                x, edge_attr, data.edge_index, data.batch,
                data.node_types
            )
            energy_pred += energy_block_pred
        
        return {
            'structural_integrity': structural_pred,
            'energy_consumption': energy_pred,
            'node_embeddings': x
        }