"""
Defines the way various components in the building interact with one another
Uses the message passing module from torch geometric
"""
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class BuildingInteractionBlock(MessagePassing):
    """Message passing block for building components"""
    
    def __init__(self, node_dim, edge_dim, basis_dim_dict):
        super().__init__(aggr="add")
        
        # Project basis functions to edge dimension
        self.basis_projections = nn.ModuleDict({
            k: nn.Linear(v, edge_dim, bias=False) 
            for k, v in basis_dim_dict.items()
        })
        
        # Message networks
        self.message_network = nn.Sequential(
            nn.Linear(edge_dim + 2 * node_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim)
        )
        
        # Node update network
        self.node_update = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        
        # Edge update network
        self.edge_update = nn.Sequential(
            nn.Linear(edge_dim + 2 * node_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim)
        )
    
    def forward(self, x, edge_index, edge_attr, basis_functions, edge_types):
        """
        Forward pass through interaction block
        
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            basis_functions: Dict of basis function values for each edge type
            edge_types: Type of each edge
            
        Returns:
            Updated node and edge features
        """
        # Propagate messages
        x_updated = self.propagate(
            edge_index, 
            x=x, 
            edge_attr=edge_attr,
            basis_functions=basis_functions,
            edge_types=edge_types
        )
        
        # Update edges after node update
        src, dst = edge_index
        src_x = x_updated[src]
        dst_x = x_updated[dst]
        
        edge_update_inputs = torch.cat([edge_attr, src_x, dst_x], dim=1)
        edge_attr_updated = self.edge_update(edge_update_inputs)
        
        return x_updated, edge_attr_updated
    
    def message(self, x_i, x_j, edge_attr, basis_functions, edge_types):
        """Create messages from source nodes to target nodes"""
        # Combine node features and edge attributes
        message_inputs = torch.cat([x_i, x_j, edge_attr], dim=1)
        
        # Create base message
        messages = self.message_network(message_inputs)
        
        # Modulate messages by appropriate basis functions
        # Store original shape for reshaping
        orig_shape = messages.shape
        result = torch.zeros_like(messages)
        
        # Apply structural basis
        struct_mask = edge_types == 0
        if struct_mask.any() and 'structural' in basis_functions:
            basis_proj = self.basis_projections['structural'](basis_functions['structural'])
            result[struct_mask] = messages[struct_mask] * basis_proj
            
        # Apply thermal basis
        thermal_mask = edge_types == 3
        if thermal_mask.any() and 'thermal' in basis_functions:
            basis_proj = self.basis_projections['thermal'](basis_functions['thermal'])
            result[thermal_mask] = messages[thermal_mask] * basis_proj
            
        # Apply flow basis
        flow_mask = (edge_types == 1) | (edge_types == 2)
        if flow_mask.any() and 'flow' in basis_functions:
            basis_proj = self.basis_projections['flow'](basis_functions['flow'])
            result[flow_mask] = messages[flow_mask] * basis_proj
            
        return result
        
    def update(self, aggr_out, x):
        """Update node embeddings based on aggregated messages"""
        # Combine current node features with aggregated messages
        node_update_input = torch.cat([x, aggr_out], dim=1)
        
        # Update node features
        return self.node_update(node_update_input)