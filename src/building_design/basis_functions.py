"""
Defines the various basis functions for our building design
Following GemNet, a basis function essentially defines the way 
to encode various features in the building
"""
import torch
import torch.nn as nn

class BuildingBasisFunctions(nn.Module):
    """Building-specific basis functions for different physical processes"""
    
    def __init__(self, num_load_basis=8, num_thermal_basis=16, num_flow_basis=12):
        super().__init__()
        self.num_load_basis = num_load_basis
        self.num_thermal_basis = num_thermal_basis
        self.num_flow_basis = num_flow_basis
        
        # Parameters for load distribution basis (structural)
        self.load_offsets = nn.Parameter(torch.linspace(0, 1, num_load_basis))
        self.load_widths = nn.Parameter(torch.ones(num_load_basis) * 0.1)
        
        # Parameters for thermal diffusion basis
        self.thermal_offsets = nn.Parameter(torch.linspace(0, 1, num_thermal_basis))
        self.thermal_widths = nn.Parameter(torch.ones(num_thermal_basis) * 0.1)
        
        # Parameters for flow network basis
        self.flow_offsets = nn.Parameter(torch.linspace(0, 1, num_flow_basis))
        self.flow_widths = nn.Parameter(torch.ones(num_flow_basis) * 0.1)
        
    def load_basis(self, values):
        """Directional basis for structural loads"""
        # Gaussian RBF
        return torch.exp(-(values.unsqueeze(-1) - self.load_offsets)**2 / self.load_widths)
    
    def thermal_basis(self, values):
        """Radial basis for thermal diffusion"""
        # Gaussian RBF with different characteristics
        return torch.exp(-(values.unsqueeze(-1) - self.thermal_offsets)**2 / self.thermal_widths)
    
    def flow_basis(self, values, capacities):
        """Network flow basis for pipes and electrical"""
        # Capacity-modulated basis
        base = torch.exp(-(values.unsqueeze(-1) - self.flow_offsets)**2 / self.flow_widths)
        return base * capacities.unsqueeze(-1)  # Modulate by capacity
    
    def forward(self, edge_types, edge_values, edge_capacities=None):
        """
        Calculate basis functions for each edge type
        
        Args:
            edge_types: Type of each edge [0=structural, 1=pipes, 2=electrical, 3=airflow]
            edge_values: Physical values for basis calculation (loads, temps, flows)
            edge_capacities: Optional capacity values for flow-based edges
            
        Returns:
            Dictionary of basis function values for each edge type
        """
        basis_outputs = {}
        
        # Structural basis
        struct_mask = edge_types == 0
        if struct_mask.any():
            basis_outputs['structural'] = self.load_basis(edge_values[struct_mask])
        
        # Thermal basis for airflow
        thermal_mask = edge_types == 3  
        if thermal_mask.any():
            basis_outputs['thermal'] = self.thermal_basis(edge_values[thermal_mask])
        
        # Flow basis for pipes and electrical
        flow_mask = (edge_types == 1) | (edge_types == 2)
        if flow_mask.any() and edge_capacities is not None:
            basis_outputs['flow'] = self.flow_basis(
                edge_values[flow_mask], 
                edge_capacities[flow_mask]
            )
        
        return basis_outputs