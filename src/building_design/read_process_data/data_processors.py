"""
Modules to process the input data from various sources
"""
import numpy as np
import torch
from torch_geometric.data import Data, Batch
import networkx as nx

class BuildingGraphProcessor:
    """Process building data into graph structure for neural network training"""
    
    def __init__(self, config=None):
        """Initialize with optional configuration"""
        self.config = config or {}
        self.node_feature_dims = {
            'room': 10,
            'hvac': 8,
            'electrical': 6
        }
        self.edge_feature_dims = {
            'structural': 9,
            'pipes': 7,
            'electrical': 6,
            'airflow': 5
        }
    
    def process_blueprint(self, blueprint_data):
        """
        Process blueprint data into initial graph structure
        
        Args:
            blueprint_data: Dict containing extracted blueprint information
            
        Returns:
            nx.Graph representing building structure
        """
        # Create networkx graph for easy manipulation
        G = nx.Graph()
        
        # Process based on blueprint data type (PDF, CAD, etc.)
        if 'text_blocks' in blueprint_data:
            # Process text-based blueprint (PDF, raster image)
            self._process_text_blocks(G, blueprint_data['text_blocks'])
            
            if 'vector_graphics' in blueprint_data:
                self._process_vector_graphics(G, blueprint_data['vector_graphics'])
            elif 'lines' in blueprint_data:
                self._process_lines(G, blueprint_data['lines'])
        
        elif 'entities' in blueprint_data:
            # Process CAD blueprint
            self._process_cad_entities(G, blueprint_data['entities'], blueprint_data.get('layers', []))
        
        elif 'shapes' in blueprint_data:
            # Process Visio blueprint
            self._process_visio_shapes(G, blueprint_data['shapes'])
            self._process_visio_connectors(G, blueprint_data['connectors'])
        
        return G
    
    def _process_text_blocks(self, G, text_blocks):
        """Process text blocks to identify rooms and components"""
        for block in text_blocks:
            text = block['text'].upper()
            
            # Simple keyword matching to categorize text
            node_type = None
            if any(kw in text for kw in ['ROOM', 'OFFICE', 'HALL']):
                node_type = 'room'
            elif any(kw in text for kw in ['HVAC', 'AC', 'HEATING']):
                node_type = 'hvac'
            elif any(kw in text for kw in ['PANEL', 'ELECTRIC']):
                node_type = 'electrical'
                
            if node_type:
                # Add node with placeholder features
                G.add_node(
                    len(G.nodes), 
                    type=node_type,
                    label=block['text'],
                    position=(block.get('x', 0) + block.get('width', 0)/2, 
                              block.get('y', 0) + block.get('height', 0)/2),
                    features=np.zeros(self.node_feature_dims[node_type])
                )
    
    def _process_vector_graphics(self, G, vector_graphics):
        """Process vector graphics to identify connections"""
        # Placeholder for processing vector paths
        pass
    
    def _process_lines(self, G, lines):
        """Process detected lines to identify connections"""
        # Find nodes that might be connected by lines
        nodes = list(G.nodes(data=True))
        
        for line in lines:
            start, end = line['start'], line['end']
            
            # Find closest nodes to line endpoints
            closest_to_start = self._find_closest_node(nodes, start)
            closest_to_end = self._find_closest_node(nodes, end)
            
            if closest_to_start is not None and closest_to_end is not None:
                if closest_to_start != closest_to_end:
                    # Determine edge type based on connected node types
                    edge_type = self._determine_edge_type(
                        G.nodes[closest_to_start]['type'], 
                        G.nodes[closest_to_end]['type']
                    )
                    
                    # Add edge with placeholder features
                    G.add_edge(
                        closest_to_start, 
                        closest_to_end, 
                        type=edge_type,
                        features=np.zeros(self.edge_feature_dims[edge_type])
                    )
    
    def _process_cad_entities(self, G, entities, layers):
        """Process CAD entities to build graph"""
        # First pass: identify nodes (rooms, components)
        for entity in entities:
            if entity['type'] == 'TEXT':
                text = entity.get('text', '').upper()
                
                # Categorize text similar to _process_text_blocks
                node_type = None
                if any(kw in text for kw in ['ROOM', 'OFFICE', 'HALL']):
                    node_type = 'room'
                elif any(kw in text for kw in ['HVAC', 'AC', 'HEATING']):
                    node_type = 'hvac'
                elif any(kw in text for kw in ['PANEL', 'ELECTRIC']):
                    node_type = 'electrical'
                    
                if node_type:
                    G.add_node(
                        entity['handle'],
                        type=node_type,
                        label=entity.get('text', ''),
                        position=entity.get('position', (0, 0, 0)),
                        features=np.zeros(self.node_feature_dims[node_type])
                    )
        
        # Second pass: identify connections
        for entity in entities:
            if entity['type'] == 'LINE':
                # Find nodes near line endpoints
                # This would use spatial indexing in a real implementation
                start, end = entity['start'], entity['end']
                # Placeholder for finding and connecting nodes
    
    def _process_visio_shapes(self, G, shapes):
        """Process Visio shapes to identify nodes"""
        # Placeholder for processing Visio shapes
        pass
    
    def _process_visio_connectors(self, G, connectors):
        """Process Visio connectors to identify edges"""
        # Placeholder for processing Visio connectors
        pass
    
    def _find_closest_node(self, nodes, point, max_distance=50):
        """Find node closest to a point"""
        closest_node = None
        min_distance = max_distance
        
        for node_id, node_data in nodes:
            pos = node_data.get('position', (0, 0))
            dist = np.sqrt((pos[0] - point[0])**2 + (pos[1] - point[1])**2)
            
            if dist < min_distance:
                min_distance = dist
                closest_node = node_id
                
        return closest_node
    
    def _determine_edge_type(self, node1_type, node2_type):
        """Determine edge type based on connected node types"""
        # Simple rules to determine edge type
        if node1_type == 'room' and node2_type == 'room':
            return 'structural'
        elif 'hvac' in (node1_type, node2_type) and 'room' in (node1_type, node2_type):
            return 'airflow'
        elif 'electrical' in (node1_type, node2_type):
            return 'electrical'
        else:
            return 'structural'  # Default
    
    def enhance_with_simulation_data(self, G, simulation_data):
        """Enhance graph with Sinergym simulation data"""
        # Placeholder for matching simulation data to graph nodes/edges
        if 'time_series' in simulation_data:
            for record in simulation_data['time_series']:
                # Match data to nodes based on naming conventions
                # This is highly dependent on simulation output format
                pass
        
        return G
    
    def enhance_with_structural_data(self, G, structural_data):
        """Enhance graph with structural analysis data"""
        # Placeholder for matching structural analysis to graph elements
        return G
    
    def generate_graph_features(self, G):
        """Generate features for nodes and edges based on available data"""
        for node, data in G.nodes(data=True):
            # Generate appropriate features based on node type
            node_type = data.get('type')
            if node_type == 'room':
                # Example features for rooms
                features = np.zeros(self.node_feature_dims['room'])
                # Would set various features based on room properties
                # e.g., dimensions, occupancy, etc.
                G.nodes[node]['features'] = features
            
            # Similar for other node types
            
        for u, v, data in G.edges(data=True):
            # Generate appropriate features based on edge type
            edge_type = data.get('type')
            if edge_type == 'structural':
                # Example features for structural edges
                features = np.zeros(self.edge_feature_dims['structural'])
                # Would set various features based on structural properties
                # e.g., load capacity, material, etc.
                G[u][v]['features'] = features
            
            # Similar for other edge types
            
        return G
    
    def convert_to_pytorch_geometric(self, G):
        """Convert networkx graph to PyTorch Geometric Data object"""
        # Extract node types and features
        node_types = []
        node_features = {
            'room': [],
            'hvac': [],
            'electrical': []
        }
        
        # Track mapping from node_id to index in the PyG tensor
        node_id_to_idx = {}
        
        for i, (node_id, data) in enumerate(G.nodes(data=True)):
            node_id_to_idx[node_id] = i
            node_type = data.get('type')
            node_types.append(['room', 'hvac', 'electrical'].index(node_type))
            
            # Add features to appropriate type list
            if node_type in node_features:
                node_features[node_type].append(data.get('features', np.zeros(self.node_feature_dims[node_type])))
        
        # Convert to tensors
        node_types = torch.tensor(node_types, dtype=torch.long)
        for key in node_features:
            if node_features[key]:
                node_features[key] = torch.tensor(np.stack(node_features[key]), dtype=torch.float)
            else:
                # Empty tensor with correct shape if no nodes of this type
                node_features[key] = torch.zeros((0, self.node_feature_dims[key]), dtype=torch.float)
        
        # Extract edge information
        edge_index = []
        edge_types = []
        edge_features = {
            'structural': [],
            'pipes': [],
            'electrical': [],
            'airflow': []
        }
        
        for u, v, data in G.edges(data=True):
            edge_index.append([node_id_to_idx[u], node_id_to_idx[v]])
            edge_type = data.get('type')
            edge_types.append(['structural', 'pipes', 'electrical', 'airflow'].index(edge_type))
            
            # Add features to appropriate type list
            if edge_type in edge_features:
                edge_features[edge_type].append(data.get('features', np.zeros(self.edge_feature_dims[edge_type])))
        
        # Convert to tensors
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_types = torch.tensor(edge_types, dtype=torch.long)
            
            for key in edge_features:
                if edge_features[key]:
                    edge_features[key] = torch.tensor(np.stack(edge_features[key]), dtype=torch.float)
                else:
                    # Empty tensor with correct shape if no edges of this type
                    edge_features[key] = torch.zeros((0, self.edge_feature_dims[key]), dtype=torch.float)
        else:
            # Empty graph
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_types = torch.zeros(0, dtype=torch.long)
            for key in edge_features:
                edge_features[key] = torch.zeros((0, self.edge_feature_dims[key]), dtype=torch.float)
        
        # Create placeholder values for basis functions
        edge_values = torch.ones(edge_index.size(1), dtype=torch.float)
        edge_capacities = torch.ones(edge_index.size(1), dtype=torch.float)
        
        # Create PyG Data object
        data = Data(
            node_features=node_features,
            node_types=node_types,
            edge_index=edge_index,
            edge_types=edge_types,
            edge_features=edge_features,
            edge_values=edge_values,
            edge_capacities=edge_capacities,
            # Single building, so all nodes are in batch 0
            batch=torch.zeros(len(G.nodes), dtype=torch.long)
        )
        
        return data


class BuildingDatasetProcessor:
    """Process multiple buildings into a dataset for training"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.graph_processor = BuildingGraphProcessor(config)
    
    def process_building(self, blueprint_data, simulation_data=None, structural_data=None):
        """Process a single building's data"""
        # Create initial graph from blueprint
        G = self.graph_processor.process_blueprint(blueprint_data)
        
        # Enhance with simulation data if available
        if simulation_data is not None:
            G = self.graph_processor.enhance_with_simulation_data(G, simulation_data)
        
        # Enhance with structural data if available
        if structural_data is not None:
            G = self.graph_processor.enhance_with_structural_data(G, structural_data)
        
        # Generate features based on available data
        G = self.graph_processor.generate_graph_features(G)
        
        # Convert to PyG Data object
        data = self.graph_processor.convert_to_pytorch_geometric(G)
        
        return data
    
    def process_dataset(self, building_data_list):
        """
        Process multiple buildings into a dataset
        
        Args:
            building_data_list: List of dicts with keys 'blueprint', 'simulation', 'structural'
            
        Returns:
            List of PyG Data objects
        """
        dataset = []
        
        for building_data in building_data_list:
            data = self.process_building(
                building_data.get('blueprint'),
                building_data.get('simulation'),
                building_data.get('structural')
            )
            dataset.append(data)
        
        return dataset
    
    def create_batch(self, dataset):
        """Create a batch from multiple building data objects"""
        return Batch.from_data_list(dataset)
    
    def split_dataset(self, dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Split dataset into training, validation and test sets"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        n = len(dataset)
        indices = np.random.permutation(n)
        
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_dataset = [dataset[i] for i in train_indices]
        val_dataset = [dataset[i] for i in val_indices]
        test_dataset = [dataset[i] for i in test_indices]
        
        return train_dataset, val_dataset, test_dataset