import hashlib
import os
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from graphviz import Digraph

class PromptNode:
    """
    Represents a single node in the prompt graph.
    Each node contains text (prompt fragment) and connections to child nodes.
    """
    def __init__(self, name: str, text: str = "", metadata: Dict[str, Any] = None):
        self.name = name
        self.text = text  # the actual prompt fragment
        self.kids = []    # downstream nodes
        self.hash = hashlib.md5(text.encode()).hexdigest()
        self.metadata = metadata or {}
        self.response = None  # Store LLM response to this node if it's a leaf
    
    def add_child(self, node):
        """Add a child node and return it for fluent API"""
        self.kids.append(node)
        return node
    
    def to_dict(self):
        """Convert node to dictionary for serialization"""
        return {
            "name": self.name,
            "text": self.text,
            "hash": self.hash,
            "metadata": self.metadata,
            "response": self.response,
            "children": [kid.hash for kid in self.kids]
        }

class PromptGraph:
    """
    Represents the entire graph of prompt nodes.
    Manages the connections between nodes and provides utilities for visualization.
    """
    def __init__(self, cache_dir: str):
        self.root_nodes = []
        self.cache_dir = cache_dir
        self.nodes_by_hash = {}  # For quick lookups
        self.timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
    
    def add_root(self, node: PromptNode):
        """Add a root node to the graph"""
        self.root_nodes.append(node)
        self.nodes_by_hash[node.hash] = node
        return node
    
    def add_node(self, node: PromptNode):
        """Add a non-root node to the graph for tracking"""
        self.nodes_by_hash[node.hash] = node
        return node
    
    def get_node(self, node_hash: str) -> Optional[PromptNode]:
        """Retrieve a node by its hash"""
        return self.nodes_by_hash.get(node_hash)
    
    def render(self):
        """Generate a visualization of the prompt graph using Graphviz"""
        try:
            g = Digraph()
            
            def visit(n):
                # Truncate text for display
                display_text = n.text[:40] + "..." if len(n.text) > 40 else n.text
                g.node(n.hash[:6], f"{n.name}\n{display_text}")
                for k in n.kids:
                    g.edge(n.hash[:6], k.hash[:6])
                    visit(k)
            
            for r in self.root_nodes:
                visit(r)
            
            output_path = os.path.join(self.cache_dir, f"prompt_graph_{self.timestamp}")
            g.render(output_path, format="png", view=False)
            return output_path + ".png"
        except ImportError:
            print("Graphviz not installed. Install with: pip install graphviz")
            return None
    
    def save(self):
        """Save the graph to a JSON file"""
        nodes_dict = {}
        
        # First collect all nodes
        for node_hash, node in self.nodes_by_hash.items():
            nodes_dict[node_hash] = node.to_dict()
        
        # Create the full graph representation
        graph_dict = {
            "timestamp": self.timestamp,
            "root_nodes": [node.hash for node in self.root_nodes],
            "nodes": nodes_dict
        }
        
        # Save to file
        output_path = os.path.join(self.cache_dir, f"prompt_graph_{self.timestamp}.json")
        with open(output_path, 'w') as f:
            json.dump(graph_dict, f, indent=2)
        
        return output_path
    
    @classmethod
    def load(cls, filepath: str):
        """Load a graph from a JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Create new graph
        cache_dir = os.path.dirname(filepath)
        graph = cls(cache_dir)
        graph.timestamp = data.get("timestamp", graph.timestamp)
        
        # Create all nodes first
        nodes = {}
        for node_hash, node_data in data["nodes"].items():
            node = PromptNode(
                name=node_data["name"],
                text=node_data["text"],
                metadata=node_data.get("metadata", {})
            )
            node.response = node_data.get("response")
            nodes[node_hash] = node
            graph.nodes_by_hash[node_hash] = node
        
        # Set up connections
        for node_hash, node_data in data["nodes"].items():
            node = nodes[node_hash]
            for child_hash in node_data.get("children", []):
                if child_hash in nodes:
                    node.kids.append(nodes[child_hash])
        
        # Set root nodes
        for root_hash in data.get("root_nodes", []):
            if root_hash in nodes:
                graph.root_nodes.append(nodes[root_hash])
        
        return graph
    
    def get_combined_prompt(self, leaf_node_hash: str) -> str:
        """
        Get the combined prompt by walking up from a leaf node.
        This traverses backwards through parents to collect all relevant prompts.
        """
        # This would require parent references which we don't currently have
        # For now, just return the prompt of the specified node
        node = self.get_node(leaf_node_hash)
        if node:
            return node.text
        return "" 