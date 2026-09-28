import numpy as np
import yaml
from pathlib import Path
from src.graph.topology import TopologyBuilder, TopologyGraph
from src.graph.schema import GraphNode, GraphEdge, NodeAttributes

def test_topology_edges(tmp_path: Path):
    config = {
        'topology': {
            'min_lane_width': 2.0,
            'max_lane_width': 5.0,
            'min_overlap_m': 3.0,
            'segment_length_m': 5.0,
            'continuation_gate_dist': 2.0,
            'continuation_gate_angle_deg': 30.0,
            'split_merge_radius': 3.0,
            'split_merge_angle_deg': 45.0,
            'junction_search_radius': 10.0,
            'junction_angle_deg': 90.0
        }
    }
    cfg_file = tmp_path / "config.yaml"
    with open(cfg_file, 'w') as f:
        yaml.dump(config, f)
        
    builder = TopologyBuilder(str(cfg_file))
    
    # Manually create nodes to test edges
    # Continuation
    n1 = GraphNode(id="n1", polyline=[[0,0], [5,0]], attributes=NodeAttributes(direction="0"))
    n2 = GraphNode(id="n2", polyline=[[5.5,0], [10,0]], attributes=NodeAttributes(direction="0"))
    
    # Split
    # n3 splits into n4 and n5
    n3 = GraphNode(id="n3", polyline=[[0,10], [5,10]], attributes=NodeAttributes(direction="0"))
    n4 = GraphNode(id="n4", polyline=[[6,11], [10,12]], attributes=NodeAttributes(direction="0"))
    n5 = GraphNode(id="n5", polyline=[[6,9], [10,8]], attributes=NodeAttributes(direction="0"))
    
    # Merge
    # n6 and n7 merge into n8
    n6 = GraphNode(id="n6", polyline=[[0,20], [5,19]], attributes=NodeAttributes(direction="0"))
    n7 = GraphNode(id="n7", polyline=[[0,18], [5,18.5]], attributes=NodeAttributes(direction="0"))
    n8 = GraphNode(id="n8", polyline=[[6,18.75], [10,18.75]], attributes=NodeAttributes(direction="0"))
    
    # Junction
    # n9 to n10 (gap too large for continuation, angle allows junction)
    n9 = GraphNode(id="n9", polyline=[[0,30], [5,30]], attributes=NodeAttributes(direction="0"))
    n10 = GraphNode(id="n10", polyline=[[13,32], [18,32]], attributes=NodeAttributes(direction="0"))
    
    nodes = [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10]
    
    edges = builder.build_edges(nodes)
    edge_types = {(e.from_node, e.to_node): e.type for e in edges}
    
    assert edge_types.get(("n1", "n2")) == "continuation"
    assert edge_types.get(("n3", "n4")) == "split"
    assert edge_types.get(("n3", "n5")) == "split"
    assert edge_types.get(("n6", "n8")) == "merge"
    assert edge_types.get(("n7", "n8")) == "merge"
    assert edge_types.get(("n9", "n10")) == "junction"
