import json
import jsonschema
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "polyline": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2
                        }
                    },
                    "attributes": {
                        "type": "object",
                        "properties": {
                            "direction": {"type": "string"},
                            "curvature": {"type": "number"},
                            "confidence": {"type": "number"},
                            "n_observations": {"type": "number"},
                            "length_m": {"type": "number"},
                            "mean_range_m": {"type": "number"}
                        },
                        "required": ["direction"]
                    }
                },
                "required": ["id", "polyline", "attributes"]
            }
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "type": {"type": "string", "enum": ["continuation", "merge", "split", "junction"]},
                    "confidence": {"type": "number"}
                },
                "required": ["from", "to", "type"]
            }
        }
    },
    "required": ["nodes", "edges"]
}

@dataclass
class NodeAttributes:
    direction: str
    curvature: Optional[float] = None
    confidence: Optional[float] = None
    n_observations: Optional[int] = None
    length_m: Optional[float] = None
    mean_range_m: Optional[float] = None

@dataclass
class GraphNode:
    id: str
    polyline: List[List[float]]
    attributes: NodeAttributes

@dataclass
class GraphEdge:
    from_node: str
    to_node: str
    type: str
    confidence: Optional[float] = None

@dataclass
class TopologyGraph:
    nodes: List[GraphNode]
    edges: List[GraphEdge]

    def to_dict(self):
        return {
            "nodes": [
                {
                    "id": n.id,
                    "polyline": n.polyline,
                    "attributes": {k: v for k, v in asdict(n.attributes).items() if v is not None}
                } for n in self.nodes
            ],
            "edges": [
                {
                    "from": e.from_node,
                    "to": e.to_node,
                    "type": e.type,
                    **({"confidence": e.confidence} if e.confidence is not None else {})
                } for e in self.edges
            ]
        }

def validate_and_save(graph: TopologyGraph, path: str):
    data = graph.to_dict()
    jsonschema.validate(instance=data, schema=SCHEMA)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
