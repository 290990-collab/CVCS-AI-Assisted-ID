# Graph Builder for Creating and Manipulating Graphs
# Nodes = rooms
# Edges = adjacencies (walls, doors, ...)
# Node features = room properties (area, type, ...)

"""
Output:
graph = {
    "nodes": [...],
    "edges": [...]
}
"""

def build_graph(raw_sample):
    """
    raw_sample → dati originali CAD

    Returns:
        dict con:
        - nodes
        - edges
    """

    graph = {
        "nodes": [],
        "edges": []
    }

    # TODO: implementare parsing reale

    return graph