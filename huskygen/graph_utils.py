from __future__ import annotations

from typing import List

import networkx as nx

# TODO: Find all generic/overly broad species tags
ignored_paths = {'canis', 'canid', 'mammal', 'felid', 'marine', 'felis', 'equid'}


def get_processed_chain(graph: nx.DiGraph, start: str) -> List[str]:
    try:
        chain = [start] + [v for u, v in nx.bfs_edges(graph, start)]
    except (nx.NetworkXNoPath, nx.NetworkXError) as e:
        # print(e)
        chain = [start]
    return [c for c in chain if c not in ignored_paths]
