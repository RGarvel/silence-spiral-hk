"""Graph container and generators.

The simulation works on *directed arcs*: arc (i -> j) means "i listens to
j", and its weight ``w[i -> j]`` is j's influence on i. Bidirectional
graphs (the paper's WBD) store both directions; the two directions may
carry different weights once the spiral-of-silence update runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Graph:
    """A social graph stored as directed arc arrays.

    Attributes
    ----------
    n : number of agents (nodes are 0 .. n-1)
    src : int array of shape (m,) — listener of each arc
    dst : int array of shape (m,) — speaker of each arc
    name : optional label for plots
    """

    n: int
    src: np.ndarray
    dst: np.ndarray
    name: str = "graph"

    @property
    def m(self) -> int:
        """Number of directed arcs."""
        return self.src.shape[0]

    @property
    def degree(self) -> np.ndarray:
        """Full degree of every node.

        For bidirectional graphs each undirected edge contributes exactly
        one outgoing arc per endpoint, so the out-degree equals the
        undirected degree used by the DWHK weight update (d_i in the paper).
        """
        return np.bincount(self.src, minlength=self.n)


def _bidirect(n: int, edges, name: str) -> Graph:
    """Build a bidirectional Graph from an iterable of undirected edges."""
    edges = np.asarray(list(edges), dtype=np.int64).reshape(-1, 2)
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    return Graph(n=n, src=src, dst=dst, name=name)


def complete_graph(n: int) -> Graph:
    """Fully mixed population (all-to-all, no self-loops)."""
    i, j = np.triu_indices(n, k=1)
    return _bidirect(n, np.stack([i, j], axis=1), name=f"complete({n})")


def barabasi_albert(n: int, m_attach: int, seed: int = 0) -> Graph:
    """Scale-free network (Barabasi-Albert preferential attachment)."""
    import networkx as nx

    g = nx.barabasi_albert_graph(n, m_attach, seed=seed)
    return _bidirect(n, list(g.edges()), name=f"BA({n},{m_attach})")


def watts_strogatz(n: int, k: int, p: float, seed: int = 0) -> Graph:
    """Small-world network (Watts-Strogatz), ring lattice degree k."""
    import networkx as nx

    g = nx.watts_strogatz_graph(n, k, p, seed=seed)
    return _bidirect(n, list(g.edges()), name=f"WS({n},{k},{p})")


def load_snap_edge_file(path: str, name: str = "ego-facebook") -> Graph:
    """Load a SNAP-style edge list (whitespace-separated 'u v' lines).

    Node ids are remapped to 0..n-1. Used for the ego-Facebook dataset
    (4039 nodes / 88234 edges); see README for the download URL.
    """
    edges = []
    ids: dict[int, int] = {}

    def remap(u: int) -> int:
        if u not in ids:
            ids[u] = len(ids)
        return ids[u]

    opener = open
    if str(path).endswith(".gz"):
        import gzip

        opener = gzip.open
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            u, v = line.split()[:2]
            edges.append((remap(int(u)), remap(int(v))))
    return _bidirect(len(ids), edges, name=name)
