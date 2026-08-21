"""spiral_hk — spiral-of-silence opinion dynamics toolkit.

Models
------
- Hegselmann-Krause (HK) bounded-confidence baseline on graphs
- Inertial HK (IHK, Chazelle & Wang 2017)
- DWHK: dynamical-weight HK grounded in the spiral of silence
  (Ruan, IEEE UV 2022, doi: 10.1109/UV56588.2022.10185469)
"""

from spiral_hk.graph import Graph, complete_graph, barabasi_albert, watts_strogatz

__all__ = ["Graph", "complete_graph", "barabasi_albert", "watts_strogatz"]
__version__ = "0.1.0"
