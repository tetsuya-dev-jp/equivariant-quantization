"""Equivariant quantization pilot package.

Data/oracle modules import only the standard library so offline data tests work
without PyTorch or Transformers. Evaluation modules import those packages only
when the user runs model scoring.
"""

__version__ = "0.1.0"
