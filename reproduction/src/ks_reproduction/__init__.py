"""PyTorch reproduction of the Krusell--Smith thesis experiments."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ks-thesis-reproduction")
except PackageNotFoundError:  # source checkout without an installed package
    __version__ = "0.1.0"

__all__ = ["__version__"]
